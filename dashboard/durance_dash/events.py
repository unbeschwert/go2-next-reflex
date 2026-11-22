"""Server-Sent Events broker (one stream per browser tab) and background tasks with logs."""
from __future__ import annotations

import asyncio
import itertools
import json
import time
import traceback
from typing import Any, Awaitable, Callable


class Broker:
    QUEUE = 500                      # events buffered per client before it counts as too slow

    def __init__(self):
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def publish(self, event: str, data: Any) -> None:
        """Thread-safe: may be called from worker threads as well as from the event loop."""
        if self.loop is None:
            return
        msg = f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            self._put(msg)
        elif not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._put, msg)

    def _put(self, msg: str) -> None:
        for q in list(self.clients):
            if q.qsize() >= self.QUEUE:          # slow client: drop it, the browser reconnects
                self.clients.discard(q)
                q.put_nowait(None)
                continue
            q.put_nowait(msg)

    async def stream(self, is_disconnected: Callable[[], Awaitable[bool]]):
        q: asyncio.Queue = asyncio.Queue()
        self.clients.add(q)
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    if await is_disconnected():
                        break
                    yield ": keep-alive\n\n"
                    continue
                if msg is None:
                    break
                yield msg
        finally:
            self.clients.discard(q)


class Task:
    MAX_LOG = 2000

    def __init__(self, tid: str, kind: str, label: str, user: str):
        self.id, self.kind, self.label, self.user = tid, kind, label, user
        self.state, self.started, self.finished = "running", time.time(), None
        self.lines: list[str] = []
        self.result: Any = None

    def view(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "user": self.user, "state": self.state,
                "started": self.started, "finished": self.finished, "log": self.lines[-400:], "result": self.result}


class Tasks:
    KEEP = 100

    def __init__(self, broker: Broker):
        self.broker = broker
        self.items: dict[str, Task] = {}
        self.ids = itertools.count(1)
        self.running_kinds: dict[str, str] = {}      # exclusive kind -> task id

    def start(self, kind: str, label: str, user: str, fn: Callable[[Callable[[str], None]], Awaitable[Any]],
              exclusive: str | None = None) -> Task:
        """Run `fn(log)` in the background. `exclusive`: refuse if a task of that group is still running."""
        if exclusive and exclusive in self.running_kinds:
            other = self.items.get(self.running_kinds[exclusive])
            if other and other.state == "running":
                from fastapi import HTTPException
                raise HTTPException(409, f"'{other.label}' is still running (task {other.id})")
        t = Task(f"t{next(self.ids)}", kind, label, user)
        self.items[t.id] = t
        if exclusive:
            self.running_kinds[exclusive] = t.id
        while len(self.items) > self.KEEP:
            self.items.pop(next(iter(self.items)))

        def log(line: str) -> None:
            for part in str(line).splitlines() or [""]:
                t.lines.append(part[:2000])
            del t.lines[:-Task.MAX_LOG]
            self.broker.publish("task", t.view())

        async def runner():
            try:
                t.result = await fn(log)
                t.state = "done"
            except Exception as e:  # noqa: BLE001 - every failure must end up in the task log
                t.state = "failed"
                detail = getattr(e, "detail", None) or str(e) or type(e).__name__
                log(f"ERROR: {detail}")
                if not hasattr(e, "detail"):
                    log(traceback.format_exc(limit=3))
            finally:
                t.finished = time.time()
                if exclusive and self.running_kinds.get(exclusive) == t.id:
                    self.running_kinds.pop(exclusive, None)
                self.broker.publish("task", t.view())

        asyncio.get_running_loop().create_task(runner())
        self.broker.publish("task", t.view())
        return t

    def get(self, tid: str) -> Task | None:
        return self.items.get(tid)

    def all(self) -> list[dict]:
        return [t.view() for t in reversed(self.items.values())]
