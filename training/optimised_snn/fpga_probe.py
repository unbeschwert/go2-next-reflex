#!/usr/bin/env python3
"""Send an arbitrary frame sequence to the board's SPLEAT server and save the FPGA's outputs.

    python3 fpga_probe.py frames.npy outputs.npz [--host 192.168.77.1] [--port 3333]

frames.npy: float32 [N, 2, S, S] (one request per frame, in order). outputs.npz: `out` [N, n_out]
(spikes per frame) and `latency_us` [N]. Used to probe the hardware's arithmetic with test patterns.
The camera client must not be feeding the server at the same time.
"""
import socket
import struct
import sys
import time

import numpy as np


def cbor_uint(n):
    if n < 24:
        return bytes([n])
    if n < 1 << 8:
        return b"\x18" + bytes([n])
    if n < 1 << 16:
        return b"\x19" + struct.pack(">H", n)
    return b"\x1a" + struct.pack(">I", n)


def request(msg_id, frame):
    payload = np.ascontiguousarray(frame, dtype="<f4").tobytes()
    head = cbor_uint(len(payload))
    body = b"\xa2\x62id" + cbor_uint(msg_id) + b"\x67payload" + bytes([head[0] | 0x40]) + head[1:] + payload
    return struct.pack(">I", len(body)) + body


def read_uint(buf, pos):
    b = buf[pos]
    if b < 24:
        return b
    return {0x18: lambda: buf[pos + 1], 0x19: lambda: struct.unpack(">H", buf[pos + 1:pos + 3])[0],
            0x1a: lambda: struct.unpack(">I", buf[pos + 1:pos + 5])[0],
            0x1b: lambda: struct.unpack(">Q", buf[pos + 1:pos + 9])[0]}[b]()


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf += chunk
    return bytes(buf)


def run(frames, host="192.168.77.1", port=3333, first_id=2_000_000):
    sock = socket.create_connection((host, port), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    outs, lats = [], []
    for i, frame in enumerate(frames):
        msg_id = first_id + i
        sock.sendall(request(msg_id, frame))
        while True:
            body = recv_exact(sock, struct.unpack(">I", recv_exact(sock, 4))[0])
            j = body.find(b"\x62id")
            if j >= 0 and read_uint(body, j + 3) == msg_id:
                break
        k = body.find(b"\x67payload")
        h = body[k + 8]
        n, off = (h - 0x40, k + 9) if h < 0x58 else (body[k + 9], k + 10)
        outs.append(np.frombuffer(body[off:off + n], "<f4").copy())
        m = body.find(b"\x67latency")
        lats.append(read_uint(body, m + 8) if m >= 0 else 0)
    sock.close()
    return np.stack(outs), np.array(lats)


if __name__ == "__main__":
    frames = np.load(sys.argv[1])
    kw = {}
    if "--host" in sys.argv:
        kw["host"] = sys.argv[sys.argv.index("--host") + 1]
    if "--port" in sys.argv:
        kw["port"] = int(sys.argv[sys.argv.index("--port") + 1])
    t0 = time.time()
    out, lat = run(frames, **kw)
    np.savez_compressed(sys.argv[2], out=out, latency_us=lat)
    print(f"{len(frames)} frames in {time.time() - t0:.1f}s -> {sys.argv[2]}; output rate per neuron {(out > 0).mean(0).round(3)}; "
          f"median latency {np.median(lat) / 1000:.2f} ms")
