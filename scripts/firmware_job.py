"""Generate SPLEAT firmware on riversdk.dev from the command line: submit, monitor, download.

The firmware depends only on model.json (layer shapes) and the form options (board, Prophesee IP,
weight/potential formats, strategy), not on the weights (those go in weights.txt at run time). So
`submit` first looks for a finished job with the same model.json and options and reuses it
(--force submits anyway).

Login: copy the value of the `__Host-riversdk_session` cookie from your browser (DevTools → Storage →
Cookies → riversdk.dev) into ~/.config/riversdk/session, or set RIVERSDK_SESSION. Never commit it.

Usage (from the repo root):
    uv run scripts/firmware_job.py list
    uv run scripts/firmware_job.py submit --run runs/motion-buddy-40-e300          # uses deploy/model.json + formats
    uv run scripts/firmware_job.py submit --run runs/X --wait --download           # all in one
    uv run scripts/firmware_job.py submit --model model.json --weights Q2.6 --potentials Q4.12 --dry-run
    uv run scripts/firmware_job.py status 29
    uv run scripts/firmware_job.py wait 29 [--interval 30]
    uv run scripts/firmware_job.py download 29 --out runs/X/firmware-job29        # checks SHA256, unpacks
"""
from __future__ import annotations

import argparse
import hashlib
import html
import http.cookiejar
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = "https://riversdk.dev"
COOKIE = "__Host-riversdk_session"
SESSION_FILE = Path.home() / ".config" / "riversdk" / "session"
STRATEGIES = ("better-latency", "less-resources", "faster-generation")
STRATEGY_LABELS = {"Better latency": "better-latency", "Less resources": "less-resources",
                   "Faster generation": "faster-generation"}


class RiverError(RuntimeError):
    pass


# --- HTTP ---------------------------------------------------------------------------------------

class Client:
    def __init__(self, session: str):
        jar = http.cookiejar.CookieJar()
        jar.set_cookie(http.cookiejar.Cookie(
            0, COOKIE, session, None, False, "riversdk.dev", False, False, "/", True, True, None, False,
            None, None, {}))
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self.opener.addheaders = [("User-Agent", "go2-next-reflex-firmware-client/1.0")]

    def request(self, path: str, data: bytes | None = None, headers: dict | None = None):
        req = urllib.request.Request(BASE + path, data=data, headers=headers or {})
        try:
            resp = self.opener.open(req, timeout=120)
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            raise RiverError(f"{req.get_method()} {path}: HTTP {error.code}\n{page_text(body)[:600]}") from None
        if "/login" in resp.geturl() or "/auth" in resp.geturl():
            raise RiverError("not logged in: the session cookie is missing or expired (see --help)")
        return resp

    def get(self, path: str) -> str:
        return self.request(path).read().decode()


def load_session() -> str:
    value = os.environ.get("RIVERSDK_SESSION") or (SESSION_FILE.read_text().strip() if SESSION_FILE.exists() else "")
    if not value:
        raise RiverError(f"no session cookie: put it in {SESSION_FILE} or set RIVERSDK_SESSION (see --help)")
    return value.removeprefix(COOKIE + "=")


def page_text(page: str) -> str:
    main = re.search(r"<main.*?</main>", page, re.S)
    text = re.sub(r"<[^>]+>", " ", main.group(0) if main else page)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def multipart(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = []
    for name, value in fields:
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, filename, content, ctype in files:
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                   f"Content-Type: {ctype}\r\n\r\n".encode() + content + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


# --- Pages --------------------------------------------------------------------------------------

def list_jobs(client: Client) -> list[dict]:
    page = client.get("/jobs")
    jobs = []
    for row in re.findall(r"<tr[^>]*cursor-pointer.*?</tr>", page, re.S):
        job = re.search(r'href="/jobs/(\d+)"[^>]*title="([^"]*)"', row)
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if job:
            jobs.append({"id": int(job.group(1)), "model": html.unescape(job.group(2)), "created": cells[0],
                         "description": cells[2] if len(cells) > 2 else "", "status": cells[4] if len(cells) > 4 else ""})
    return jobs


def job_info(client: Client, job_id: int) -> dict:
    page = client.get(f"/jobs/{job_id}")
    fields = {}
    for dt, dd in re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", page, re.S):
        key = re.sub(r"<[^>]+>", "", dt).strip()
        fields[key] = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", dd))).strip()
    name = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
    status = fields.get("Status")
    if not status:   # while building the page may not have the <dl> yet: show the page text instead
        text = page_text(page)
        after = text.split(html.unescape(re.sub(r"<[^>]+>", "", name.group(1))).strip(), 1)[-1] if name else text
        status = after.replace("Back to jobs", "").strip()[:80] or "?"
    lowered = status.lower()
    state = ("done" if f'href="/jobs/{job_id}/firmware"' in page and fields.get("SHA256")
             else "failed" if any(w in lowered for w in ("fail", "error", "✗", "cancel", "abort"))
             else "running")
    return {"id": job_id, "name": html.unescape(re.sub(r"<[^>]+>", "", name.group(1))).strip() if name else "",
            "status": status, "state": state, "duration": fields.get("Duration"), "sha256": fields.get("SHA256"),
            "firmware_file": fields.get("Firmware file"), "board": fields.get("Board"),
            "prophesee_ip": fields.get("Prophesee IP"), "weights": fields.get("Weights quantization"),
            "potentials": fields.get("Potentials quantization"),
            "strategy": STRATEGY_LABELS.get(fields.get("Implementation strategy", ""), fields.get("Implementation strategy")),
            "page": page_text(page) if state == "failed" else None}


def same_model(a: bytes, b: bytes) -> bool:
    try:
        return json.loads(a) == json.loads(b)
    except ValueError:
        return a == b


def find_reusable(client: Client, model: bytes, opts: dict) -> dict | None:
    """A finished job with the same model.json and the same options, newest first."""
    for job in sorted(list_jobs(client), key=lambda j: -j["id"]):
        info = job_info(client, job["id"])
        if info["state"] != "done":
            continue
        wanted = {"board": opts["board"].upper(), "weights": f"Q{opts['weights_integer_bits']}.{opts['weights_fractional_bits']}",
                  "potentials": f"Q{opts['potentials_integer_bits']}.{opts['potentials_fractional_bits']}",
                  "prophesee_ip": {"none": "None", "genx320": "GenX320"}[opts["prophesee_ip"]],
                  "strategy": opts["strategy"]}
        if all(str(info[k]).lower() == str(v).lower() for k, v in wanted.items() if k != "strategy" or opts["strategy_matters"]):
            if same_model(client.request(f"/jobs/{job['id']}/model").read(), model):
                return info
    return None


# --- Commands -----------------------------------------------------------------------------------

def parse_q(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"Q?(\d+)\.(\d+)", text.strip())
    if not m:
        raise RiverError(f"format must look like Q2.6, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def submit_options(args) -> tuple[bytes, str, dict]:
    run = Path(args.run) if args.run else None
    model_path = Path(args.model) if args.model else (run / "deploy" / "model.json" if run else None)
    if not model_path or not model_path.exists():
        raise RiverError("give --run runs/<name> (with deploy/model.json) or --model path/to/model.json")
    weights = potentials = None
    strategy = args.strategy
    if run:
        summary = json.loads((run / "deploy" / "export_summary.json").read_text())["formats"]
        weights = (summary["weight_int"], summary["weight_frac"])
        potentials = (summary["potential_int"], summary["potential_frac"])
        if strategy is None and (run / "config.yaml").exists():
            m = re.search(r"^\s*strategy:\s*(\S+)", (run / "config.yaml").read_text(), re.M)
            strategy = m.group(1) if m else None
    weights = parse_q(args.weights) if args.weights else weights
    potentials = parse_q(args.potentials) if args.potentials else potentials
    if not weights or not potentials:
        raise RiverError("weight/potential formats unknown: give --weights Q2.6 --potentials Q4.12 (or --run)")
    strategy = strategy or "faster-generation"
    if strategy not in STRATEGIES:
        raise RiverError(f"--strategy must be one of {STRATEGIES}")
    name = args.name or (f"{run.name}" if run else model_path.stem)
    description = args.description or (f"{run.name}: model.json from the go2-next-reflex training pipeline" if run
                                       else f"{model_path.name}, submitted with scripts/firmware_job.py")
    opts = {"name": name[:100], "board": args.board, "prophesee_ip": args.prophesee_ip,
            "weights_integer_bits": weights[0], "weights_fractional_bits": weights[1],
            "potentials_integer_bits": potentials[0], "potentials_fractional_bits": potentials[1],
            "strategy": strategy, "description": description[:2000],
            "strategy_matters": args.strategy is not None}
    return model_path.read_bytes(), model_path.name, opts


def cmd_submit(client: Client, args) -> int:
    model, filename, opts = submit_options(args)
    shown = {k: v for k, v in opts.items() if k != "strategy_matters"}
    print("firmware job:", json.dumps(shown, indent=1))
    job_id = None
    if not args.force:
        reuse = find_reusable(client, model, opts)
        if reuse:
            print(f"→ job #{reuse['id']} ({reuse['name']}) already built this model with these options "
                  f"(SHA256 {reuse['sha256'][:12]}…); reusing it. --force submits a new job anyway.")
            job_id = reuse["id"]
    if job_id is None:
        if args.dry_run:
            print("dry run: not submitted")
            return 0
        form = client.get("/jobs/new")
        tokens = {}
        action_form = re.search(r'<form[^>]*action="/jobs"[^>]*>.*?</form>', form, re.S)
        for name in ("_csrf", "_submission_key"):
            m = re.search(rf'name="{name}" value="([^"]*)"', action_form.group(0) if action_form else "")
            if not m:
                raise RiverError(f"could not find {name} in the /jobs/new form (did the page change?)")
            tokens[name] = m.group(1)
        fields = [("_csrf", tokens["_csrf"]), ("_submission_key", tokens["_submission_key"])]
        fields += [(k, str(v)) for k, v in shown.items()]
        body, ctype = multipart(fields, [("model", filename, model, "application/json")])
        resp = client.request("/jobs", data=body, headers={
            "Content-Type": ctype, "Origin": BASE, "Referer": BASE + "/jobs/new", "Sec-Fetch-Site": "same-origin"})
        m = re.search(r"/jobs/(\d+)", resp.geturl())
        if not m:
            raise RiverError(f"submission not accepted (ended at {resp.geturl()}):\n{page_text(resp.read().decode())[:800]}")
        job_id = int(m.group(1))
        print(f"→ submitted: job #{job_id}  {BASE}/jobs/{job_id}")
    if args.wait or args.download:
        info = wait(client, job_id, args.interval, args.timeout)
        if info["state"] != "done":
            return 1
        if args.download:
            out = Path(args.out) if args.out else (Path(args.run) / f"firmware-job{job_id}" if args.run
                                                    else Path(f"firmware-job{job_id}"))
            download(client, job_id, out)
    return 0


def wait(client: Client, job_id: int, interval: float, timeout: float) -> dict:
    start, last = time.time(), None
    while True:
        info = job_info(client, job_id)
        if info["status"] != last:
            print(f"  [{time.strftime('%H:%M:%S')}] job #{job_id}: {info['status']}"
                  + (f" ({info['duration']})" if info.get("duration") else ""), flush=True)
            last = info["status"]
        if info["state"] == "done":
            return info
        if info["state"] == "failed":
            print(f"job #{job_id} failed:\n{info['page'][:1500]}", file=sys.stderr)
            return info
        if time.time() - start > timeout:
            print(f"gave up waiting after {timeout / 60:.0f} min; job #{job_id} is still '{info['status']}'", file=sys.stderr)
            return info
        time.sleep(interval)


def download(client: Client, job_id: int, out: Path) -> Path:
    info = job_info(client, job_id)
    if info["state"] != "done":
        raise RiverError(f"job #{job_id} has no firmware yet: {info['status']}")
    out.mkdir(parents=True, exist_ok=True)
    archive = out / (info["firmware_file"] or "firmware.tar.gz")
    data = client.request(f"/jobs/{job_id}/firmware").read()
    digest = hashlib.sha256(data).hexdigest()
    if info["sha256"] and digest != info["sha256"]:
        raise RiverError(f"SHA256 mismatch: page says {info['sha256']}, download is {digest}")
    archive.write_bytes(data)
    (out / f"job{job_id}-model.json").write_bytes(client.request(f"/jobs/{job_id}/model").read())
    (out / "job.json").write_text(json.dumps({k: v for k, v in info.items() if k != "page"}, indent=1) + "\n")
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
        names = tar.getnames()
    print(f"→ downloaded job #{job_id} to {out}/ ({len(data) / 1e6:.1f} MB, SHA256 ok): {', '.join(names)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("status", "wait", "download"):
        p = sub.add_parser(name)
        p.add_argument("job", type=int)
        if name == "wait":
            p.add_argument("--interval", type=float, default=30)
            p.add_argument("--timeout", type=float, default=3600)
        if name == "download":
            p.add_argument("--out", help="folder (default firmware-job<ID>)")
    p = sub.add_parser("submit")
    p.add_argument("--run", help="run folder: model.json and formats from <run>/deploy/")
    p.add_argument("--model", help="model.json (instead of --run, or to override it)")
    p.add_argument("--weights", help="e.g. Q2.6 (default: from --run)")
    p.add_argument("--potentials", help="e.g. Q4.12 (default: from --run)")
    p.add_argument("--board", default="kv260", choices=("kv260", "vpk180"))
    p.add_argument("--prophesee-ip", default="genx320", choices=("none", "genx320"),
                   help="genx320 (default) builds the camera IP in, needed to use the camera on the board")
    p.add_argument("--strategy", choices=STRATEGIES, help="default: the run's fpga.strategy, else faster-generation")
    p.add_argument("--name")
    p.add_argument("--description")
    p.add_argument("--force", action="store_true", help="submit even if an identical finished job exists")
    p.add_argument("--dry-run", action="store_true", help="show what would be submitted, submit nothing")
    p.add_argument("--wait", action="store_true", help="poll until the job finishes")
    p.add_argument("--download", action="store_true", help="wait, then download + unpack (implies --wait)")
    p.add_argument("--out", help="download folder (default <run>/firmware-job<ID>)")
    p.add_argument("--interval", type=float, default=30, help="seconds between status checks")
    p.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    try:
        client = Client(load_session())
        if args.command == "list":
            for job in list_jobs(client):
                print(f"#{job['id']:<4} {job['created']:22s} {job['status']:12s} {job['model']:28s} {job['description'][:60]}")
        elif args.command == "status":
            print(json.dumps({k: v for k, v in job_info(client, args.job).items() if k != "page"}, indent=1))
        elif args.command == "wait":
            return 0 if wait(client, args.job, args.interval, args.timeout)["state"] == "done" else 1
        elif args.command == "download":
            download(client, args.job, Path(args.out or f"firmware-job{args.job}"))
        else:
            return cmd_submit(client, args)
        return 0
    except RiverError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
