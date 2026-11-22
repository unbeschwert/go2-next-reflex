#!/usr/bin/env python3
"""Convert SPLEAT weights between the board's text file and the riverops parameter JSON.

    python3 weights_convert.py weights.txt weights.json [--arch nb40]     # text -> JSON
    python3 weights_convert.py weights.json weights.txt                   # JSON -> text

Text (what /root/weights.txt on the board is): one integer per line - all weights layer by layer,
then all biases, then threshold and leak of every layer. JSON (what riverops export_params /
the training notebook writes): {"weights_0": [...], ..., "bias_0": [...], ..., "threshold_0": n,
"leak_0": n, ...}. Same numbers, same order.
"""
import json
import sys

ARCHS = {  # input size, [(in, out, kernel, stride, padding), ...]
    "nb40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    "best80": (80, [(2, 16, 5, 2, 1), (16, 32, 3, 1, 1), (32, 32, 6, 2, 2),
                    (32, 256, 6, 2, 0), (256, 128, 3, 2, 0), (128, 5, 3, 1, 0)]),
    "notebook32": (32, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 6, 2, 0), (256, 5, 1, 1, 0)]),
}


def txt_to_json(src, dst, arch):
    vals = [int(v) for v in open(src).read().split()]
    layers = ARCHS[arch][1]
    n_w = [ci * co * k * k for (ci, co, k, _, _) in layers]
    n_b = [co for (_, co, _, _, _) in layers]
    need = sum(n_w) + sum(n_b) + 2 * len(layers)
    if len(vals) != need:
        raise SystemExit(f"{src}: {len(vals)} values, architecture {arch} needs {need}")
    out, o = {}, 0
    for i, n in enumerate(n_w):
        out[f"weights_{i}"] = vals[o:o + n]; o += n
    for i, n in enumerate(n_b):
        out[f"bias_{i}"] = vals[o:o + n]; o += n
    for i in range(len(layers)):
        out[f"threshold_{i}"], out[f"leak_{i}"] = vals[o], vals[o + 1]; o += 2
    json.dump(out, open(dst, "w"))
    print(f"{dst}: {len(layers)} layers, {need} values ({arch})")


def json_to_txt(src, dst):
    vals = [v for x in json.load(open(src)).values() for v in (x if isinstance(x, list) else [x])]
    open(dst, "w").write("\n".join(map(str, vals)))
    print(f"{dst}: {len(vals)} values")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "nb40"
    if "--arch" in sys.argv:
        args.remove(arch)
    if len(args) != 2:
        raise SystemExit(__doc__)
    (json_to_txt(*args) if args[0].endswith(".json") else txt_to_json(args[0], args[1], arch))
