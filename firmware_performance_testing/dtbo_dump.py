"""Dump a device-tree blob (.dtbo) as path/property = value lines, no dtc needed.

usage: python3 dtbo_dump.py FILE [--all]   (default: only clock-related properties and firmware-name)
"""
import struct
import sys

path = sys.argv[1]
show_all = "--all" in sys.argv
data = open(path, "rb").read()
magic, total, off_struct, off_strings = struct.unpack(">4I", data[:16])
assert magic == 0xD00DFEED, "not a device-tree blob"


def cstr(buf, at):
    end = buf.index(b"\0", at)
    return buf[at:end].decode(errors="replace"), end


def fmt(name, value):
    if not value:
        return "<empty>"
    text = value[:-1]
    if value.endswith(b"\0") and all(32 <= c < 127 or c == 0 for c in text) and text and not text.startswith(b"\0"):
        return " | ".join(s.decode() for s in text.split(b"\0"))
    if len(value) % 4 == 0:
        return "<" + " ".join(str(v) for v in struct.unpack(f">{len(value) // 4}I", value)) + ">"
    return value.hex()


KEEP = ("clock", "freq", "rate", "firmware-name", "compatible")
stack, i = [], off_struct
while True:
    token = struct.unpack(">I", data[i:i + 4])[0]
    i += 4
    if token == 1:
        name, end = cstr(data, i)
        stack.append(name)
        i = (end + 4) & ~3
    elif token == 2:
        stack.pop()
    elif token == 3:
        length, nameoff = struct.unpack(">2I", data[i:i + 8])
        i += 8
        value = data[i:i + length]
        i = (i + length + 3) & ~3
        prop, _ = cstr(data, off_strings + nameoff)
        if show_all or any(k in prop for k in KEEP):
            print(f"{'/'.join(stack) or '/'}  {prop} = {fmt(prop, value)}")
    elif token == 4:
        continue
    elif token == 9:
        break
    else:
        raise ValueError(f"bad token {token} at {i - 4}")
