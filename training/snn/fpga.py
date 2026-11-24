"""On-chip memory estimate for a SPLEAT model on the Kria KV260.

The KV260's K26 SOM (XCK26, Zynq UltraScale+) has two kinds of large on-chip RAM:

    Block RAM (BRAM)   144 x 36 Kb  (5.1 Mb)   each block is 36 Kb, or two independent 18 Kb halves,
                                               with a fixed set of depth x width shapes (below)
    UltraRAM  (URAM)    64 x 288 Kb (18 Mb)    each block is 4096 x 72 bit, nothing else

A memory always occupies whole blocks, and a word is stored in the next native width up, so
the cost depends on the shape, not just the bit count: an 8-bit weight takes a 9-bit BRAM
slot (4096 per 36 Kb block); a 10-bit one takes an 18-bit slot (2048 per block), i.e. 2x the
BRAM. This is what makes the weight/potential bit widths a memory decision.

SPLEAT runs one processing unit per layer, pipelined, and on-chip memory is the limiting
factor (Courtois et al., arXiv:2406.17617). Its exact memory layout is not public, so this is
an ESTIMATE under these assumptions, all visible in the report:

    * per layer: weights (C_out*C_in*k*k words of weight_bits), biases (C_out words),
      membrane potentials (one potential_bits word per neuron, C_out*H_out*W_out; they must
      persist between time steps), and an input-event FIFO (event_fifo_depth entries, each
      an address of the input neuron);
    * thresholds/leaks are per-layer constants (registers);
    * memories of at most lutram_max_bits go to LUT RAM instead of BRAM/URAM;
    * a fraction `reserve` of BRAM and URAM is kept for the GenX320 camera IP, DMA and the
      rest of the design.

Two placements are reported: everything in BRAM (worst case), and the largest memories
moved to URAM where that saves the most BRAM (URAM words packed with as many values as fit
in 72 bits). The firmware generator decides the real placement; its "Less resources"
strategy is the closest to these numbers.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import Config
from .model import LayerShape, layer_shapes

BOARDS = {
    # name: (description, BRAM36 blocks, URAM blocks)
    "kv260": ("Kria KV260 / K26 SOM (XCK26, Zynq UltraScale+ MPSoC)", 144, 64),
}

# Simple-dual-port shapes: (depth, width). Widths 9/18/36/72 include the parity bits, usable as data.
BRAM36_SHAPES = [(32768, 1), (16384, 2), (8192, 4), (4096, 9), (2048, 18), (1024, 36), (512, 72)]
BRAM18_SHAPES = [(16384, 1), (8192, 2), (4096, 4), (2048, 9), (1024, 18), (512, 36)]
URAM_DEPTH, URAM_WIDTH = 4096, 72
BRAM36_BITS, URAM_BITS = 36 * 1024, URAM_DEPTH * URAM_WIDTH


def bram_blocks(depth: int, width: int) -> float:
    """Fewest 36 Kb blocks (in steps of 0.5 = one 18 Kb half) for a depth x width memory."""
    best = math.inf
    for shapes, cost in ((BRAM18_SHAPES, 0.5), (BRAM36_SHAPES, 1.0)):
        for d, w in shapes:
            best = min(best, cost * math.ceil(width / w) * math.ceil(depth / d))
    return best


def uram_blocks(depth: int, width: int) -> int:
    """URAM blocks, packing floor(72 / width) values per 72-bit word."""
    if width > URAM_WIDTH:
        return math.ceil(width / URAM_WIDTH) * math.ceil(depth / URAM_DEPTH)
    return math.ceil(math.ceil(depth / (URAM_WIDTH // width)) / URAM_DEPTH)


@dataclass
class Memory:
    layer: int
    kind: str          # weights | biases | potentials | event_fifo
    depth: int
    width: int
    bram: float = 0.0
    uram: int = 0
    place: str = ""    # lutram | bram | uram

    @property
    def bits(self) -> int:
        return self.depth * self.width


def layer_memories(shapes: list[LayerShape], weight_bits: int, potential_bits: int,
                   fifo_depth: int, lutram_max_bits: int) -> list[Memory]:
    memories = []
    for s in shapes:
        address_bits = max(1, math.ceil(math.log2(s.in_c * s.in_h * s.in_w)))
        memories += [
            Memory(s.index, "weights", s.weights, weight_bits),
            Memory(s.index, "biases", s.biases, weight_bits),
            Memory(s.index, "potentials", s.neurons, potential_bits),
            Memory(s.index, "event_fifo", fifo_depth, address_bits),
        ]
    for m in memories:
        m.bram = bram_blocks(m.depth, m.width)
        m.uram = uram_blocks(m.depth, m.width)
        m.place = "lutram" if m.bits <= lutram_max_bits else "bram"
    return memories


def estimate(cfg: Config, weight_bits: int | None = None, potential_bits: int | None = None) -> dict:
    """Memory estimate and verdict for cfg (weight/potential widths default to the config's)."""
    f = cfg.fpga
    description, bram_total, uram_total = BOARDS[f.board]
    weight_bits = weight_bits or cfg.quant.weight_bits
    potential_bits = potential_bits or cfg.quant.potential_bits
    shapes = layer_shapes(cfg)
    memories = layer_memories(shapes, weight_bits, potential_bits, f.event_fifo_depth, f.lutram_max_bits)
    bram_budget = bram_total * (1 - f.reserve)
    uram_budget = math.floor(uram_total * (1 - f.reserve))

    on_chip = [m for m in memories if m.place != "lutram"]
    bram_only = sum(m.bram for m in on_chip)

    # Move memories to URAM, the biggest BRAM saving per URAM block first.
    uram_used, bram_with_uram = 0, bram_only
    for m in sorted(on_chip, key=lambda m: m.bram / m.uram, reverse=True):
        if bram_with_uram <= bram_budget * 0.5:     # comfortably fits already: keep the rest in BRAM
            break
        if m.bram > m.uram and uram_used + m.uram <= uram_budget:
            m.place = "uram"
            uram_used += m.uram
            bram_with_uram -= m.bram

    if bram_only <= bram_budget:
        verdict = "fits"
    elif bram_with_uram <= bram_budget:
        verdict = "fits_with_uram"
    else:
        verdict = "too_big"

    return {
        "board": description,
        "weight_bits": weight_bits,
        "potential_bits": potential_bits,
        "bram_total": bram_total,
        "uram_total": uram_total,
        "reserve": f.reserve,
        "bram_budget": bram_budget,
        "uram_budget": uram_budget,
        "bram_only": bram_only,
        "bram_with_uram": bram_with_uram,
        "uram_used": uram_used,
        "lutram_bits": sum(m.bits for m in memories if m.place == "lutram"),
        "total_bits": sum(m.bits for m in memories),
        "verdict": verdict,
        "memories": [asdict(m) | {"bits": m.bits} for m in memories],
        "layers": [asdict(s) for s in shapes],
    }


def compare_weight_widths(cfg: Config, widths=(8, 9, 16)) -> list[tuple[int, float]]:
    """BRAM-only cost of the same model with other weight widths (16 = firmware form default Q4.12)."""
    return [(w, estimate(cfg, weight_bits=w)["bram_only"]) for w in widths]


def report_markdown(r: dict, cfg: Config) -> str:
    pct = lambda used, total: f"{100 * used / total:.0f} %"  # noqa: E731
    verdicts = {
        "fits": "FITS in BRAM alone",
        "fits_with_uram": "FITS only if the large memories go to URAM",
        "too_big": "DOES NOT FIT: shrink the model or the bit widths",
    }
    lines = [
        f"# FPGA memory estimate: {cfg.name}",
        "",
        f"Board: {r['board']}: {r['bram_total']} BRAM36 + {r['uram_total']} URAM. "
        f"Budget after keeping {r['reserve']:.0%} free: **{r['bram_budget']:.0f} BRAM, {r['uram_budget']} URAM**.",
        f"Formats: weights/biases {r['weight_bits']} bit, membrane potentials {r['potential_bits']} bit.",
        "",
        f"**Verdict: {verdicts[r['verdict']]}**",
        "",
        "| Placement | BRAM36 | URAM |",
        "|---|---|---|",
        f"| Everything in BRAM | {r['bram_only']:.1f} ({pct(r['bram_only'], r['bram_total'])}) | 0 |",
        f"| Large memories in URAM | {r['bram_with_uram']:.1f} ({pct(r['bram_with_uram'], r['bram_total'])}) "
        f"| {r['uram_used']} ({pct(r['uram_used'], r['uram_total'])}) |",
        "",
        f"Total stored: {r['total_bits'] / 8 / 1024:.1f} KiB ({r['lutram_bits'] / 8:.0f} B in LUT RAM).",
        "",
        "| Layer | Memory | Words x bits | BRAM36 | URAM | Placed in |",
        "|---|---|---|---|---|---|",
    ]
    for m in r["memories"]:
        lines.append(f"| {m['layer']} | {m['kind']} | {m['depth']:,} x {m['width']} | {m['bram']:g} | "
                     f"{m['uram']} | {m['place']} |")
    lines += [
        "",
        "Estimate only: SPLEAT's real memory layout is not public. Assumptions are listed in "
        "`training/snn/fpga.py` and `training/README.md`. The firmware job on riversdk.dev is the real check.",
    ]
    return "\n".join(lines) + "\n"


def report_text(r: dict) -> str:
    return (f"  BRAM-only: {r['bram_only']:.1f}/{r['bram_total']} BRAM36 "
            f"(budget {r['bram_budget']:.0f}); with URAM: {r['bram_with_uram']:.1f} BRAM36 + "
            f"{r['uram_used']}/{r['uram_total']} URAM → {r['verdict'].upper()}")
