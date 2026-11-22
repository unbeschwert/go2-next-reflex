"""Training configuration.

One YAML file describes a whole run: data, model, training, quantization and the FPGA
memory budget. Every value that must agree between training and the Kria
(input size, frame length, number formats) is derived from this one place.
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SENSOR_SIZE = 320                                # GenX320 is 320x320
DESPOT_DOWNSAMPLING = (1, 2, 4, 8, 16, 32, 64)   # values metavision_camera.json accepts
ENCODINGS = ("mean", "sum", "binary")
STRATEGIES = ("better-latency", "less-resources", "faster-generation")  # riversdk.dev firmware form


class ConfigError(ValueError):
    pass


@dataclass
class DataConfig:
    root: str = "data/motion_buddy_dataset"
    cache_dir: str = "cache"
    # Output neuron i = classes[i]. Matched against the name in gesture_<NN>-<name>-<style>-<take>.bin;
    # files of classes not listed here are skipped.
    classes: list[str] = field(default_factory=lambda: ["sit", "lay-down", "hello", "dance", "no-action"])
    # Person folders per split. Never put the same person in two splits.
    train: list[str] = field(default_factory=lambda: [f"{i:02d}" for i in range(1, 11)])
    val: list[str] = field(default_factory=lambda: ["11", "12"])
    test: list[str] = field(default_factory=lambda: ["13", "14"])
    clip_us: int = 1_000_000
    time_window_us: int = 10_000      # one SNN time step = one despot frame; frequency = 1e6 / this
    downsampling: int = 8             # 320 / downsampling = input size; same value goes to despot
    # How a frame pixel is computed from the events in its downsampling x downsampling block.
    # Must match what despot feeds the FPGA (see training/README.md).
    encoding: str = "mean"
    max_shift: int = 2                # random +-pixels shift (whole pixels, exact), training only
    limit_per_split: int | None = None  # use only N random files per split (smoke tests)

    @property
    def time_steps(self) -> int:
        return self.clip_us // self.time_window_us

    @property
    def size(self) -> int:
        return SENSOR_SIZE // self.downsampling

    @property
    def frequency_hz(self) -> float:
        return 1e6 / self.time_window_us


# --- Data augmentation (training batches only; see doc/data-augmentation.md) --------------------------
# Every augmentation has a probability p (per clip, 0 = off) and its strength. They run on the GPU on
# the cached event counts, in the order listed in AugmentConfig. All are off by default.

@dataclass
class FlipAug:
    p: float = 0.0                     # mirror left/right


@dataclass
class AffineAug:
    p: float = 0.0
    scale: list[float] = field(default_factory=lambda: [0.8, 1.2])   # zoom (>1: person appears bigger)
    rotate_deg: float = 10.0           # +- degrees


@dataclass
class TimeShiftAug:
    p: float = 0.0
    max_frames: int = 20               # move the clip content +- frames, empty frames fill the gap


@dataclass
class SpeedAug:
    p: float = 0.0
    range: list[float] = field(default_factory=lambda: [0.8, 1.25])  # >1: gesture played faster


@dataclass
class DropoutAug:
    p: float = 0.0
    max: float = 0.3                   # remove up to this fraction of events (binomial thinning)


@dataclass
class RateAug:
    p: float = 0.0
    range: list[float] = field(default_factory=lambda: [0.7, 1.3])   # multiply all event counts


@dataclass
class CutoutAug:
    p: float = 0.0
    size: list[float] = field(default_factory=lambda: [0.2, 0.4])    # rectangle side, fraction of the frame


@dataclass
class PolarityAug:
    p: float = 0.0                     # swap the OFF and ON channels


@dataclass
class NoiseAug:
    p: float = 0.0
    max_rate: float = 0.3              # extra random events per pixel per frame (after downsampling)


@dataclass
class HotPixelAug:
    p: float = 0.0
    max_count: int = 2                 # 1..max_count always-firing pixels per clip
    rate: list[float] = field(default_factory=lambda: [2.0, 20.0])   # their events per frame


@dataclass
class MixupAug:
    p: float = 0.0                     # probability per batch
    alpha: float = 0.4                 # blend weight lambda ~ Beta(alpha, alpha); targets are blended the same way


@dataclass
class WarmStartAug:
    p: float = 0.0                     # probability per batch
    frames: list[int] = field(default_factory=lambda: [10, 50])      # prefix length (same for the batch)
    from_clip: float = 0.5             # fraction of prefixes taken from another clip's end (else empty frames)


@dataclass
class AugmentConfig:
    qat: bool = True                   # also augment during the QAT phase
    dropout: DropoutAug = field(default_factory=DropoutAug)
    speed: SpeedAug = field(default_factory=SpeedAug)
    time_shift: TimeShiftAug = field(default_factory=TimeShiftAug)
    affine: AffineAug = field(default_factory=AffineAug)
    flip: FlipAug = field(default_factory=FlipAug)
    # then the integer shift (data.max_shift)
    cutout: CutoutAug = field(default_factory=CutoutAug)
    rate: RateAug = field(default_factory=RateAug)
    polarity: PolarityAug = field(default_factory=PolarityAug)
    noise: NoiseAug = field(default_factory=NoiseAug)
    hot_pixels: HotPixelAug = field(default_factory=HotPixelAug)
    warm_start: WarmStartAug = field(default_factory=WarmStartAug)
    mixup: MixupAug = field(default_factory=MixupAug)   # last: blends two augmented clips of the batch

    def active(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if f.name != "qat" and getattr(self, f.name).p > 0]


@dataclass
class NeuronConfig:
    initial_tau: float = 1.0 / (1.0 - 0.93)   # potential keeps 93 % per step
    learnable_leak: bool = False
    v_threshold: float = 1.0
    reset_mode: str = "HARD"


@dataclass
class LayerConfig:
    out: int | str          # number of channels, or "classes" for the output layer
    kernel: int
    stride: int = 1
    padding: int = 0


@dataclass
class ModelConfig:
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    layers: list[LayerConfig] = field(default_factory=list)
    # Dropout2d on the layer's currents (between BatchNorm and the neuron), training only: riverops
    # SpikingConvBNDropout2d, which fuses to the same exported layer, so it costs nothing on the FPGA.
    dropout: float = 0.0
    dropout_layers: list[int] = field(default_factory=list)   # layer indices, e.g. [1, 2, 3]


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 30
    lr: float = 0.003
    # Learning-rate schedule: None = constant lr. Otherwise cosine from lr to lr_min over cosine_epochs
    # (default: epochs - average_last), then constant lr_min.
    lr_min: float | None = None
    cosine_epochs: int | None = None
    # Weight averaging: the final float model is the average of the last N epochs' weights instead of
    # the single epoch with the best validation accuracy (0 = best epoch, as before).
    average_last: int = 0
    # With average_last: recompute the BatchNorm running statistics for the averaged weights (one pass
    # over the training data, as SWA) instead of averaging the stored statistics of those epochs.
    recompute_bn: bool = True
    # adam: weight decay added to the gradient (L2, as the reference notebook). adamw: decoupled decay.
    # Not equivalent strengths: Adam's 1e-4 is divided by the (small) gradient, so it matched AdamW 1e-2
    # in layers 0-1 but was ~15x stronger in layer 3 (doc/training-log.md, run #21).
    optimizer: str = "adam"
    # Which parameters get weight_decay, for either optimizer: all (as the notebook) | conv_weights
    # (not BatchNorm γ/β or the learnable leaks).
    decay_on: str = "all"
    # mse: spike counts to target_high/target_low * T (the notebook). ce: cross-entropy on counts /
    # ce_temperature. ce+mse: both, MSE weighted by ce_mse_weight (keeps the counts in the 20-80 % range).
    loss: str = "mse"
    ce_temperature: float = 10.0
    ce_mse_weight: float = 1.0
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    weight_clamp: float | None = 1.0   # clamp conv weights and BN affine params after each step
    target_high: float = 0.8           # target spike count = target_high * T for the true class
    target_low: float = 0.2            # ... and target_low * T for the others
    device: str = "auto"


@dataclass
class QuantConfig:
    # Signed fixed point "Q<int>.<frac>": int bits include the sign, total = int + frac.
    weight_bits: int = 8               # 8 → weights fit 9-bit BRAM words (see fpga.py)
    weight_int_bits: int | str = "auto"  # "auto": smallest that holds the BN-folded weights
    weight_percentile: float = 99.9    # "auto" sizes for this percentile of |weight| (and max |bias|)
    potential_bits: int = 16
    potential_int_bits: int = 4        # Q4.12 → spleat-server scaling 4096
    qat_epochs: int = 10
    qat_lr: float = 0.0001
    # One weight format for all layers (the firmware takes one), but each layer has its own threshold in
    # weights.txt. A layer whose folded weights/biases don't fit the format is divided by a power of 2,
    # k, together with its threshold (same spikes, wider range, coarser steps for that layer only).
    layer_scaling: bool = False
    max_layer_scale: int = 16
    layer_scale_biases: bool = True    # size each layer for its biases too (False: biases may clip; QAT adapts)


@dataclass
class FpgaConfig:
    board: str = "kv260"
    reserve: float = 0.25        # fraction of BRAM/URAM kept free for the camera IP, DMA, despot
    event_fifo_depth: int = 1024  # assumed input-event FIFO entries per layer
    lutram_max_bits: int = 1024  # memories this small are assumed to go to LUT RAM
    strategy: str = "less-resources"  # firmware form "Implementation strategy"
    fail_on_overflow: bool = True


@dataclass
class Config:
    name: str = "run"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    fpga: FpgaConfig = field(default_factory=FpgaConfig)

    @property
    def n_classes(self) -> int:
        return len(self.data.classes)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, values: Any, where: str):
    """Mapping → dataclass, recursively, rejecting unknown keys (catches typos in the YAML)."""
    if values is None:
        return cls()
    if not isinstance(values, dict):
        raise ConfigError(f"{where}: expected a mapping, got {values!r}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs = {}
    for name, value in values.items():
        kind = hints[name]
        if name == "layers":
            kwargs[name] = [_build(LayerConfig, v, f"{where}.layers[{i}]") for i, v in enumerate(value)]
        elif dataclasses.is_dataclass(kind):
            kwargs[name] = _build(kind, value, f"{where}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _read_yaml(path: Path, seen: tuple = ()) -> dict:
    """Read a config; `base: other.yaml` (relative to this file) inherits everything not set here."""
    if path.resolve() in seen:
        raise ConfigError(f"config base loop: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _merge(_read_yaml(path.parent / base, seen + (path.resolve(),)), raw)


def load_config(path: str | Path) -> Config:
    cfg = _build(Config, _read_yaml(Path(path)), "config")
    validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def validate(cfg: Config) -> None:
    d, q, f = cfg.data, cfg.quant, cfg.fpga
    if d.downsampling not in DESPOT_DOWNSAMPLING:
        raise ConfigError(f"data.downsampling must be one of {DESPOT_DOWNSAMPLING} (despot only supports "
                          f"these), got {d.downsampling}")
    if d.encoding not in ENCODINGS:
        raise ConfigError(f"data.encoding must be one of {ENCODINGS}")
    if d.clip_us % d.time_window_us:
        raise ConfigError("data.clip_us must be a multiple of data.time_window_us")
    if len(set(d.classes)) != len(d.classes) or not d.classes:
        raise ConfigError("data.classes must be a non-empty list of unique names")
    splits = {"train": set(d.train), "val": set(d.val), "test": set(d.test)}
    for a in splits:
        for b in splits:
            if a < b and splits[a] & splits[b]:
                raise ConfigError(f"person folders {sorted(splits[a] & splits[b])} are in both {a} and {b}")
    if not cfg.model.layers:
        raise ConfigError("model.layers is empty")
    for name, bits, ib in (("weight", q.weight_bits, q.weight_int_bits),
                           ("potential", q.potential_bits, q.potential_int_bits)):
        if not 2 <= bits <= 32:
            raise ConfigError(f"quant.{name}_bits must be 2..32")
        if ib != "auto" and not (isinstance(ib, int) and 1 <= ib < bits):
            raise ConfigError(f"quant.{name}_int_bits must be 'auto' or 1..{bits - 1}")
    if q.potential_int_bits == "auto":
        raise ConfigError("quant.potential_int_bits cannot be 'auto'")
    if f.strategy not in STRATEGIES:
        raise ConfigError(f"fpga.strategy must be one of {STRATEGIES}")
    if not 0 <= f.reserve < 1:
        raise ConfigError("fpga.reserve must be in [0, 1)")
    t, m = cfg.train, cfg.model
    if t.loss not in ("mse", "ce", "ce+mse"):
        raise ConfigError(f"train.loss must be 'mse', 'ce' or 'ce+mse', got {t.loss!r}")
    if t.ce_temperature <= 0 or t.ce_mse_weight < 0:
        raise ConfigError("train.ce_temperature must be > 0 and train.ce_mse_weight >= 0")
    if t.optimizer not in ("adam", "adamw"):
        raise ConfigError(f"train.optimizer must be 'adam' or 'adamw', got {t.optimizer!r}")
    if t.decay_on not in ("all", "conv_weights"):
        raise ConfigError(f"train.decay_on must be 'all' or 'conv_weights', got {t.decay_on!r}")
    if t.average_last < 0 or t.average_last > t.epochs:
        raise ConfigError("train.average_last must be 0..train.epochs")
    if t.lr_min is not None and not 0 < t.lr_min <= t.lr:
        raise ConfigError("train.lr_min must be in (0, train.lr]")
    if t.cosine_epochs is not None and not 1 <= t.cosine_epochs <= t.epochs:
        raise ConfigError("train.cosine_epochs must be 1..train.epochs")
    if not 0 <= m.dropout < 1:
        raise ConfigError("model.dropout must be in [0, 1)")
    if any(not 0 <= i < len(m.layers) for i in m.dropout_layers):
        raise ConfigError("model.dropout_layers: index out of range")
    if m.neuron.learnable_leak and m.neuron.initial_tau <= 1:
        raise ConfigError("model.neuron.initial_tau must be > 1 with learnable_leak")
    if cfg.model.neuron.reset_mode not in ("HARD", "SOFT"):
        raise ConfigError("model.neuron.reset_mode must be HARD or SOFT")
    validate_augment(cfg.augment, cfg.data)


def validate_augment(a: AugmentConfig, d: DataConfig) -> None:
    for f in dataclasses.fields(a):
        if f.name == "qat":
            continue
        aug = getattr(a, f.name)
        if not 0 <= aug.p <= 1:
            raise ConfigError(f"augment.{f.name}.p must be in [0, 1]")
        for g in dataclasses.fields(aug):
            value = getattr(aug, g.name)
            if isinstance(value, list) and (len(value) != 2 or value[0] > value[1] or value[0] < 0):
                raise ConfigError(f"augment.{f.name}.{g.name} must be [low, high] with 0 <= low <= high")
    if a.affine.scale[0] <= 0 or a.speed.range[0] <= 0:
        raise ConfigError("augment.affine.scale and augment.speed.range must be > 0")
    if not 0 <= a.dropout.max < 1:
        raise ConfigError("augment.dropout.max must be in [0, 1)")
    if a.cutout.size[1] > 1:
        raise ConfigError("augment.cutout.size is a fraction of the frame (<= 1)")
    if not 0 <= a.warm_start.from_clip <= 1:
        raise ConfigError("augment.warm_start.from_clip must be in [0, 1]")
    if a.warm_start.frames[1] > d.time_steps:
        raise ConfigError(f"augment.warm_start.frames cannot exceed the clip length ({d.time_steps} frames)")
    if a.time_shift.max_frames >= d.time_steps:
        raise ConfigError("augment.time_shift.max_frames must be shorter than the clip")
