"""Input cadence and spike decoding shared by native palm training and replay."""

from event_data import SIZE


BIN_US = 10_000
READOUT_STEPS = 10
FORECAST_WINDOW_US = 50_000


def coordinates(rates):
    """Decode average spikes ordered X+, X-, Y+, Y-, unused."""
    return ((rates[..., 0:4:2] - rates[..., 1:4:2]) + 1) * (SIZE / 2)
