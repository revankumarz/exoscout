"""AstroNet: dual-view 1D CNN for transit classification.

Two input columns:
  * global view  (default 2001 bins) - the whole phase-folded light curve;
                 captures overall shape, secondary eclipses, out-of-transit trend.
  * local view   (default 201 bins)  - zoomed on the transit; captures depth and
                 the ingress/egress shape.

Each column is a stack of (conv - conv - maxpool) blocks; the two flattened
outputs are concatenated and passed through a fully-connected head to a single
logit = score for "planet". Faithful to Shallue & Vanderburg (2018), sized to
train quickly on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

GLOBAL_BINS = 2001
LOCAL_BINS = 201


class _ConvColumn(nn.Module):
    """A stack of (Conv-Conv-MaxPool) blocks with doubling filter counts."""

    def __init__(self, filters: list[int], kernel: int, pool: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        pad = kernel // 2
        for out_ch in filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=pad), nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, kernel, padding=pad), nn.ReLU(),
                nn.MaxPool1d(pool, stride=2),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, L)
        return torch.flatten(self.net(x), start_dim=1)


class AstroNet(nn.Module):
    def __init__(self, global_bins: int = GLOBAL_BINS, local_bins: int = LOCAL_BINS,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.global_bins = global_bins
        self.local_bins = local_bins
        # Global column: 5 blocks (deep, wide input). Local column: 2 blocks.
        self.global_col = _ConvColumn([16, 32, 64, 128, 256], kernel=5, pool=5)
        self.local_col = _ConvColumn([16, 32], kernel=5, pool=7)

        # LazyLinear infers the concatenated flatten size on first forward.
        self.head = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, g: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        # g: (B, global_bins), l: (B, local_bins) -> add channel dim.
        gf = self.global_col(g.unsqueeze(1))
        lf = self.local_col(l.unsqueeze(1))
        return self.head(torch.cat([gf, lf], dim=1)).squeeze(1)  # logits, (B,)


def load_model(path: str, map_location: str = "cpu") -> AstroNet:
    """Instantiate and load weights (runs a dummy forward to build LazyLinear)."""
    model = AstroNet()
    # Materialise lazy parameters before loading the state dict.
    with torch.no_grad():
        model(torch.zeros(1, model.global_bins), torch.zeros(1, model.local_bins))
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model
