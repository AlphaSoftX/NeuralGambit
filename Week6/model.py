"""
A HalfKP NNUE in 256x2-32-32-1 shape
"""

import torch
import torch.nn as nn

from converter import NUM_HALFKP_FEATURES

FT_SIZE = 256           # feature transformer output width, per perspective
HIDDEN_SIZE = 32        # width of the two small dense layers
PADDING_INDEX = NUM_HALFKP_FEATURES  # the dedicated always-zero row


def clipped_relu(x: torch.Tensor) -> torch.Tensor:
    """Introduces non linearity in NNUE layers."""
    return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):
    def __init__(self, ft_size: int = FT_SIZE, hidden_size: int = HIDDEN_SIZE):
        super().__init__()

        self.feature_transformer = nn.EmbeddingBag(
            num_embeddings=NUM_HALFKP_FEATURES + 1, # + 1 for PADDING_INDEX (empty feature)
            embedding_dim=ft_size,
            mode="sum",
            padding_idx=PADDING_INDEX,
        )

        self.l1 = nn.Linear(ft_size * 2, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.feature_transformer.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.feature_transformer.weight[PADDING_INDEX].zero_()

    def forward(self, own_idx: torch.Tensor, opp_idx: torch.Tensor) -> torch.Tensor:
        """Main network layers computation."""
        own_acc = self.feature_transformer(own_idx)
        opp_acc = self.feature_transformer(opp_idx)

        x = torch.cat([own_acc, opp_acc], dim=1)
        x = clipped_relu(x)
        x = clipped_relu(self.l1(x))
        x = clipped_relu(self.l2(x))
        logit = self.output(x).squeeze(-1)
        return logit