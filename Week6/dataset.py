"""
PyTorch Dataset for the HalfKP training data produced by converter.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from converter import (
    RECORD_DTYPE, NUM_HALFKP_FEATURES, FLAG_STM_IS_BLACK, FLAG_IS_MATE,
    FEATURE_PAD_VALUE,
)

PADDING_INDEX = NUM_HALFKP_FEATURES

# just an initial value, tunable later
SIGMOID_SCALE = 400.0


class NNUEDataset(Dataset):
    def __init__(self, positions_path, indices_path, eval_clamp_cp=None,
                 endgame_weight_boost=0.0, endgame_phase_threshold=8,
                 decisive_weight_boost=0.0, decisive_cp_threshold=500,
                 sigmoid_scale=SIGMOID_SCALE):
        
        self.records = np.memmap(positions_path, dtype=RECORD_DTYPE, mode="r")
        self.indices = np.load(indices_path)
        self.eval_clamp_cp = eval_clamp_cp
        self.endgame_weight_boost = endgame_weight_boost
        self.endgame_phase_threshold = endgame_phase_threshold
        self.decisive_weight_boost = decisive_weight_boost
        self.decisive_cp_threshold = decisive_cp_threshold
        self.sigmoid_scale = sigmoid_scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        rec = self.records[self.indices[i]]

        is_mate = bool(rec["flags"] & FLAG_IS_MATE)
        stm_is_black = bool(rec["flags"] & FLAG_STM_IS_BLACK)

        if stm_is_black:
            own_raw = rec["black_features"]
            opp_raw = rec["white_features"]
        else:
            own_raw = rec["white_features"]
            opp_raw = rec["black_features"]

        own_idx = np.where(own_raw == FEATURE_PAD_VALUE, PADDING_INDEX, own_raw).astype(np.int64)
        opp_idx = np.where(opp_raw == FEATURE_PAD_VALUE, PADDING_INDEX, opp_raw).astype(np.int64)

        raw_score = float(rec["score_cp"])
        pre_clamp_abs = abs(raw_score)
        if self.eval_clamp_cp is not None and not is_mate:
            raw_score = max(-self.eval_clamp_cp, min(self.eval_clamp_cp, raw_score))

        target_prob = 1.0 / (1.0 + np.exp(-raw_score / self.sigmoid_scale))

        weight = 1.0
        if self.endgame_weight_boost > 0.0 and rec["phase"] <= self.endgame_phase_threshold:
            weight += self.endgame_weight_boost
        if self.decisive_weight_boost > 0.0 and (is_mate or pre_clamp_abs > self.decisive_cp_threshold):
            weight += self.decisive_weight_boost

        return (
            torch.from_numpy(own_idx),
            torch.from_numpy(opp_idx),
            torch.tensor(target_prob, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


def _records_to_tensors(dataset: "NNUEDataset", recs: np.ndarray):
    is_mate = (recs["flags"] & FLAG_IS_MATE).astype(bool)
    stm_is_black = (recs["flags"] & FLAG_STM_IS_BLACK).astype(bool)

    white_feats = recs["white_features"]
    black_feats = recs["black_features"]

    mask = stm_is_black[:, None]
    own_raw = np.where(mask, black_feats, white_feats)
    opp_raw = np.where(mask, white_feats, black_feats)

    own_idx = np.where(own_raw == FEATURE_PAD_VALUE, PADDING_INDEX, own_raw).astype(np.int64)
    opp_idx = np.where(opp_raw == FEATURE_PAD_VALUE, PADDING_INDEX, opp_raw).astype(np.int64)

    raw_score = recs["score_cp"].astype(np.float64)
    pre_clamp_abs = np.abs(raw_score)

    if dataset.eval_clamp_cp is not None:
        clamped = np.clip(raw_score, -dataset.eval_clamp_cp, dataset.eval_clamp_cp)
        raw_score = np.where(is_mate, raw_score, clamped)

    target_prob = 1.0 / (1.0 + np.exp(-raw_score / dataset.sigmoid_scale))

    weight = np.ones(len(recs), dtype=np.float64)
    if dataset.endgame_weight_boost > 0.0:
        weight = weight + np.where(recs["phase"] <= dataset.endgame_phase_threshold,
                                    dataset.endgame_weight_boost, 0.0)
    if dataset.decisive_weight_boost > 0.0:
        decisive = is_mate | (pre_clamp_abs > dataset.decisive_cp_threshold)
        weight = weight + np.where(decisive, dataset.decisive_weight_boost, 0.0)

    return (
        torch.from_numpy(own_idx),
        torch.from_numpy(opp_idx),
        torch.from_numpy(target_prob.astype(np.float32)),
        torch.from_numpy(weight.astype(np.float32)),
    )


def load_chunk(dataset: "NNUEDataset", sorted_real_indices: np.ndarray):
    recs = dataset.records[sorted_real_indices]
    return _records_to_tensors(dataset, recs)


def build_split_membership(dataset: "NNUEDataset") -> np.ndarray:
    total_records = len(dataset.records)
    membership = np.zeros(total_records, dtype=bool)
    membership[dataset.indices] = True
    return membership


def load_window(dataset: "NNUEDataset", membership: np.ndarray, window_start: int, window_size: int):
    window_end = min(window_start + window_size, len(dataset.records))
    window_recs = dataset.records[window_start:window_end]
    window_membership = membership[window_start:window_end]

    recs = window_recs[window_membership]
    return _records_to_tensors(dataset, recs)