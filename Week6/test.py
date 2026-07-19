"""
Runs the trained model against test_idx.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import NNUEDataset, load_chunk
from model import NNUE
from train import iterate_batches, weighted_mse_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--eval-clamp-cp", type=int, default=None,
                         help="MUST match whatever --eval-clamp-cp value the checkpoint was trained with")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    test_full = NNUEDataset(data_dir / "positions.bin", data_dir / "test_idx.npy",
                             eval_clamp_cp=args.eval_clamp_cp)
    print(f"Test pool: {len(test_full):,} positions (evaluating ALL of them - one-time job, no subsampling)")
    
    sorted_real_indices = np.sort(test_full.indices)
    print("Loading test set (one bulk sorted read)...")
    own, opp, target, weight = load_chunk(test_full, sorted_real_indices)
    print(f"Loaded {own.shape[0]:,} positions.")

    model = NNUE().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(recorded best_val_loss at save time: {ckpt['best_val_loss']:.6f})")

    if device == "cuda":
        own, opp, target, weight = (t.pin_memory() for t in (own, opp, target, weight))

    total_sq_err = 0.0
    total_abs_err = 0.0
    total_direction_agree = 0
    total_weighted_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for b_own, b_opp, b_target, b_weight in iterate_batches(
                own, opp, target, weight, args.batch_size, device, shuffle=False, drop_last=False):
            logits = model(b_own, b_opp)
            pred_prob = torch.sigmoid(logits)

            diff = pred_prob - b_target
            total_sq_err += (diff ** 2).sum().item()
            total_abs_err += diff.abs().sum().item()
            total_direction_agree += ((pred_prob > 0.5) == (b_target > 0.5)).sum().item()
            total_weighted_loss += weighted_mse_loss(logits, b_target, b_weight).item() * b_own.size(0)
            total_count += b_own.size(0)

    mse = total_sq_err / total_count
    rmse = mse ** 0.5
    mae = total_abs_err / total_count
    direction_agreement = total_direction_agree / total_count
    weighted_loss = total_weighted_loss / total_count

    print("\n" + "=" * 50)
    print("FINAL TEST SET RESULTS (one-time, honest number)")
    print("=" * 50)
    print(f"Positions evaluated:   {total_count:,}")
    print(f"MSE (win-prob space):  {mse:.6f}")
    print(f"RMSE:                  {rmse:.6f}  (typical error, in win-probability points)")
    print(f"MAE:                   {mae:.6f}")
    print(f"Direction agreement:   {100*direction_agreement:.2f}%  "
          f"(model and target agree on which side is favored)")
    print(f"Weighted loss (same metric train.py reports as val_loss): {weighted_loss:.6f}")
    print("=" * 50)

if __name__ == "__main__":
    main()