"""
Trains the NNUE network (model.py) on the HalfKP dataset produced by
converter.py, using the dataset.py loader.
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dataset import NNUEDataset, load_chunk, load_window, build_split_membership
from model import NNUE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path: Path, model, optimizer, epoch, best_val_loss, patience_counter, seed):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "seed": seed,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, model, optimizer, device, override_lr=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if override_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = override_lr
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["patience_counter"], ckpt["seed"]


def iterate_batches(own, opp, target, weight, batch_size, device, shuffle=True, drop_last=True):
    n = own.size(0)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    n_batches = n // batch_size if drop_last else -(-n // batch_size)
    for b in range(n_batches):
        idx = order[b * batch_size: min((b + 1) * batch_size, n)]
        yield (
            own[idx].to(device, non_blocking=True),
            opp[idx].to(device, non_blocking=True),
            target[idx].to(device, non_blocking=True),
            weight[idx].to(device, non_blocking=True),
        )


def weighted_mse_loss(pred_logits: torch.Tensor, target_prob: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred_logits)
    return (weight * (pred_prob - target_prob) ** 2).mean()


def run_validation(model, own, opp, target, weight, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for b_own, b_opp, b_target, b_weight in iterate_batches(own, opp, target, weight, batch_size,
                                                                  device, shuffle=False, drop_last=False):
            logits = model(b_own, b_opp)
            loss = weighted_mse_loss(logits, b_target, b_weight)
            total_loss += loss.item() * b_own.size(0)
            total_count += b_own.size(0)
    model.train()
    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing positions.bin + ...idx.npy")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--positions-per-epoch", type=int, default=20_000_000,
                         help="Random subsample size drawn fresh each epoch")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience, in epochs")
    parser.add_argument("--checkpoint-minutes", type=float, default=8.0,
                         help="Save a checkpoint at least this often, in addition to every epoch end")
    parser.add_argument("--val-subsample", type=int, default=500_000,
                         help="How many validation positions to check each epoch")
    parser.add_argument("--eval-clamp-cp", type=int, default=None,
                         help="Clamp |score_cp| before the sigmoid target.")
    parser.add_argument("--endgame-weight-boost", type=float, default=0.0)
    parser.add_argument("--decisive-weight-boost", type=float, default=0.0)
    parser.add_argument("--decisive-cp-threshold", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "checkpoint_latest.pt"
    best_path = ckpt_dir / "checkpoint_best.pt"

    set_seed(args.seed)

    train_full = NNUEDataset(
        data_dir / "positions.bin", data_dir / "train_idx.npy",
        eval_clamp_cp=args.eval_clamp_cp, endgame_weight_boost=args.endgame_weight_boost,
        decisive_weight_boost=args.decisive_weight_boost, decisive_cp_threshold=args.decisive_cp_threshold,
    )
    val_full = NNUEDataset(
        data_dir / "positions.bin", data_dir / "val_idx.npy",
        eval_clamp_cp=args.eval_clamp_cp,
    )
    print(f"Train pool: {len(train_full):,}  Val pool: {len(val_full):,}")

    model = NNUE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if latest_path.exists():
        start_epoch, best_val_loss, patience_counter, saved_seed = load_checkpoint(
            latest_path, model, optimizer, device, override_lr=args.lr)
        print(f"  LR set to {args.lr} for this run (this ALWAYS applies the current --lr on resume, "
              f"even if it differs from what the checkpoint was saved with - see load_checkpoint() docstring)")
        start_epoch += 1
        print(f"Resumed from {latest_path}: starting at epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.6f}, patience_counter={patience_counter}")
        if saved_seed != args.seed:
            print(f"  NOTE: checkpoint was trained with seed={saved_seed}, current run uses seed={args.seed} - "
                  f"per-epoch subsamples from here on will differ from an uninterrupted run, harmless but noting it.")
    else:
        print("No checkpoint found - starting fresh.")

    train_membership = build_split_membership(train_full)
    total_records = len(train_full.records)

    val_rng = np.random.default_rng(args.seed)
    val_sample_size = min(args.val_subsample, len(val_full))
    val_positions = val_rng.choice(len(val_full), size=val_sample_size, replace=False)
    val_real_indices = np.sort(val_full.indices[val_positions])
    print(f"Loading validation subsample ({val_sample_size:,} positions, one bulk sorted read)...")
    load_start = time.time()
    val_own, val_opp, val_target, val_weight = load_chunk(val_full, val_real_indices)
    print(f"  loaded {val_own.shape[0]:,} val positions in {time.time()-load_start:.1f}s")
    if device == "cuda":
        val_own, val_opp, val_target, val_weight = (
            t.pin_memory() for t in (val_own, val_opp, val_target, val_weight)
        )

    last_checkpoint_time = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()
        epoch_rng = np.random.default_rng(args.seed + epoch)
        sample_size = min(args.positions_per_epoch, len(train_full))
        train_fraction = len(train_full) / total_records
        window_size = min(int(sample_size / max(train_fraction, 1e-6) * 1.2), total_records)
        window_start = int(epoch_rng.integers(0, max(total_records - window_size, 1)))

        load_start = time.time()
        own, opp, target, weight = load_window(train_full, train_membership, window_start, window_size)
        load_elapsed = time.time() - load_start
        n_loaded = own.shape[0]
        print(f"  [epoch {epoch}] loaded {n_loaded:,} positions in {load_elapsed:.1f}s "
              f"({n_loaded/max(load_elapsed,1e-9):,.0f} pos/s read)")

        running_loss = 0.0
        running_count = 0

        for step, (own_idx, opp_idx, b_target, b_weight) in enumerate(
                iterate_batches(own, opp, target, weight, args.batch_size, device, shuffle=True, drop_last=True)):
            optimizer.zero_grad()
            logits = model(own_idx, opp_idx)
            loss = weighted_mse_loss(logits, b_target, b_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            running_loss += loss.item() * own_idx.size(0)
            running_count += own_idx.size(0)

            if time.time() - last_checkpoint_time >= args.checkpoint_minutes * 60:
                save_checkpoint(latest_path, model, optimizer, epoch, best_val_loss, patience_counter, args.seed)
                last_checkpoint_time = time.time()
                elapsed = time.time() - epoch_start
                rate = running_count / elapsed if elapsed > 0 else 0
                print(f"  [epoch {epoch}] step {step:,}  running_loss={running_loss/max(running_count,1):.6f}  "
                      f"rate={rate:,.0f} pos/s  (mid-epoch checkpoint saved)")

        train_loss = running_loss / max(running_count, 1)
        val_loss = run_validation(model, val_own, val_opp, val_target, val_weight, args.batch_size, device)
        epoch_elapsed = time.time() - epoch_start

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"elapsed={epoch_elapsed/60:.1f}m  ({running_count/epoch_elapsed:,.0f} pos/s)")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, patience_counter, args.seed)
            print(f"  New best val_loss - saved {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        save_checkpoint(latest_path, model, optimizer, epoch, best_val_loss, patience_counter, args.seed)
        last_checkpoint_time = time.time()

        if patience_counter >= args.patience:
            print(f"Early stopping: no val improvement for {args.patience} epochs.")
            break

    print(f"\nDone. Best val_loss={best_val_loss:.6f}, checkpoint at {best_path}")


if __name__ == "__main__":
    main()