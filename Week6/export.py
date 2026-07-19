"""
Converts a trained checkpoint (checkpoint_best.pt)
into a quantized, integer-only .nnue binary.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from converter import NUM_HALFKP_FEATURES, MAX_ACTIVE_FEATURES, FEATURE_PAD_VALUE, FLAG_IS_MATE
from model import NNUE, FT_SIZE, HIDDEN_SIZE
from dataset import SIGMOID_SCALE, NNUEDataset, load_chunk

MAGIC = b"MNUE"
FORMAT_VERSION = 3
PADDING_INDEX = NUM_HALFKP_FEATURES

WEIGHT_INT_MAX = 32767  # int16 ceiling for l1/l2/output weights
SMALL_TENSOR_ELEMENT_THRESHOLD = 256
FATAL_CLIP_FRACTION = 0.02

HEADER_STRUCT = struct.Struct("<4sIIIIIiiiifiifI") # layout for data storage


def largest_pow2_leq(x: float) -> int:
    if x < 1:
        return 1
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def load_checkpoint_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = NNUE().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def compute_weight_scale(weight: np.ndarray, name: str, clip_fraction: float = 0.001,
                          int_max: int = WEIGHT_INT_MAX) -> int:
    abs_w = np.abs(weight)
    max_abs = float(abs_w.max())
    if max_abs == 0.0:
        return 64

    if weight.size < SMALL_TENSOR_ELEMENT_THRESHOLD:
        clip_val = max_abs
    else:
        clip_val = float(np.percentile(abs_w, 100.0 * (1.0 - clip_fraction)))
        if clip_val == 0.0:
            clip_val = max_abs

    ideal = int_max / clip_val
    return largest_pow2_leq(ideal)


def quantize(model: NNUE, qa: int, clip_fraction: float = 0.001):
    warnings = []
    fatal_reasons = []

    def to_int_checked(arr: np.ndarray, dtype, name: str, expected_clip: bool = False):
        info = np.iinfo(dtype)
        rounded = np.round(arr)
        clipped = np.clip(rounded, info.min, info.max)
        n_clipped = int((clipped != rounded).sum())
        if n_clipped > 0:
            frac = n_clipped / arr.size
            if expected_clip:
                over_budget = frac > FATAL_CLIP_FRACTION
                warnings.append(
                    f"{name}: {n_clipped} of {arr.size} values ({100*frac:.3f}%) clipped to fit "
                    f"{dtype.__name__} range - EXPECTED (percentile-based scale targets "
                    f"clip_fraction={100*clip_fraction:.3f}% directly, no distribution shape "
                    f"assumed)."
                    + (f"  ** {100*frac:.1f}% is well above the {100*clip_fraction:.3f}% "
                       f"targeted - investigate. **" if over_budget else "")
                )
                if over_budget:
                    fatal_reasons.append(f"{name} clipped {100*frac:.1f}% of values, "
                                          f"exceeding FATAL_CLIP_FRACTION={100*FATAL_CLIP_FRACTION:.0f}%")
            else:
                warnings.append(
                    f"{name}: {n_clipped} of {arr.size} values ({100*frac:.3f}%) clipped to fit "
                    f"{dtype.__name__} range [{info.min},{info.max}] - UNEXPECTED for this tensor "
                    f"(no clipping tolerance applies here); investigate."
                )
                fatal_reasons.append(f"{name} had unexpected clipping ({n_clipped} values)")
        return clipped.astype(dtype)

    sd = model.state_dict()
    ft_w = sd["feature_transformer.weight"].detach().cpu().numpy().astype(np.float64)
    l1_w = sd["l1.weight"].detach().cpu().numpy().astype(np.float64)
    l1_b = sd["l1.bias"].detach().cpu().numpy().astype(np.float64)
    l2_w = sd["l2.weight"].detach().cpu().numpy().astype(np.float64)
    l2_b = sd["l2.bias"].detach().cpu().numpy().astype(np.float64)
    out_w = sd["output.weight"].detach().cpu().numpy().astype(np.float64)
    out_b = sd["output.bias"].detach().cpu().numpy().astype(np.float64)

    pad_row = ft_w[PADDING_INDEX]
    if not np.allclose(pad_row, 0.0):
        warnings.append(
            f"feature_transformer padding row (index {PADDING_INDEX}) is NOT "
            f"all-zero (max abs value {np.abs(pad_row).max():.6f}) - this will "
            f"corrupt evaluation of any position with fewer than "
            f"{MAX_ACTIVE_FEATURES} non-king pieces. Investigate before using "
            f"this export."
        )
        fatal_reasons.append("feature_transformer padding row is not all-zero")

    l1_scale = compute_weight_scale(l1_w, "l1.weight", clip_fraction)
    l2_scale = compute_weight_scale(l2_w, "l2.weight", clip_fraction)
    out_scale = compute_weight_scale(out_w, "output.weight", clip_fraction)

    quantized = {
        "ft_weight": to_int_checked(ft_w * qa, np.int16, "feature_transformer.weight"),
        "l1_weight": to_int_checked(l1_w * l1_scale, np.int16, "l1.weight", expected_clip=True),
        "l1_bias": to_int_checked(l1_b * qa * l1_scale, np.int32, "l1.bias"),
        "l2_weight": to_int_checked(l2_w * l2_scale, np.int16, "l2.weight", expected_clip=True),
        "l2_bias": to_int_checked(l2_b * qa * l2_scale, np.int32, "l2.bias"),
        "out_weight": to_int_checked(out_w.reshape(-1) * out_scale, np.int16, "output.weight", expected_clip=True),
        "out_bias": to_int_checked(out_b.reshape(-1) * qa * out_scale, np.int32, "output.bias"),
    }
    scales = {"l1_scale": l1_scale, "l2_scale": l2_scale, "out_scale": out_scale}
    return quantized, scales, warnings, fatal_reasons


def write_nnue_file(path, quantized, qa, scales, output_cp_clamp, epoch, val_loss, seed):
    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, FT_SIZE, HIDDEN_SIZE, NUM_HALFKP_FEATURES,
        PADDING_INDEX, qa,
        scales["l1_scale"], scales["l2_scale"], scales["out_scale"],
        float(SIGMOID_SCALE), output_cp_clamp,
        int(epoch), float(val_loss), int(seed) & 0xFFFFFFFF,
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(quantized["ft_weight"].tobytes(order="C"))
        f.write(quantized["l1_weight"].tobytes(order="C"))
        f.write(quantized["l1_bias"].tobytes(order="C"))
        f.write(quantized["l2_weight"].tobytes(order="C"))
        f.write(quantized["l2_bias"].tobytes(order="C"))
        f.write(quantized["out_weight"].tobytes(order="C"))
        f.write(quantized["out_bias"].tobytes(order="C"))


def simulate_quantized_forward(quantized, qa, scales, sigmoid_scale, output_cp_clamp,
                                own_idx: np.ndarray, opp_idx: np.ndarray):
    ft_w = quantized["ft_weight"].astype(np.int64)

    def accumulate(idx):
        return ft_w[idx].sum(axis=1)

    own_acc = accumulate(own_idx)
    opp_acc = accumulate(opp_idx)

    x = np.concatenate([own_acc, opp_acc], axis=1)
    x = np.clip(x, 0, qa)

    def dense(x_in, w_int, b_int32, layer_scale, clamp_output: bool):
        raw = x_in.astype(np.int64) @ w_int.astype(np.int64).T + b_int32.astype(np.int64)
        if clamp_output:
            rescaled = np.round(raw / layer_scale).astype(np.int64)
            return np.clip(rescaled, 0, qa)
        return raw

    h1 = dense(x, quantized["l1_weight"], quantized["l1_bias"], scales["l1_scale"], clamp_output=True)
    h2 = dense(h1, quantized["l2_weight"], quantized["l2_bias"], scales["l2_scale"], clamp_output=True)
    raw_out = dense(h2, quantized["out_weight"].reshape(1, -1), quantized["out_bias"],
                     scales["out_scale"], clamp_output=False)

    float_logit = raw_out.reshape(-1) / (qa * scales["out_scale"])
    score_cp = float_logit * sigmoid_scale
    score_cp = np.clip(score_cp, -output_cp_clamp, output_cp_clamp)
    return score_cp


def run_verification(model: NNUE, quantized, qa, scales, output_cp_clamp, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)

    own_idx = np.full((n_samples, MAX_ACTIVE_FEATURES), PADDING_INDEX, dtype=np.int64)
    opp_idx = np.full((n_samples, MAX_ACTIVE_FEATURES), PADDING_INDEX, dtype=np.int64)
    for row in range(n_samples):
        n_active = rng.integers(0, MAX_ACTIVE_FEATURES + 1)
        if n_active > 0:
            own_idx[row, :n_active] = rng.choice(NUM_HALFKP_FEATURES, size=n_active, replace=False)
            opp_idx[row, :n_active] = rng.choice(NUM_HALFKP_FEATURES, size=n_active, replace=False)

    with torch.no_grad():
        own_t = torch.from_numpy(own_idx)
        opp_t = torch.from_numpy(opp_idx)
        float_logits = model(own_t, opp_t).numpy()
    float_cp = np.clip(float_logits * SIGMOID_SCALE, -output_cp_clamp, output_cp_clamp)

    quant_cp = simulate_quantized_forward(quantized, qa, scales, SIGMOID_SCALE, output_cp_clamp, own_idx, opp_idx)

    abs_err = np.abs(float_cp - quant_cp)
    return {
        "n_samples": n_samples,
        "max_abs_err_cp": float(abs_err.max()),
        "mean_abs_err_cp": float(abs_err.mean()),
        "p99_abs_err_cp": float(np.percentile(abs_err, 99)),
        "direction_agreement": float((np.sign(float_cp) == np.sign(quant_cp)).mean()),
    }


def bucket_breakdown(float_cp, quant_cp, label_cp, is_mate):
    bucket_edges = [(0, 50), (50, 150), (150, 500), (500, 1000), (1000, float("inf"))]
    bucket_labels = ["0 to 50", "50 to 150", "150 to 500", "500 to 1000", "> 1000 (non-mate)"]

    abs_label = np.abs(label_cp)
    rows = []
    for (lo, hi), blabel in zip(bucket_edges, bucket_labels):
        mask = (~is_mate) & (abs_label > lo) & (abs_label <= hi if hi != float("inf") else True)
        n = int(mask.sum())
        if n == 0:
            rows.append({"label": blabel, "n": 0})
            continue
        rows.append({
            "label": blabel,
            "n": n,
            "float_mean_err_cp": float(np.abs(float_cp[mask] - label_cp[mask]).mean()),
            "float_direction_agree": float((np.sign(float_cp[mask]) == np.sign(label_cp[mask])).mean()),
            "quant_mean_err_cp": float(np.abs(quant_cp[mask] - label_cp[mask]).mean()),
            "quant_direction_agree": float((np.sign(quant_cp[mask]) == np.sign(label_cp[mask])).mean()),
        })
    return rows


def run_real_verification(model: NNUE, quantized, qa, scales, output_cp_clamp,
                           positions_dir: str, split_file: str, n_samples: int, seed: int = 0):
    positions_dir = Path(positions_dir)
    dataset = NNUEDataset(positions_dir / "positions.bin", positions_dir / split_file)

    if split_file == "test_idx.npy":
        print("  NOTE: verifying against test_idx.npy - this doesn't tune anything "
              "(pure quantization-error check, not model selection), but val_idx.npy "
              "works identically for this and keeps the test split untouched.")

    n_samples = min(n_samples, len(dataset))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset), size=n_samples, replace=False)
    real_indices = np.sort(dataset.indices[chosen])

    own_t, opp_t, target_t, _ = load_chunk(dataset, real_indices)
    own_idx = own_t.numpy()
    opp_idx = opp_t.numpy()

    recs = dataset.records[real_indices]
    is_mate = (recs["flags"] & FLAG_IS_MATE).astype(bool)
    label_cp = np.clip(recs["score_cp"].astype(np.float64), -output_cp_clamp, output_cp_clamp)

    with torch.no_grad():
        float_logits = model(own_t, opp_t).numpy()
    float_cp = np.clip(float_logits * SIGMOID_SCALE, -output_cp_clamp, output_cp_clamp)

    quant_cp = simulate_quantized_forward(quantized, qa, scales, SIGMOID_SCALE, output_cp_clamp, own_idx, opp_idx)

    def err_stats(pred, label, mask=None):
        p = pred[mask] if mask is not None else pred
        l = label[mask] if mask is not None else label
        if len(p) == 0:
            return None
        err = np.abs(p - l)
        return {
            "n": int(len(p)),
            "mean_abs_err_cp": float(err.mean()),
            "max_abs_err_cp": float(err.max()),
            "p99_abs_err_cp": float(np.percentile(err, 99)),
            "direction_agreement": float((np.sign(p) == np.sign(l)).mean()),
        }

    return {
        "n_samples": int(n_samples),
        "split_file": split_file,
        "n_mate": int(is_mate.sum()),
        "float_vs_label": err_stats(float_cp, label_cp),
        "float_vs_label_nonmate": err_stats(float_cp, label_cp, ~is_mate),
        "quant_vs_label": err_stats(quant_cp, label_cp),
        "quant_vs_label_nonmate": err_stats(quant_cp, label_cp, ~is_mate),
        "quant_vs_float": err_stats(quant_cp, float_cp),
        "buckets": bucket_breakdown(float_cp, quant_cp, label_cp, is_mate),
    }


def print_real_verification(result):
    def line(label, stats):
        if stats is None:
            return
        print(f"  {label:<28} n={stats['n']:>7,}  mean_abs_err_cp={stats['mean_abs_err_cp']:>7.2f}  "
              f"max_abs_err_cp={stats['max_abs_err_cp']:>8.2f}  p99={stats['p99_abs_err_cp']:>7.2f}  "
              f"direction_agree={100*stats['direction_agreement']:>6.2f}%")

    print(f"\n  ({result['n_mate']:,} of {result['n_samples']:,} sampled positions are forced-mate scores)")
    print("  --- vs real dataset labels (score_cp) - what actually matters ---")
    line("FLOAT model (all)", result["float_vs_label"])
    line("FLOAT model (non-mate only)", result["float_vs_label_nonmate"])
    line("QUANTIZED model (all)", result["quant_vs_label"])
    line("QUANTIZED model (non-mate only)", result["quant_vs_label_nonmate"])
    print("  --- quantized vs float directly (pure quantization error, isolated from model accuracy) ---")
    line("QUANTIZED vs FLOAT", result["quant_vs_float"])

    print("  --- by |label| magnitude, non-mate only (separates near-zero sign-flip noise from "
          "real decisive-position errors) ---")
    print(f"    {'range':<20} {'n':>8}  {'float_err':>10}  {'float_dir%':>11}  "
          f"{'quant_err':>10}  {'quant_dir%':>11}")
    for b in result["buckets"]:
        if b["n"] == 0:
            print(f"    {b['label']:<20} {0:>8}")
            continue
        print(f"    {b['label']:<20} {b['n']:>8,}  {b['float_mean_err_cp']:>10.2f}  "
              f"{100*b['float_direction_agree']:>10.2f}%  {b['quant_mean_err_cp']:>10.2f}  "
              f"{100*b['quant_direction_agree']:>10.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--output", required=True, help="Output .nnue path")
    parser.add_argument("--qa", type=int, default=127,
                         help="Activation-domain scale")
    parser.add_argument("--output-clamp-cp", type=int, default=30000,
                         help="Clamp final centipawn output to +/- this value.")
    parser.add_argument("--clip-fraction", type=float, default=0.001,
                         help="Fraction of each l1/l2/output tensor's most extreme |weight| "
                              "values to deliberately clip/saturate.")
    parser.add_argument("--verify", action="store_true",
                         help="Run the float-vs-simulated-quantized numerical equivalence check "
                              "before writing the file.")
    parser.add_argument("--real-positions-dir", default=None,
                         help="Directory containing positions.bin + ..._idx.npy")
    parser.add_argument("--real-split", default="val_idx.npy",
                         help="Which split to sample real positions from.")
    parser.add_argument("--verify-samples", type=int, default=2000,
                         help="Number of positions to check.")
    parser.add_argument("--verify-max-cp-err", type=float, default=50.0,
                         help="If --verify's max abs error (in cp) exceeds this, abort without "
                              "writing the file.")
    parser.add_argument("--force", action="store_true",
                         help="Write the .nnue file even if --verify reports errors above "
                              "--verify-max-cp-err, or if fatal warnings fired.")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_checkpoint_model(args.checkpoint)
    epoch = ckpt.get("epoch", -1)
    val_loss = ckpt.get("best_val_loss", float("nan"))
    seed = ckpt.get("seed", 0)
    print(f"  epoch={epoch}  best_val_loss={val_loss}  seed={seed}")

    print(f"Quantizing (QA={args.qa}, clip_fraction={args.clip_fraction}, "
          f"per-tensor auto scales for l1/l2/output, weights int16)...")
    quantized, scales, warnings, fatal_reasons = quantize(model, args.qa, args.clip_fraction)
    print(f"  l1.weight_scale  = {scales['l1_scale']}")
    print(f"  l2.weight_scale  = {scales['l2_scale']}")
    print(f"  output.weight_scale = {scales['out_scale']}")
    for w in warnings:
        print(f"  WARNING: {w}")

    if fatal_reasons and not args.force:
        print("\nFATAL: " + "; ".join(fatal_reasons), file=sys.stderr)
        print("Re-run with --force to export anyway, or investigate the checkpoint / "
              "--clip-fraction choice first.", file=sys.stderr)
        sys.exit(1)

    if args.verify:
        if args.real_positions_dir:
            print(f"\nRunning numerical equivalence check ({args.verify_samples} REAL positions "
                  f"from {args.real_split})...")
            result = run_real_verification(model, quantized, args.qa, scales, args.output_clamp_cp,
                                            args.real_positions_dir, args.real_split,
                                            n_samples=args.verify_samples)
            print_real_verification(result)
            gate_stats = result["quant_vs_float"]
        else:
            print(f"\nRunning numerical equivalence check ({args.verify_samples} SYNTHETIC "
                  f"positions - pass --real-positions-dir to check against real data instead)...")
            gate_stats = run_verification(model, quantized, args.qa, scales, args.output_clamp_cp,
                                           n_samples=args.verify_samples)
            print(f"  direction_agreement = {100*gate_stats['direction_agreement']:.2f}%")
            print(f"  max_abs_err_cp  = {gate_stats['max_abs_err_cp']:.2f}")
            print(f"  p99_abs_err_cp  = {gate_stats['p99_abs_err_cp']:.2f}")
            print(f"  mean_abs_err_cp = {gate_stats['mean_abs_err_cp']:.2f}")

        if gate_stats["max_abs_err_cp"] > args.verify_max_cp_err and not args.force:
            print(f"\nMax quantized-vs-float error ({gate_stats['max_abs_err_cp']:.2f} cp) exceeds "
                  f"--verify-max-cp-err ({args.verify_max_cp_err}). Aborting without writing "
                  f"the file. Re-run with --force to write anyway.", file=sys.stderr)
            sys.exit(1)

    print(f"\nWriting {args.output}...")
    write_nnue_file(args.output, quantized, args.qa, scales, args.output_clamp_cp,
                     epoch, val_loss, seed)
    print("Done.")


if __name__ == "__main__":
    main()