"""
Converts the Lichess raw data (lichess_db_eval.jsonl.zst) into a
compact, fixed-size binary dataset of HalfKP training records, ready to be
memory-mapped by a PyTorch dataloader (dataset.py).
"""

import argparse
import functools
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import chess
import numpy as np
import zstandard as zstd

NUM_PIECE_TYPES_NO_KING = 5 # Pawn, Knight, Bishop, Rook, Queen (in HalfKP king gets no feature)
NUM_PIECE_CODES = NUM_PIECE_TYPES_NO_KING * 2   # x2 for "friend" vs "enemy"
FEATURES_PER_KING_SQUARE = NUM_PIECE_CODES * 64
NUM_HALFKP_FEATURES = 64 * FEATURES_PER_KING_SQUARE  # active features per perspective

MAX_ACTIVE_FEATURES = 30 # 32 squares - 2 kings = 30 max non-king pieces on any legal board
FEATURE_PAD_VALUE = 0xFFFF # sentinel for unused feature slots

PIECE_TYPE_TO_CODE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

MATE_CP = 30000
MATE_DIST_CAP = 1000

RECORD_DTYPE = np.dtype([
    ("white_features", np.uint16, MAX_ACTIVE_FEATURES),  # HalfKP indices, White's own perspective
    ("black_features", np.uint16, MAX_ACTIVE_FEATURES),  # HalfKP indices, Black's own perspective
    ("piece_count",    np.uint8),    # how many of the 30 slots above are real (rest are FEATURE_PAD_VALUE)
    ("score_cp",       np.int16),    # side-to-move-relative score, mate pre-encoded
    ("depth",          np.uint8),    # search depth of the eval entry we kept
    ("knodes",         np.uint32),   # kilonodes searched
    ("flags",          np.uint8),    # bit 0 = is_mate, bit 1 = side-to-move-is-black
    ("phase",          np.uint8),    # 0 (bare kings) .. 24 (full material)
])
RECORD_SIZE_BYTES = RECORD_DTYPE.itemsize

FLAG_IS_MATE = 0b00000001
FLAG_STM_IS_BLACK = 0b00000010

ESTIMATED_TOTAL_POSITIONS = 394_669_566 # number of positions in data I used from lichess

# Small helpers
def mirror_square(square: int) -> int:
    """Vertically mirror a 0-63 square index (flip rank, keep file)."""
    return square ^ 56


def compute_phase(board: "chess.Board") -> int:
    """Cheap material-based game phase: 0 (bare kings) .. 24 (full material)."""
    weights = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    phase = 0
    for piece_type, weight in weights.items():
        phase += weight * len(board.pieces(piece_type, chess.WHITE))
        phase += weight * len(board.pieces(piece_type, chess.BLACK))
    return min(phase, 24)


def deterministic_uniform(key: str) -> float:
    """A reproducible pseudo-random float in [0, 1), derived purely from a
    string key (not from any shared RNG state)."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def build_halfkp_features(board: "chess.Board"):
    white_king_sq = board.king(chess.WHITE)
    black_king_sq_mirrored = mirror_square(board.king(chess.BLACK))

    white_feats = []
    black_feats = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue # n feature for kings

        piece_code = PIECE_TYPE_TO_CODE[piece.piece_type]

        is_friend_for_white = piece.color == chess.WHITE
        code_w = piece_code + (0 if is_friend_for_white else NUM_PIECE_TYPES_NO_KING)
        white_feats.append(white_king_sq * FEATURES_PER_KING_SQUARE + code_w * 64 + square)

        # Mirroring for change in perspective
        is_friend_for_black = piece.color == chess.BLACK
        code_b = piece_code + (0 if is_friend_for_black else NUM_PIECE_TYPES_NO_KING)
        mirrored_square = mirror_square(square)
        black_feats.append(
            black_king_sq_mirrored * FEATURES_PER_KING_SQUARE + code_b * 64 + mirrored_square
        )

    return white_feats, black_feats

def process_line(line: str, args: argparse.Namespace):
    """_process_line_inner wrapper"""
    try:
        return _process_line_inner(line, args)
    except Exception:
        return None


def _process_line_inner(line: str, args: argparse.Namespace):
    line = line.strip()
    if not line:
        return None

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None  # corrupted line

    fen_stub = record.get("fen")
    evals = record.get("evals")
    if not fen_stub or not evals:
        return None

    def _sort_key(e):
        d = e.get("depth")
        k = e.get("knodes")
        return (d if d is not None else -1, k if k is not None else -1)

    best = max(evals, key=_sort_key)
    depth = best.get("depth")
    knodes = best.get("knodes")
    pvs = best.get("pvs")

    if depth is None or depth < args.min_depth:
        return None
    if not pvs:
        return None

    top = pvs[0]
    mate_val = top.get("mate")
    is_mate = mate_val is not None

    if is_mate:
        # knodes limit should not be applied for mates
        pass
    else:
        cp_val = top.get("cp")
        if cp_val is None:
            return None
        if knodes is None or knodes < args.min_knodes:
            return None

    full_fen = f"{fen_stub} 0 1" # complete the fen
    try:
        board = chess.Board(full_fen)
    except ValueError:
        return None

    if not board.is_valid():
        return None

    if not any(True for _ in board.legal_moves):
        return None

    total_pieces_on_board = len(board.piece_map())
    piece_count = total_pieces_on_board - 2  # minus the two kings
    if piece_count > MAX_ACTIVE_FEATURES or piece_count < 0:
        return None # just checking

    captured = 32 - total_pieces_on_board
    castling_full = (
        board.has_kingside_castling_rights(chess.WHITE)
        and board.has_queenside_castling_rights(chess.WHITE)
        and board.has_kingside_castling_rights(chess.BLACK)
        and board.has_queenside_castling_rights(chess.BLACK)
    )
    if castling_full and captured <= 2:
        keep_prob = args.opening_keep_prob_early
    elif captured <= 4:
        keep_prob = args.opening_keep_prob_mid
    else:
        keep_prob = 1.0

    if keep_prob < 1.0 and deterministic_uniform(f"{args.seed}:opening:{fen_stub}") >= keep_prob:
        return None

    if is_mate:
        dist = min(abs(mate_val), MATE_DIST_CAP)
        score_white_relative = (MATE_CP - dist) * (1 if mate_val > 0 else -1)
    else:
        score_white_relative = cp_val

    score_white_relative = max(-32000, min(32000, score_white_relative))

    score_stm = -score_white_relative if board.turn == chess.BLACK else score_white_relative

    # HalfKP features
    white_feats, black_feats = build_halfkp_features(board)
    n = len(white_feats)

    rec = np.zeros(1, dtype=RECORD_DTYPE)
    rec["white_features"][0, :n] = white_feats
    rec["white_features"][0, n:] = FEATURE_PAD_VALUE
    rec["black_features"][0, :n] = black_feats
    rec["black_features"][0, n:] = FEATURE_PAD_VALUE
    rec["piece_count"] = n
    rec["score_cp"] = score_stm
    rec["depth"] = min(depth, 255)
    rec["knodes"] = min(knodes or 0, 2**32 - 1)
    rec["flags"] = (FLAG_IS_MATE if is_mate else 0) | (FLAG_STM_IS_BLACK if board.turn == chess.BLACK else 0)
    rec["phase"] = compute_phase(board)

    return rec.tobytes()

def iter_lines(zst_path: str, start_line: int, limit: int | None):
    """Yield decoded text lines from a .zst file, decompressing on the fly."""
    import io

    with open(zst_path, "rb") as raw_fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw_fh) as decompressed_stream:
            text_stream = io.TextIOWrapper(decompressed_stream, encoding="utf-8", errors="replace")
            end_line = None if limit is None else start_line + limit
            for i, line in enumerate(text_stream):
                if i < start_line:
                    continue
                if end_line is not None and i >= end_line:
                    break
                yield line

# Main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to lichess_db_eval.jsonl.zst")
    parser.add_argument("output_dir", help="Directory to write positions.bin / index files / metadata.json into")
    parser.add_argument("--min-depth", type=int, default=16)
    parser.add_argument("--min-knodes", type=int, default=1000, help="Only applied to non-mate evals")
    parser.add_argument("--opening-keep-prob-early", type=float, default=0.10)
    parser.add_argument("--opening-keep-prob-mid", type=float, default=0.50)
    parser.add_argument("--train-frac", type=float, default=0.96)
    parser.add_argument("--val-frac", type=float, default=0.02)
    # test-frac is implicitly 1 - train_frac - val_frac
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--chunksize", type=int, default=2000, help="Lines handed to each worker at a time")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process N lines starting at --start-line (for test purpose)")
    parser.add_argument("--start-line", type=int, default=0,
                         help="Skip this many lines before processing")
    parser.add_argument("--progress-every", type=int, default=200_000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = out_dir / "positions.bin"

    print(f"Record size: {RECORD_SIZE_BYTES} bytes")
    print(f"HalfKP feature space size (per perspective): {NUM_HALFKP_FEATURES}")
    print(f"Workers: {args.workers}, chunksize: {args.chunksize}")
    print(f"Filters: min_depth={args.min_depth}, min_knodes(cp-only)={args.min_knodes}")
    if args.start_line or args.limit is not None:
        end_desc = "EOF" if args.limit is None else f"{args.start_line + args.limit:,}"
        print(f"Shard range: lines [{args.start_line:,}, {end_desc})")
    print("Starting conversion...\n")

    start_time = time.time()
    n_seen = 0
    n_kept = 0

    worker_fn = functools.partial(process_line, args=args)

    with open(positions_path, "wb") as out_fh:
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(worker_fn, iter_lines(args.input, args.start_line, args.limit), chunksize=args.chunksize):
                n_seen += 1
                if result is not None:
                    out_fh.write(result)
                    n_kept += 1

                if n_seen % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    rate = n_seen / elapsed
                    pct = 100.0 * n_seen / ESTIMATED_TOTAL_POSITIONS
                    eta_seconds = (ESTIMATED_TOTAL_POSITIONS - n_seen) / rate if rate > 0 else float("inf")
                    print(
                        f"  seen={n_seen:,}  kept={n_kept:,} ({100.0*n_kept/n_seen:.1f}%)  "
                        f"rate={rate:,.0f} pos/s  elapsed={elapsed/60:.1f}m  "
                        f"~{pct:.2f}% of estimated total  ETA={eta_seconds/3600:.1f}h"
                    )

    elapsed_total = time.time() - start_time
    print(f"\nDone. seen={n_seen:,}  kept={n_kept:,}  elapsed={elapsed_total/60:.1f} minutes")

    print("\nBuilding train/val/test index split...")
    rng = np.random.default_rng(args.seed)
    indices = np.arange(n_kept, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(n_kept * args.train_frac)
    n_val = int(n_kept * args.val_frac)
    # whatever remains goes to test, so the three splits always sum to n_kept exactly
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    np.save(out_dir / "test_idx.npy", test_idx)

    metadata = {
        "record_size_bytes": RECORD_SIZE_BYTES,
        "num_halfkp_features_per_perspective": NUM_HALFKP_FEATURES,
        "max_active_features": MAX_ACTIVE_FEATURES,
        "n_seen": n_seen,
        "n_kept": n_kept,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "opening_keep_prob_early": args.opening_keep_prob_early,
        "opening_keep_prob_mid": args.opening_keep_prob_mid,
        "seed": args.seed,
        "mate_cp": MATE_CP,
        "mate_dist_cap": MATE_DIST_CAP,
        "elapsed_seconds": elapsed_total,
        "flags_bit_meaning": {
            "bit0_is_mate": int(FLAG_IS_MATE),
            "bit1_side_to_move_is_black": int(FLAG_STM_IS_BLACK),
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote:\n  {positions_path}\n  {out_dir/'train_idx.npy'} ({len(train_idx):,})\n"
          f"  {out_dir/'val_idx.npy'} ({len(val_idx):,})\n  {out_dir/'test_idx.npy'} ({len(test_idx):,})\n"
          f"  {out_dir/'metadata.json'}")

if __name__ == "__main__":
    main()