# NNUE Training Pipeline — Commands

Run the scripts in this order.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. `converter.py` — build the dataset

Converts the raw Lichess eval file into `positions.bin` + train/val/test index files.

```bash
python converter.py lichess_db_eval.jsonl.zst ./data
```

- `lichess_db_eval.jsonl.zst` — path to the raw input file
- `./data` — output directory (will contain `positions.bin`, `train_idx.npy`, `val_idx.npy`, `test_idx.npy`, `metadata.json`)

Useful flags: `--min-depth`, `--min-knodes`, `--workers`, `--limit`/`--start-line` (for a quick test run on a small slice).

## 3. `dataset.py` — not run directly

This is just the `NNUEDataset` loader used by `train.py` and `test.py`. No command needed.

## 4. `model.py` — not run directly

Defines the `NNUE` network. No command needed.

## 5. `train.py` — train the model

Reads `./data`, trains the model, writes checkpoints.

```bash
python train.py --data-dir ./data --checkpoint-dir ./checkpoints
```

- `--data-dir` — folder from step 2
- `--checkpoint-dir` — where `checkpoint_latest.pt` / `checkpoint_best.pt` get saved
- Re-running the same command **resumes automatically** from `checkpoint_latest.pt` if it exists
- Useful flags: `--batch-size`, `--lr`, `--max-epochs`, `--patience`, `--eval-clamp-cp`

## 6. `test.py` — evaluate on the held-out test set

```bash
python test.py --data-dir ./data --checkpoint ./checkpoints/checkpoint_best.pt
```

- `--data-dir` — same folder as before
- `--checkpoint` — the trained checkpoint to evaluate
- If you trained with `--eval-clamp-cp`, pass the **same value** here

## 7. `export.py` — export to a quantized `.nnue` file

```bash
python export.py --checkpoint ./checkpoints/checkpoint_best.pt --output model.nnue --verify --real-positions-dir ./data
```

- `--checkpoint` — trained checkpoint
- `--output` — output `.nnue` file path
- `--verify` — sanity-checks quantized vs float outputs before writing
- `--real-positions-dir ./data` — verify against real positions instead of synthetic ones (recommended)
- Add `--force` to write anyway if verification warns/fails