# Search & Speed Techniques — `search.cpp` Reference

## Core search

- **Negamax** — single `search()` for both sides, sign-flipped on recursion.
- **Alpha-beta pruning** — stop exploring once `alpha >= beta`.
- **PVS** — move 0 searched full-window; later moves get a cheap zero-window probe, re-searched full-window only if it beats alpha.
- **Iterative deepening** — depth 1, 2, 3... until time/depth runs out.
- **Aspiration windows** — depth ≥ 5 starts narrow (`prevScore ± 40`), doubles and re-searches on fail-high/low.

## Move ordering

TT move → winning/equal captures (by SEE) → killers (2/ply) → countermove → history heuristic → losing captures.

- **SEE** — plays out the capture sequence on a square to get a signed material value; used for ordering and filtering.
- Insertion sort over `std::sort`: lists are short and near-sorted already.

## Quiescence search

Runs at leaves instead of a raw static eval.
- Not in check: stand-pat cutoff, captures only, skip SEE<0 captures.
- In check: no stand-pat, search all evasions.
- **Delta pruning**: skip entirely if `standPat + queen value + 200 < alpha`.

## Transposition table

256MB default, 16 bytes/entry. Stores key/score/move/depth/bound/age.
- Replace if slot is empty, stale (old generation), or new depth ≥ stored depth.
- Mate scores stored ply-independent, adjusted on read/write.

## Pruning & reductions

- **Mate-distance pruning** — clamp window to best/worst possible mate from this ply.
- **Reverse futility pruning** — cut if `staticEval - margin >= beta` (shallow depth, non-PV, non-check).
- **Razoring** — depth ≤ 3, staticEval far below alpha → drop to qsearch, trust result.
- **IIR** — no TT move at depth ≥ 4 → reduce depth by 1 (bad move ordering not worth full depth).
- **Null-move pruning** — search a passed move at reduced depth; beat beta → cut. Verified at depth ≥ 8. Skipped in check/PV/root/zugzwang-risk.
- **Futility pruning** — depth ≤ 3, skip quiet moves if `staticEval + margin <= alpha` (unless killer/countermove/high history).
- **LMR** — reduce depth for later quiet moves via `log(depth)·log(moveIndex)`; less for PV/killers, more for bad history. Re-search full depth if it beats alpha.
- **Improving flag** — compares staticEval to 2-plies-ago; widens margins above when not improving.

## Extensions

- **Check extension** — +1 ply when in check, capped at 16 total.

## Tablebases (Syzygy/Fathom)

- Root probe: instant exact move if position is covered, no search needed.
- Mid-search probe: folds WDL result into alpha-beta window like a TT bound, cached in TT.

## Evaluation

- **NNUE**: incremental accumulator updates per move (add/remove piece), full refresh only on king moves/castling.
- Falls back to classical eval if NNUE unloaded/disabled.

## Time management

- Soft (`optimalMs`) / hard (`maximumMs`, up to 3x) budget from clock+increment, capped at 85% remaining, 50ms reserve.
- Instability (best move changed / score dropped 50+) relaxes the soft limit toward the hard one.
- Time checked every 2048 nodes, not every node.