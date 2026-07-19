# Chess Engine — Quick Guide

## Build

```bash
make          # builds ./chess_engine
make clean    # removes build/ and the binary
```

## UCI Options

Set these after `uci`, before `isready` / `go` (standard UCI `setoption` syntax):

```
setoption name Hash value 256
setoption name BookFile value books/komodo.bin
setoption name SyzygyPath value syzygy
setoption name NNUEFile value nnue/.nnue/default.nnue
setoption name UseNNUE value true
```

| Option        | Type   | What it does                                      |
|---------------|--------|----------------------------------------------------|
| `Hash`        | spin   | Transposition table size in MB. Default 256, range 1–8192. |
| `BookFile`    | string | Path to a Polyglot `.bin` opening book. Set to empty value to disable. |
| `SyzygyPath`  | string | Path to Syzygy tablebase folder. Set to empty value to disable. |
| `NNUEFile`    | string | Path to the `.nnue` eval file to load. Set to empty value to unload. |
| `UseNNUE`     | check  | `true`/`1` to enable NNUE eval, `false`/`0` to disable it (falls back to classical eval). |

Note: the engine also tries to auto-load a book/tablebase/NNUE file at startup from defaults baked into `uci/uci.cpp` (currently empty book/syzygy paths, and `nnue/.nnue/default.nnue` with NNUE on) — the `setoption` commands above override those.

## Downloading the book and Syzygy files

Neither is bundled — `/books/` and `/syzygy/` are gitignored, so if you want to run this engine with either of these, you need to download them yourself before pointing `BookFile` / `SyzygyPath` at them:

- Polyglot opening book: https://github.com/michaeldv/donna_opening_books
- Syzygy tablebases (3-4-5 preferred, ~1GB): http://tablebase.sesse.net/syzygy/3-4-5/