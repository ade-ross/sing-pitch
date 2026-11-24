# sing-pitch

Personal Mac app: live singing pitch against a Reference (Nearest note, Key, Melody), every Take logged, History view.

- Spec: `docs/spec.md` (copy; source of truth is Simone's vault, `docs/interests/sing-pitch-app/prd/`)
- Glossary: `CONTEXT.md` (use these terms in code, issues and commits)
- Work: GitHub issues labelled `ready-for-agent`, each lists what blocks it.

## Run

```
uv sync
uv run sing-pitch            # launches the app (Yeti auto-selected, built-in mic otherwise)
uv run sing-pitch --smoke    # headless check: QT_QPA_PLATFORM=offscreen uv run sing-pitch --smoke
uv run pytest                # fast tests, no microphone needed
```

Takes are saved under `~/Library/Application Support/sing-pitch/` (`takes.csv` + `takes/<id>.npz`), outside git.
Set `SING_PITCH_HOME` to use another folder. Pinned: Python 3.12, pyqtgraph 0.14.0, pyside6-essentials 6.12.0.

## Import a song

```
brew install ffmpeg                        # once
uv sync --extra import                     # heavy: torch, audio-separator, RMVPE, yt-dlp
uv run sing-pitch import "/path/to/song.mp3"
uv run sing-pitch import "https://www.youtube.com/watch?v=..."
uv run sing-pitch import "/path/to/Logic stems folder"   # a Vocals file + other stems: no separation
uv run pytest -m slow                      # opt-in: real separator + RMVPE on a generated clip (~1 min)
```

Songs go to `<data dir>/songs/<song-id>/` (`vocals.wav`, `backing.wav`, `reference.npz` = t, hz, confidence, `meta.json`,
`tempo.json` = bpm + beat times for the Count-in).
Models (`rmvpe.onnx`, `audio-separator/`) go to `<data dir>/models/` (override: `SING_PITCH_MODELS`); the first import
downloads about 1.3 GB, after that import works offline. Importing the same song again returns the stored one
(`--force` redoes it; `--model` picks another separator model).
