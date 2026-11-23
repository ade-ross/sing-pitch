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
