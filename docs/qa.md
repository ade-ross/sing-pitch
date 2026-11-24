# Driving the app without a mic or speakers (test mode)

Test mode lets an agent (or a script) run the real window, tracker, Take engine and Take store end to end on a fake
microphone and a null output. Nothing is heard and no device is opened. It is switched on only by the flags below or
by the QA driver; a plain `sing-pitch` launch is unchanged.

## How it works

Think of one fake sound card. A single thread (`testmode.FakeAudioHub`) steps 256 frames at a time, paced by the
wall clock. Each step it first plays a step of every open output stream (calling the Backing, Guide vocal, Drone or
calibration callback whenever its 1024-frame block runs out, as PortAudio does), then hands every fake mic one
256-frame chunk, the same chunk size as the real input. So the Transport's song clock advances exactly as with a
device, and output and input keep a fixed relation.

The fake mic is one of two things:

- **a wav file**: a singer who sings that file in time with the song. It starts reading at `--fake-mic-offset` at the
  moment the output stream started, and is heard `--fake-latency` seconds late. It is silent before that and after
  the file ends (or loops with `--fake-mic-loop`). A pause or seek after the start is not followed: it is a file.
- **`echo`**: the mic hears the output mix itself, `--fake-latency` late. With the Guide vocal on and the Backing at
  0 it sings exactly what plays, through pauses, seeks, loops and Speed, and it answers the calibration clicks.

`--fake-latency` defaults to the latency the window uses for the fake output (named `Fake output` in
settings.json), so either mic is a perfect singer out of the box.

## Command line

```
SING_PITCH_HOME=/tmp/qa-home sing-pitch --fake-mic <wav|echo> [--fake-mic-offset S] [--fake-mic-loop]
                                        [--fake-latency S] [--fake-output [--record-output out.wav]]
```

Use a scratch `SING_PITCH_HOME` (copy a song folder into `<home>/songs/`): Takes are saved there as usual.
`--record-output` writes everything that would have been played to a wav when the app quits.

## The QA driver (`sing_pitch.qa`)

```python
from sing_pitch.qa import QA

with QA(home, mic="songs/x/vocals.wav", time_scale=1.0) as qa:   # mic="echo" is the default
    qa.set("reference_box", "Melody")
    qa.set("song_box", "My Song")
    qa.start_take(at_s=30)          # Start with the clock stopped, seek to 0:30, file mic from 0:30 too
    qa.wait(60)                     # 60 s of app time, the event loop running
    qa.pause(); qa.wait(1); qa.pause()
    qa.seek(10)                     # what a click on the Timeline does
    qa.mark_section(12, 20); qa.set("loop_box", True)
    qa.wait_until(lambda: len(qa.takes()) >= 3, timeout_s=60)
    qa.stop_take()
    print(qa.summary(), qa.takes()[-1]["pct_in_tune"])
    qa.screenshot("shot.png")
```

- `QA(home, mic=ECHO, latency_s=None, mic_offset_s=0, loop_mic=False, record_output=None, time_scale=1.0,
  tracker=None, size=(1200, 700))`: the window offscreen (QT_QPA_PLATFORM=offscreen is set if unset), always on the
  fake output. `time_scale` > 1 runs app time faster than the wall clock (the tracker keeps up to about 10x; tests
  use 4).
- Widgets: `names()`, `widget(name)`, `click(name)`, `set(name, value)` (combo text, check box, slider, spin box,
  line edit), `press(key)` (`"Space"`, `"Left"`, `"Right"`, `"L"`, `"+"`, `"-"`), `text(name)`. Each public widget's
  objectName is its MainWindow attribute name: `button` (Start/Stop), `pause_button`, `reference_box`, `song_box`,
  `key_box`, `drone_box`, `guide_box`, `guide_slider`, `backing_slider`, `speed_box`, `octave_box`, `notes_box`,
  `note_length_slider`, `section_box`, `loop_box`, `replay_button`, `calibrate_button`, `summary_label`, `timeline`,
  `plot_widget`, ...
- Time: `app_time()`, `wait(seconds)`, `wait_until(condition, timeout_s)`; both in app time.
- Takes: `start_take(at_s=None)`, `stop_take()`, `pause()`, `seek(song_s)`, `mark_section(start_s, end_s)`,
  `running`.
- State: `song_position()`, `summary()`, `takes()` (takes.csv rows), `plot_data()` (voice line, reference line,
  now-line, x range), `screenshot(path)`.
- `close()` (or the `with` block) stops a running Take, closes the window, stops the hub and writes the recording.

## The oracle

Feeding a song's own `vocals.wav` as the file mic, with the mic latency equal to the configured one, must give
about 100 % in tune on a whole-song Melody Take: the voice and the reference line come from the same audio, so any
loss is the sync or the scoring. `tests/test_qa_oracle.py` runs it on a synthetic song (5 notes, about 98 %) and
checks that a voice 0.45 s later than the configured latency scores under 60 %. On an imported song the score also
depends on how well SwiftF0 (live) and RMVPE (import) agree on the stem.

One known offset: the window aligns input and output when the first mic chunk arrives, by when the output has
rendered one block ahead, so the fake mic is 16 ms (768 frames) earlier than the latency says. A real calibration
measures that gap in; here it is about one tracker frame.

Not covered: the real mic and speakers, device lists (test mode shows one fake input and the system-default fake
output), and the headphones check (the fake output counts as headphones).
