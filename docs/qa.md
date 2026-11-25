# Driving the app without a mic or speakers (test mode)

Test mode lets an agent (or a script) run the real window, tracker, Take engine and Take store end to end on a fake
microphone and a null output. Nothing is heard and no device is opened. It is switched on only by the flags below or
by the QA driver; a plain `sing-pitch` launch is unchanged.

## How it works

Think of one fake sound card. A single thread (`testmode.FakeAudioHub`) steps 256 frames at a time, paced by the
wall clock. Each step it first plays a step of every open output stream (calling the Backing, Guide vocal, Drone or
calibration callback for exactly that step, just in time), then hands every fake mic one 256-frame chunk, the same
chunk size as the real input. So the Transport's song clock advances exactly as with a
device, and output and input keep a fixed relation.

The fake mic is one of two things:

- **a wav file**: a singer who sings that file in time with the song. It starts reading at `--fake-mic-offset` at the
  moment the output stream started, and is heard `--fake-latency` seconds late. It is silent before that and after
  the file ends (or loops with `--fake-mic-loop`). A pause or seek after the start is not followed: it is a file.
- **`echo`**: the mic hears the output mix itself, `--fake-latency` late. With the Guide vocal on and the Backing at
  0 it sings exactly what plays, through pauses, seeks, loops and Speed, and it answers the calibration clicks.

`--fake-latency` defaults to the latency the window uses for the fake output (named `Fake output` in
settings.json), so either mic is a perfect singer out of the box.

Like a real sound card, the fake stamps its audio with times on one clock (the hub's app time, `FakeAudioHub.now()`):
each output callback gets `time_info.outputBufferDacTime` (when its first frame is played) and each mic chunk is
recorded in `FakeMic.times` (when it was heard). The window keeps sync through those stamps, not by counting frames
(#49), so the faults below are survivable.

## Faults (#49)

A real device under load loses audio: when the UI thread holds Python's GIL (a slow, big-window redraw, a resize), the
audio callbacks run late and CoreAudio drops the block, silently (measured on a Scarlett 2i2: GIL held 42/90/187 ms ->
23/70/162 ms of output lost, no underflow flag). The fake can do the same, so the whole class is testable without
hardware:

| fault | spec | what the fake does |
| --- | --- | --- |
| output underflow | `underflow=EVERY_S:MS` | every EVERY_S app seconds the output plays MS of silence; its callback is not called and the stream stands still, the next block's DAC time shows the gap |
| mic overflow | `overflow=EVERY_S:MS` | every EVERY_S seconds MS of mic input is never delivered |
| clock drift | `drift=PPM` | the output's crystal runs PPM parts per million faster (negative: slower) than the mic's |
| real-time deadlines | `deadline=OUT_MS:IN_MS` | when the hub thread runs late (the GIL held elsewhere), an output step more than OUT_MS late is lost (underflow), a mic step more than IN_MS late too (overflow); without it the hub just catches up |
| UI stall | `ui_stall=EVERY_S:MS[:sleep]` | the UI thread is blocked MS (wall clock) every EVERY_S app seconds, holding the GIL (default: like a slow C++ paint) or sleeping |

Join them with commas: `--fake-faults "deadline=20:60,ui_stall=1:150"` is the realistic one, a GIL hog that makes
both sides lose blocks by different amounts. `QA(home, faults="underflow=0.5:40")` (or a `testmode.Faults`) does the
same in a script; `qa.hub.lost_out_steps` / `lost_in_steps` count what the faults cost.

## Command line

```
SING_PITCH_HOME=/tmp/qa-home sing-pitch --fake-mic <wav|echo> [--fake-mic-offset S] [--fake-mic-loop]
                                        [--fake-latency S] [--fake-output [--record-output out.wav]]
                                        [--fake-faults SPEC]
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
  tracker=None, size=(1200, 700), faults=None)`: the window offscreen (QT_QPA_PLATFORM=offscreen is set if unset), always on the
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
- Sync: `display_error_s()` = song time under the centre line minus song time heard at that moment (the fake output's
  own record of what it played when, one output latency ago), measured right after a redraw; `glitches()` = lost
  output blocks and mic chunks as the window's clocks saw them, and what the fake really lost.
  `tests/test_sync_under_faults.py` holds it under 15 ms mean / 30 ms max under every fault, at Speed 50-120, while
  the window is resized from 640x420 to 3000x1800.
- `close()` (or the `with` block) stops a running Take, closes the window, stops the hub and writes the recording.

## The oracle

Feeding a song's own `vocals.wav` as the file mic, with the mic latency equal to the configured one, must give
about 100 % in tune on a whole-song Melody Take: the voice and the reference line come from the same audio, so any
loss is the sync or the scoring. `tests/test_qa_oracle.py` runs it on a synthetic song (5 notes, about 99.5 %) and
checks that a voice 0.45 s later than the configured latency scores under 60 %. On an imported song the score also
depends on how well SwiftF0 (live) and RMVPE (import) agree on the stem.

Why the null output renders just in time: the window aligns the song clock when the first mic chunk arrives, taking
what the output has rendered by then as what is playing. A real device renders a block or more ahead and calibration
measures that gap in. If the fake rendered 1024-frame blocks ahead, a perfect singer would be 16 ms early, and on a
real vocal with vibrato that alone cost 25 points of % in tune (63 % instead of 88 %). Rendering one step at a time
makes a fake latency of L exactly a calibrated latency of L (`tests/test_fake_audio.py` checks it to the frame).

Not covered: the real mic and speakers, device lists (test mode shows one fake input and the system-default fake
output), and the headphones check (the fake output counts as headphones). The fake's timestamps are exact; how exact
a real device's are (Bluetooth earbuds especially) is not tested here.
