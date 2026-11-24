---
type: prd
name: Sing Pitch App
status: ready-for-agent
updated: 2026-11-23
glossary: wiki/interests/sing-pitch-app/CONTEXT.md
---

# PRD — Sing Pitch App

Terms (Take, Reference, Nearest note, Key, Melody, Backing, Drone, Unscored stretch, Octave shift, Band, In tune,
Steadiness) are defined in `wiki/interests/sing-pitch-app/CONTEXT.md`. Decisions come from the grilling of 2026-11-23.

## Problem Statement

Simone wants to start singing, and the first thing he cannot tell on his own is whether he is in tune. When he sings
he has no pointer showing how far off he is or which way to move his voice, and nothing records whether he gets better
over weeks. Existing tools either give live pitch with no logging (Singing Carrots, Vocal Lab), score karaoke while
forgiving octave errors (AllKaraoke), or analyse only recordings after the fact (Praat, Tony). None of them does live
feedback, all three Reference types, songs he picks and a progress history in one place. He wants the best personal app
built for this. Understanding how it is built is explicitly not a goal for this project (his words, 2026-11-23).

## Solution

A Python desktop app on his Mac. He sings into the Blue Yeti (or the built-in mic) and sees his pitch as a live line on
a note grid, coloured by Band, against one of three References:

- **Nearest note**: works like a tuner.
- **Key**: he picks a key, notes outside it are flagged, and an optional Drone can play.
- **Melody**: he imports a song (a file or a YouTube link). The app splits out the vocal and the Backing, extracts the
  vocal's pitch line, then plays the Backing in his headphones while scoring him in sync.

Every Take is saved with its raw pitch line and five numbers. A History view plots his progress over time. Voice only:
he does not play guitar while using it.

## User Stories

1. As Simone, I want to open the app and start singing within seconds, so that using it costs no setup effort.
2. As Simone, I want the app to choose the Blue Yeti automatically when it is plugged in, so that I don't pick the
   input every time.
3. As Simone, I want to switch the input to the built-in Mac mic, so that I can sing when the Yeti is not connected.
4. As Simone, I want to see my live pitch as a line on a note grid, so that I can see where my voice is.
5. As Simone, I want the line coloured green within ±25 cents, yellow from 25 to 50 and red beyond ±50, so that I can
   see at a glance whether I am in tune.
6. As Simone, I want to see the current note name and how many cents sharp or flat I am, so that I know which way to
   move my voice.
7. As Simone, I want the display to lag no more than about a quarter of a second, so that I can correct while I sing.
8. As Simone, I want silence, breaths and background noise not drawn as pitch, so that the line shows only my voice.
9. As Simone, I want to press Start and Stop to make a Take, so that each attempt is scored on its own.
10. As Simone, I want a summary right after each Take (% in tune, median cents off, tendency sharp or flat,
    Steadiness on held notes, lowest and highest note), so that I know how that attempt went.
11. As Simone, I want every Take saved automatically with its raw pitch line, so that nothing is lost and old Takes
    can be re-scored if the Bands change.
12. As Simone, I want to replay a saved Take's pitch line, so that I can see where it went wrong.
13. As Simone, I want to choose the Reference (Nearest note, Key or Melody) before a Take, so that "in tune" means what
    I am practising.
14. As Simone, I want to pick a key from a list, so that notes outside it are flagged in a Key Take.
15. As Simone, I want an optional Drone of the key note in my headphones, so that I have something to sing against.
16. As Simone, I want to import a song from an audio file on my Mac, so that I can sing songs I already have.
17. As Simone, I want to import a song from a YouTube link, so that I can sing any song.
18. As Simone, I want the app to split the vocal and the Backing itself in one command, so that I don't do it by hand.
19. As Simone, I want to drop in stems I split with Logic's Stem Splitter instead, so that I have a fallback when the
    automatic split is poor.
20. As Simone, I want the app to extract the original singer's pitch line from the vocal stem, so that my voice is
    compared with the real melody.
21. As Simone, I want to hear the Backing in my headphones during a Melody Take, so that I sing along with the song.
22. As Simone, I want my voice scored in sync with the song, with the app's own delays compensated, so that a correct
    note is never marked wrong for being late.
23. As Simone, I want the reference melody line drawn next to mine, so that I can see what I should sing.
24. As Simone, I want stretches where the extracted reference is unreliable greyed out and left out of every score, so
    that extraction errors never count as my errors.
25. As Simone, I want the app to detect the Octave shift between me and the original singer for each Phrase, so that
    singing a song an octave lower is not marked wrong.
26. As Simone, I want octave jumps within a Phrase not forgiven note by note, so that real errors are not hidden.
27. As Simone, I want imported songs kept in a library, so that I can sing a song again without re-importing it.
28. As Simone, I want a History view plotting % in tune and median cents off over dates, so that I can see whether I
    am improving.
29. As Simone, I want History filterable by Reference and by song, so that I compare like with like.
30. As Simone, I want every Take recorded in one table I can open outside the app, so that the data is mine and
    analysable.
31. As Simone, I want the app to warn me when Melody playback goes to speakers and not headphones, so that the Backing
    does not leak into the mic and spoil the score.
32. As Simone, I want the app to work offline once the models are downloaded, so that it does not depend on a service.

## Implementation Decisions

- **Platform:** a Python desktop app with a pyqtgraph/Qt window, with Qt and pyqtgraph versions pinned because of a
  known macOS performance bug. It lives in its own git repo under `~/Builds/` (`~/Builds/sing-pitch/`) with a private
  GitHub remote. It is not under `~/Work/` and not in the vault.
- **Licences:** not checked. The app is for personal use only (his decision).
- **Pitch tracker:** SwiftF0 in streaming mode: MIT, 16 kHz, 16 ms hop, 176 ms lookahead, a confidence and loudness
  value per frame. It was best of 19 trackers in lars76/pitch-benchmark, a benchmark written by its own author. Frames
  below a confidence or loudness gate are unvoiced. The tracker sits behind one small interface, so it can be swapped
  for PESTO or RMVPE if testing on his voice disagrees.
- **Reference-melody extraction:** RMVPE (made for vocal pitch, robust to leftover accompaniment) runs on the vocal stem
  at import time. Real-time speed is not required here. Low-confidence frames become Unscored stretches.
- **Stem separation:** the current best open vocal separator, a RoFormer-family model through the audio-separator
  package. The exact model is picked by listening at build time. A folder of Logic stems can be used instead.
- **Song download:** yt-dlp.
- **Modules:**
  - **Audio input:** device choice (Yeti preferred) and a chunk stream.
  - **Pitch tracker:** chunks in, frames out (time, Hz, confidence, loudness).
  - **Reference:** one interface with three kinds. For any time and frequency it gives the target pitch in cents, or
    "unscored". Nearest note means equal temperament at A4 = 440 Hz. Key means in-key or out-of-key plus the nearest
    in-key note. Melody means the extracted line plus the Octave shift.
  - **Take engine (the main seam):** frames plus a Reference in, live Band and cents per frame out, then on Stop a Take
    record. Pure logic with no audio device and no UI.
  - **Take store:** each Take is saved as its raw pitch line plus one row in a single summary table.
  - **Song library / importer:** download, separate, extract, store.
  - **Playback:** Drone, or the Transport (Backing + Guide vocal; pure core + thin sounddevice shell) whose song-time
    log the Take engine reads.
  - **UI:** live view, Take summary, History.
- **Scoring:**
  - **Bands:** green within the Tolerance, yellow up to twice it, red beyond. The Tolerance defaults to ±25 cents
    (green ±25, yellow to ±50). The ±50 line is the research convention for "in tune" (Berkowska & Dalla Bella 2013);
    the ±25 line is the planner's choice, set near where untrained listeners start to hear an error (about 20–30
    cents, Larrouy-Maestri group, abstracts only). Simone sets the Tolerance with a slider (±10 to ±100 cents, 5-cent
    steps; #41). It applies to every Reference, is remembered in settings.json, is locked during a Take and is saved
    with each Take (`tolerance_cents`); the summary, the History legend and a History filter show it.
  - **Timing window (Melody, QA #44):** each Phrase gets one timing shift, how late (+) or early (−) the voice is,
    searched from −100 to +100 ms in 10 ms steps. It is chosen together with the Phrase's Octave shift, in one search:
    the pair (k, shift) whose Reference, the line at song time − shift plus 12 k, gives the smallest median absolute
    error over the Phrase's scored frames (ties within 0.5 c go to the smaller mean error with each frame capped at
    300 c, then the smaller |k|, then the smaller |shift|; a pinned Octave searches the shift only). Every frame of the
    Phrase is then scored strictly at that shift, so it forgives timing, not pitch: a voice 40 cents sharp the whole
    time scores ~7 % in tune and shows ~40 c. The line is read between its 10 ms frames where two neighbours are within
    a semitone (a glide or vibrato), at the nearest frame across a note change and on Notes-mode bars. A shift never
    moves a frame across an Unscored stretch (it is held at the edge of the scored run), so whether a frame is scored
    does not change. The held note for Steadiness is still read at the frame's own song time. Live, the open Phrase's
    pair is re-estimated as it grows; after ~2 s of scored singing its octave is kept and its shift re-estimated each
    time the Phrase's scored frames double; the summary uses whole-Phrase pairs. Saved as `timing_window_s` and
    `timing_shift_ms` (the Take's median shift over its scored frames). History of the rule: #41 took, per frame, the
    reference frame within ±100 ms closest to the voice; on a vocal with vibrato that hid a constant 40 c error (75 %
    in tune, 8 c). Rows without `timing_shift_ms` re-score with that #41 rule.
  - **% in tune:** green time divided by voiced, scored time. Notes mode (#41): the green share of the sung notes'
    length, each note judged once (see spec-practice-controls.md, Notes mode).
  - **Median cents off:** median of the absolute error. **Tendency:** sign of the median signed error. In Notes mode
    both are the length-weighted medians over the notes' verdicts.
  - **Old Takes:** a row without `tolerance_cents` (saved before #41) is re-scored as it was scored: ±25 cents, no
    Timing window, Notes mode frame by frame (`take_store.row_config`), so its numbers stay as saved. A row with
    `timing_window_s` but no `timing_shift_ms` (saved by #41) re-scores with the per-frame window. A replay and a
    re-score always use the Take's saved scoring, never the slider.
  - **Steadiness:** SD in cents around the note's own median, over held notes longer than about 0.5 s, with the first
    and last ~120 ms or 15% removed (Dai & Dixon 2019).
    A held note is a run of frames on one note of the unmerged Melody line, so Steadiness does not change with the
    Notes slider. Tracker octave blips (a run of up to 0.3 s that sits 1000 cents or more from the median of the
    scored frames a second either side) are left out of Steadiness and of the range. **Range:** the 5th to 95th
    percentile of the scored frames, blips left out. Takes saved before 2026-11-24 keep their saved numbers; a
    re-score uses these rules.
- **Octave shift:** per Phrase (Auto, the default) or pinned for the whole Take. A Phrase is a run of singing between
  pauses of about 0.5 s or more (`phrase_gap_s`); voiced runs shorter than 48 ms do not count. Phrase boundaries are
  a pure function of the frames and their voiced mask, the same live and on re-score. Auto gives each Phrase the
  whole octave k in −2…+2 closest to its Reference (smallest median absolute error, ties to the smaller |k|), so a
  verse sung an octave down and a chorus at pitch in one Take are both in tune. An octave jump inside a Phrase is
  still not forgiven. The Octave box next to Speed offers Auto, −2, −1, 0, +1; a pinned value applies to every Phrase
  of the Take and is remembered per song with the other player settings. Live view: each Phrase's k is re-estimated
  while the Phrase grows, until about 2 s of it (125 scored frames) are scored, then kept for the rest of that
  Phrase; the summary and the saved row use each Phrase's whole-Phrase k, so the first seconds of a Phrase on screen
  can differ from them. The blue line is drawn at the active k (live and on replay, each frame at the k it was scored
  at), with faint Guide lines one octave above and below; the y-window follows the voice and the coming Reference (see Vertical range). `octave_shift` in the
  summary is the pinned k, or for Auto the most common Phrase shift (ties to the one with more scored frames, then
  the smaller |k|); `octave_mode` (auto / pinned) is a new column of takes.csv. Takes saved before this have an empty
  `octave_mode` and re-score with one shift for the whole Take, exactly as they were scored.
- **Vertical range:** 24 semitones (12 either side of a centre in whole semitones), because 4 s of a song spans up to
  ~20. The centre follows the median of the last 2 s of voice and the coming 4 s of the Reference at the Octave shift
  drawn (the voice only when it is within 5 semitones of it, i.e. not an octave off), and it moves only when that
  content would come within 1 semitone of an edge (hysteresis); a move then centres on the next 10 s of the Reference
  too and eases over ~0.3 s. One stray voice frame or one wild Reference frame moves nothing. With no Reference (Nearest
  note, Key) the voice median alone moves it, when it is 4 semitones off the centre.
- **Speed presets and replaying from History:** buttons 70 / 85 / 100 % next to the Speed box set it (same preparation
  and cache path). History lists the Takes (newest first) with a Replay button and double-click; the main window then
  replays that Take with its saved Reference, song, Octave, Notes and Speed on the controls, and puts the controls back
  when the replay ends. Nothing is remembered or stretched by a replay; a missing song or a running Take says so.
- **Unscored stretches:** reference frames with RMVPE confidence under 0.3, smoothed: dips shorter than 0.12 s between
  reliable frames are filled, reliable runs shorter than 0.12 s are dropped. Greyed on the live and replay plot, left
  out of all five numbers and of Steadiness.
- **Centred plot:** the now-line sits in the middle of the plot: the last 4 s on its left (the voice and the Reference),
  the next 4 s on its right (the Reference, its Guide lines and Unscored bands, never the voice). The right half is
  an approximation: it takes the song to go on from where the Transport's rendering ended, at that Speed, wrapping at the
  end of a looped Section to the next Pass's Lead-in and ending at the song end; a seek, pause or Speed change made
  inside those seconds shows once rendered. Paused, it shows what plays on resume. The Octave shift of the last frame
  is held in the right half (a later Phrase may be scored at another one).
  The now-line is the song heard in the headphones now (the newest mic sample's time plus the input latency, i.e. the
  rendered output minus the output part of the calibrated latency), not the newest voice frame: the voice is drawn
  where it was sung, so it ends ~0.2 s left of the line (the tracker's look-ahead). The x axis is voice time with
  every pause of the song folded away (`PauseFold`): while paused the whole plot stands still and singing is neither
  drawn nor scored; on resume it goes on from the same place, and the Take is not split. Replays do the same.
- **Sync:** the Transport logs which song time it rendered at each output time. A voice frame heard at input time t
  is scored against the song time rendered at t − calibrated latency (a LogClock over that log), so pauses (later
  seeks, loops, Speed) keep sync. The song time of every frame is saved with the Take (npz `song_t`); Takes saved
  before this have none and are re-scored with t − `latency_offset_s`, as they were scored.
- **Build order:** Nearest note plus the Take log, then History, then Key plus Drone, then import and stem split, then
  Melody, then Unscored stretches and Octave shift. Each step is usable on its own.

## Testing Decisions

- A good test drives the **Take engine** from the outside, with audio or frames in and the Take record out. It never
  checks internals. Inputs are synthesised:
  - sine tones at known cents offsets
  - a tone with 6 Hz vibrato
  - a held note with a deliberate scoop at the attack
  - silence and noise between notes
  - a melody sung an octave down

  Expected numbers are known exactly. Examples: a note +30 cents off lands in yellow and has a sharp tendency; the scoop
  does not hurt Steadiness; a melody an octave down gives an Octave shift of −1 and full marks.
- Through the same seam: a Key Take flags an out-of-key note, and a Melody Take with an injected unreliable stretch
  leaves it unscored.
- **Importer:** one slow, opt-in test on a short real clip. It checks that the stems and a reference line exist and that
  most frames are voiced where the singer sings. Quality is judged by Simone listening, not by a proxy score (lesson of
  the audio-to-tab bench, 2026-10-29).
- **Latency:** one manual check. Play a click through headphones and sing on it; the measured offset becomes the
  default.
- **UI:** a manual smoke check only, with no automated UI tests.
- **Prior art:** none in the new repo. The tab-trials bench (`~/Work/tab-trials/`) is the nearest audio precedent.

## Out of Scope

- singing while playing guitar, or any guitar-bleed handling
- an acoustic guitar
- breath support, jitter, shimmer, CPP, absolute loudness
- vibrato reporting (later, if he starts using vibrato)
- lyrics on screen
- slowed-down Backing
- Guitar Pro or MIDI as Melody sources (later)
- singing other stems' lines
- auto-detecting the key from his voice
- links into guitar cards or practice planning: it is a tool he opens when he chooses
- mobile, web, sharing

## Further Notes

- Evidence for live visual feedback: Blanco, Tassani & Ramirez 2021, Frontiers in Psychology, 47 beginners. Only the
  group that got visual feedback with right/wrong marking kept their gains.
- Absolute voice-quality numbers from a USB mic at a varying distance are not meaningful, which is why they are out of
  scope.
- Research brief sources (2026-11-23):
  - github.com/lars76/swift-f0
  - github.com/lars76/pitch-benchmark
  - arxiv.org/abs/2306.15412 (RMVPE)
  - pmc.ncbi.nlm.nih.gov/articles/PMC3799299
  - frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.684693
