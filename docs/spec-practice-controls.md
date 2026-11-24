---
type: prd
name: Sing Pitch App — Practice Controls
status: ready-for-agent
updated: 2026-11-24
glossary: wiki/interests/sing-pitch-app/CONTEXT.md
parent: docs/interests/sing-pitch-app/prd/2026-11-23-sing-pitch-app.md
---

# PRD — Practice Controls (Guide vocal, Sections, Speed)

Grilled 2026-11-24. Terms (Guide vocal, Section, Pass, Lead-in, Speed, Take, Backing, Melody) are defined in
`wiki/interests/sing-pitch-app/CONTEXT.md`.

## Problem Statement

Simone can now sing a whole song against its Backing, from start to end, at full speed. That is not how anyone learns a
song. He cannot hear the original singer while learning a part, so he has no model to copy. He cannot jump to the
chorus he wants to work on, and he cannot repeat a hard line again and again. A fast passage plays at full tempo before
he can follow it. Every Take is also one long run through the song, so History can't show whether one hard line is
getting better.

## Solution

Melody Takes get a practice transport:

- **Guide vocal:** an on/off toggle plus two volume sliders, one for the Guide vocal and one for the Backing. All of
  them can be changed while he sings.
- **Timeline:** shows the whole song with the reference melody. He clicks it to jump anywhere, and drags across it to
  mark a Section, which he can save by name.
- **Loop:** a looped Section repeats with a ~2 s Lead-in before each Pass. Each Pass is scored and saved as its own
  Take.
- **Speed:** 50–120% in 5% steps, with pitch unchanged. The scoring line follows the new speed.
- **Take settings:** each Take records its Guide vocal state, Speed and Section, and History can filter on them.
- **Keyboard:** shortcuts cover play/pause, loop, jumps and Speed, so his hands stay free.

## User Stories

1. As Simone, I want to switch the Guide vocal on during a Melody Take, so that I can hear how the singer phrases a line.
2. As Simone, I want to switch the Guide vocal off mid-song, so that I can try the same part alone without stopping.
3. As Simone, I want a Guide vocal volume slider, so that the singer can sit quietly under my voice.
4. As Simone, I want a Backing volume slider, so that I can turn the band down and hear the melody better.
5. As Simone, I want both sliders and the toggle to act instantly while singing, so that I adjust without breaking flow.
6. As Simone, I want my slider settings remembered per song, so that a song opens the way I left it.
7. As Simone, I want a timeline of the whole song with the reference melody drawn on it, so that I can see where the
   sung parts are.
8. As Simone, I want to click the timeline to jump there, so that I can go straight to the part I want.
9. As Simone, I want ←/→ to jump 5 s, so that I can nudge position without the mouse.
10. As Simone, I want Space to play and pause, so that I can stop and restart quickly.
11. As Simone, I want a jump outside a loop to end the current Take and start a new one, so that every Take is one
    unbroken stretch and scores honestly.
12. As Simone, I want Takes with less than about 3 s of singing discarded, so that stray clicks don't fill History.
13. As Simone, I want to drag across the timeline to mark a Section, so that I choose exactly the bit to practise.
14. As Simone, I want to adjust a Section's start and end by dragging its edges, so that I can fix a sloppy mark.
15. As Simone, I want to save a Section under a name like "Chorus 1", so that I can come back to it another day.
16. As Simone, I want to pick, rename and delete saved Sections per song, so that the list stays useful.
17. As Simone, I want L to switch looping of the marked Section on and off, so that I repeat a hard line hands-free.
18. As Simone, I want ~2 s of song played before the Section on every Pass, so that I can come in on time.
19. As Simone, I want the Lead-in never scored, so that it can't count against me.
20. As Simone, I want each Pass scored and saved as its own Take, tagged with the Section, so that I can see
    pass-by-pass progress.
21. As Simone, I want a short result per Pass shown while the loop continues, so that I know how that Pass went
    without stopping.
22. As Simone, I want to change Speed from 50% to 120% in 5% steps, so that I can learn fast passages slowly and then
    build up.
23. As Simone, I want +/− to change Speed by 5%, so that I can step tempo without the mouse.
24. As Simone, I want the pitch unchanged at any Speed, so that I still practise the real notes.
25. As Simone, I want the reference melody and the scoring to follow the Speed, so that I'm scored correctly when
    slowed down.
26. As Simone, I want a clear "preparing 70%…" message when a new Speed is being made, so that I know the app hasn't
    frozen.
27. As Simone, I want prepared Speeds kept, so that switching back to one is instant.
28. As Simone, I want a Speed change during a loop to apply from the next Pass, so that a Pass is never half at one
    speed.
29. As Simone, I want each Take to record Guide vocal on/off, Speed and Section, so that easy and hard attempts are
    never mixed silently.
30. As Simone, I want History filterable by Guide vocal, Speed and Section, so that I compare like with like.
31. As Simone, I want replay and re-scoring of a Pass to reproduce its numbers, so that saved data stays trustworthy.
32. As Simone, I want the calibrated latency to stay correct after jumps, loops and Speed changes, so that sync never
    drifts.

## Implementation Decisions

- **Transport (new seam):** one module owns song playback for Melody Takes. It mixes Backing and Guide vocal with
  gains that can change live, handles play/pause, seek, Section loop with Lead-in, and Speed. Its core is pure logic
  over arrays: given a requested block size it returns the audio block plus the song time of each rendered frame. The
  sounddevice stream is a thin shell around it, as the Drone and Backing players already are.
- **Song-time log replaces the constant offset:** today Melody sync assumes linear playback and uses one latency
  offset per Take. After seek, loop or Speed that no longer holds. The Transport therefore keeps a log of output frame
  index to song time (original-tempo seconds). The Take engine scores a voice frame heard at input time t against the
  reference at the song time that was rendered at t − calibrated latency. The latency itself is still wall-clock and is
  still measured by the existing calibration.
- **Pass and Take boundaries:** the Transport emits events (pass started, pass ended, jumped, speed changed). The
  live runner turns them into Take boundaries. A jump outside a loop closes the Take and opens a new one. Loop
  wrap-around closes one Pass Take and opens the next. A Take with under ~3 s of scored singing is discarded; a Pass
  Take is kept when it scored at least half of its Section's singable time (so Sections under 3 s still save). Lead-in
  frames are marked as not scored.
- **Speed:** pitch-preserving time-stretch with Rubber Band (best quality available; the implementer verifies an
  install that works on Apple Silicon). The Backing and the vocal stem are each pre-rendered whole at the chosen
  Speed, then cached per song and Speed in the song's library folder. The reference line needs no re-extraction,
  because song time stays in original seconds and only the time map changes. A Speed change during a loop takes effect
  at the next Pass.
- **Sections:** stored per song in the library as name, start and end (original-tempo seconds). Unnamed marks are
  kept until replaced.
- **Per-song player settings:** Guide vocal on/off, the two volumes, and the last Speed and Section, stored per song.
- **Take record:** new summary-table columns `guide_vocal` (on, off or mixed when it changed mid-Take), `guide_gain`,
  `speed_pct`, `section_name` (or start–end), `pass_index`. Existing rows read as guide off, speed 100, no section.
- **UI:** a timeline widget with the reference melody, playhead, click to seek, drag for a Section and handles for its
  edges. Controls for Guide vocal and both sliders, a Speed stepper, a Section picker and loop toggle. Shortcuts: Space,
  L, ←/→ (5 s), +/−. The History view gains the three filters.
- **Build order:** Guide vocal plus both sliders first, so they are usable on their own while the rest is built. Then
  the Transport with seek and the song-time log. Then Sections and loop with Passes. Then Speed. Then History filters
  and shortcuts.

## Testing Decisions

- **Seams:** the existing Take engine seam stays the main one. The one new seam is the Transport's pure core, which
  takes a block request and returns audio plus song times. Both are driven from outside only, with synthesised stems
  (sine "vocal", noise "backing") and fake streams, never a sound card.
- **Transport tests:**
  - Gains mix linearly and change at the next block.
  - Seek moves song time.
  - A loop wraps from end to start − Lead-in and emits pass events.
  - At 50% Speed, song time advances half as fast per output frame.
- **Sync test (the key one):** a synthetic voice that exactly follows the reference melody, delayed by the
  calibrated latency, scores ~100% in tune across a seek, a loop wrap and a Speed change. Without the song-time log,
  the same voice must fail.
- **Pass tests:**
  - Three loop wraps give three Takes, each tagged with Section and pass index, with Lead-in frames unscored.
  - A Take under 3 s is discarded; a Pass Take under half its Section is discarded.
  - Re-scoring a Pass Take reproduces its numbers.
- **Speed tests:** a stretched stem keeps its pitch (a 220 Hz tone stays within a few cents after stretching), and
  its length is the original divided by the Speed fraction. A cached Speed is not recomputed.
- **Manual:** Simone listens to stretched audio quality at 70% and 50%, checks the Guide vocal toggle and sliders
  feel instant, and checks the loop Lead-in lets him come in on time.
- **Prior art:** the tests for the Melody Take, octave shift, Unscored stretches and backing player.

## Out of Scope

- changing key (transpose); a later ticket if songs sit badly in his voice
- automatic verse/chorus detection
- Speeds outside 50–120%
- per-section Speed presets
- the earlier unanswered offers (song choice switching to Melody by itself; never auto-picking an earbud mic)
- an import button in the window

## Further Notes

The Lead-in is about 2 s by default. It is the planner's choice: long enough to hear the beat at 50% Speed, and short
enough not to waste time at 100%.

## Notes mode (#19)

The Notes box shows and scores the Melody as flat bars on exact semitones, each labelled with its note name, instead of
the continuous line; the Guide lines become bars too, an octave either side, and the bars run through both halves of
the centred plot. The bars come from one pure function of the reference line, its scored mask and the Minimum note
length (`reference/notes.py`): the scored frames are split into reference Phrases (pauses of 0.5 s or more), rounded
to semitones, and any run shorter than the Minimum note length is merged into its nearer-pitched neighbour until none
is left; each note takes the semitone held longest over its span (a slide E-F-F#-G held on G is a G, never the mean).
The slider runs 0.1-4 s; its far end gives one note per Phrase. Unscored stretches stay unscored. In Notes mode the
target is the note's semitone (the Octave shift applies as before), so live scoring, replay and re-score use the same
bars and reproduce the numbers; the segmentation is cached per song and length. Each Take records `note_mode` (off/on)
and `min_note_s` (seconds, or `phrase`) at the end of takes.csv; old rows read as off. History filters by Notes mode,
and the toggle and length are remembered per song in player.json. Both are locked during a Take.
