# Sing Pitch App

The language of Simone's personal singing app: it shows his live voice pitch against a Reference and logs how in tune each Take was. Voice only; guitar is out of scope for now.

## Language

**Take**:
One recording from Start to Stop, scored and saved with its raw pitch line.
_Avoid_: session, recording, attempt

**Reference**:
What "in tune" is measured against. Exactly one of three: Nearest note, Key, Melody.
_Avoid_: target, mode

**Nearest note**:
The Reference where each sung moment is compared with the closest equal-tempered note, like a tuner.
_Avoid_: tuner mode, free mode

**Key**:
The Reference where notes outside a key Simone picks are flagged; may play a Drone.
_Avoid_: scale mode

**Melody**:
The Reference where the voice is compared, in sync, with the pitch line extracted from a song's vocal stem while its Backing plays.
_Avoid_: song mode, karaoke

**Backing**:
The song with its vocal removed, played in headphones during a Melody Take.
_Avoid_: instrumental, accompaniment

**Guide vocal**:
The original singer's isolated vocal, optionally played with the Backing during a Melody Take at a volume Simone sets.
_Avoid_: original voice, vocal track

**Section**:
A stretch of a song between a start and an end point, marked by Simone on the timeline and optionally saved by name.
_Avoid_: part, region, loop

**Pass**:
One time through a looped Section; each Pass is scored and saved as its own Take.
_Avoid_: repetition, loop iteration

**Lead-in**:
The ~2 s of song played before a Section's start on every Pass, never scored.
_Avoid_: pre-roll, count-in

**Speed**:
The playback tempo of the song as a percentage of the original (50–120 %), with pitch unchanged.
_Avoid_: tempo, rate

**Drone**:
A held key note played in headphones during a Key Take.

**Unscored stretch**:
A part of a Melody where the extracted reference pitch is unreliable; shown greyed out and left out of every score.
_Avoid_: gap, ignored part

**Octave shift**:
The whole-octave offset between Simone's voice and the original singer, chosen per Phrase (Auto) or pinned by Simone for the whole Take.

**Phrase**:
A run of singing between pauses of about half a second or more; the unit an Auto Octave shift is chosen for.
_Avoid_: line, segment

**Guide lines**:
Faint copies of the reference melody one octave above and below the active Octave shift.

**Band**:
How far from the Reference a moment is: green within ±25 cents, yellow 25–50, red beyond ±50.
_Avoid_: zone, tolerance

**In tune**:
In the green Band. "% in tune" is the share of sung time in green.

**Steadiness**:
The spread of pitch (SD in cents) across the stable middle of a held note, with attack and release left out.
_Avoid_: stability, precision
