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

**Drone**:
A held key note played in headphones during a Key Take.

**Unscored stretch**:
A part of a Melody where the extracted reference pitch is unreliable; shown greyed out and left out of every score.
_Avoid_: gap, ignored part

**Octave shift**:
The single whole-octave offset between Simone's voice and the original singer, detected once per Take.

**Band**:
How far from the Reference a moment is: green within ±25 cents, yellow 25–50, red beyond ±50.
_Avoid_: zone, tolerance

**In tune**:
In the green Band. "% in tune" is the share of sung time in green.

**Steadiness**:
The spread of pitch (SD in cents) across the stable middle of a held note, with attack and release left out.
_Avoid_: stability, precision
