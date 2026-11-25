"""The sing-pitch logo: melody bars, the voice as dots landing on them, the now-line. Writes the SVG given as argument.
Run: python3 docs/logo/make_logo.py docs/logo/logo.svg   (banner.html renders social-preview.png in a browser at 1280x640)"""
import math, sys
S = 512
BARS = [(58, 168, 372), (192, 282, 272), (318, 392, 168), (414, 458, 236)]       # (x0, x1, y) of each melody note
NOW_X = 300
BLUE, GREEN, YELLOW, RED = "#5dade2", "#2ecc71", "#f1c40f", "#e74c3c"

def voice_y(x):
    """The singer: slides up into each note, overshoots a little, settles with a small vibrato."""
    for i, (x0, x1, y) in enumerate(BARS):
        if x <= x1 + 9 or i == len(BARS) - 1:
            prev = BARS[i - 1][2] if i else y + 46
            k = (x - x0 + 20) / 34                                   # 0..1 across the slide into the note
            if k < 1:
                ease = 0.5 - 0.5 * math.cos(math.pi * max(k, 0))
                return prev + (y - prev) * ease * 1.12               # overshoot
            settle = math.exp(-(x - x0 - 10) / 22)
            return y - (y - prev) * 0.10 * settle + 5.0 * math.sin((x - x0) / 11.0)
    return BARS[-1][2]

def colour(x, y):
    bar = next((b for b in BARS if b[0] - 24 <= x <= b[1] + 2), None)   # a dot sliding in is judged on its note
    if bar is None:
        return RED
    off = abs(y - bar[2])
    return GREEN if off < 12 else YELLOW if off < 35 else RED

out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="{S}" height="{S}">',
       '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
       '<stop offset="0" stop-color="#22305a"/><stop offset="1" stop-color="#0c1020"/></linearGradient>'
       '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="9"/></filter>'
       '<clipPath id="r"><rect width="512" height="512" rx="114"/></clipPath></defs>',
       '<g clip-path="url(#r)">',
       f'<rect width="{S}" height="{S}" fill="url(#bg)"/>']
for y in range(110, 430, 38):                                        # the plot's semitone grid
    out.append(f'<line x1="0" x2="{S}" y1="{y}" y2="{y}" stroke="#ffffff" stroke-opacity="0.05" stroke-width="2"/>')
out.append(f'<rect x="{NOW_X}" y="0" width="{S - NOW_X}" height="{S}" fill="#000" fill-opacity="0.22"/>')   # what is coming
for x0, x1, y in BARS:                                               # Tolerance band, then the note bar
    out.append(f'<rect x="{x0 - 6}" y="{y - 25}" width="{x1 - x0 + 12}" height="50" rx="25" fill="{BLUE}" fill-opacity="0.20"/>')
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" stroke="{BLUE}" stroke-width="30" stroke-linecap="round"/>')
out.append(f'<line x1="{NOW_X}" x2="{NOW_X}" y1="62" y2="450" stroke="#f5f5f5" stroke-opacity="0.9" stroke-width="9" stroke-linecap="round"/>')
dots, x = [], 50.0
while x <= NOW_X - 14:                                               # the voice, sung so far
    y = voice_y(x); dots.append((x, y, colour(x, y))); x += 28.0
out.append('<g filter="url(#glow)" opacity="0.55">' + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="{c}"/>' for x, y, c in dots) + '</g>')
out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12.5" fill="{c}"/>' for x, y, c in dots]
out.append('</g><rect x="1.5" y="1.5" width="509" height="509" rx="113" fill="none" stroke="#ffffff" stroke-opacity="0.14" stroke-width="3"/></svg>')
open(sys.argv[1], "w").write("\n".join(out))
