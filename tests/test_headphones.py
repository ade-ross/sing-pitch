"""Is the output headphones? The pure rule over CoreAudio facts (fake dicts), no hardware."""
import pytest

from sing_pitch.playback.outputs import classify_output


@pytest.mark.parametrize("name,facts,kind", [
    ("MacBook Pro Speakers", {"transport": "bltn", "out_source": "ispk"}, "speakers"),
    ("MacBook Pro Speakers", {"transport": "bltn", "out_source": "hdpn"}, "headphones"),   # jack plugged in
    ("External Headphones", {"transport": "bltn", "out_source": "hdpn"}, "headphones"),
    ("WF-1000XM5", {"transport": "blue", "out_source": None}, "headphones"),
    ("Simone's AirPods Pro", {"transport": "blea", "out_source": None}, "headphones"),
    ("JBL Flip 6", {"transport": "blue", "out_source": None}, "headphones"),             # BT: assumed headphones...
    ("Kitchen Speaker", {"transport": "blue", "out_source": None}, "speakers"),          # ...unless named a speaker
    ("LG HDR 4K", {"transport": "dprt", "out_source": None}, "speakers"),
    ("DELL U2720Q", {"transport": "hdmi", "out_source": None}, "speakers"),
    ("Scarlett 2i2 USB", {"transport": "usb ", "out_source": None}, "unknown"),
    ("BlackHole 2ch", {"transport": "virt", "out_source": None}, "unknown"),
    ("Multi-Output Device", {"transport": "grup", "out_source": None}, "unknown"),
])
def test_coreaudio_facts_decide(name, facts, kind):
    assert classify_output(name, facts) == kind


@pytest.mark.parametrize("name,kind", [
    ("AirPods Pro", "headphones"), ("WH-1000XM4", "headphones"), ("Beats Studio", "headphones"),
    ("USB Headset", "headphones"), ("MacBook Pro Speakers", "speakers"), ("LG HDR 4K", "unknown"),
])
def test_without_coreaudio_facts_the_name_decides(name, kind):
    assert classify_output(name, None) == kind


def test_a_usb_device_named_headphones_counts_as_headphones():
    assert classify_output("Schiit Headphone Amp", {"transport": "usb ", "out_source": None}) == "headphones"
