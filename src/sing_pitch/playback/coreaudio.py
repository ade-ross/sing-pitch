"""CoreAudio device transport type + data source via ctypes (no extra deps). macOS only."""
import ctypes, ctypes.util, struct
ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

def fourcc(s): return struct.unpack(">I", s.encode())[0]
def unfourcc(n): return struct.pack(">I", n).decode("latin-1")

class AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32), ("mScope", ctypes.c_uint32), ("mElement", ctypes.c_uint32)]

ca.AudioObjectGetPropertyDataSize.argtypes = [ctypes.c_uint32, ctypes.POINTER(AudioObjectPropertyAddress), ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
ca.AudioObjectGetPropertyData.argtypes = [ctypes.c_uint32, ctypes.POINTER(AudioObjectPropertyAddress), ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
cf.CFRelease.argtypes = [ctypes.c_void_p]

SYSTEM = 1
GLOBAL = fourcc("glob"); OUTPUT = fourcc("outp"); INPUT = fourcc("inp ")

def _addr(sel, scope=GLOBAL, el=0): return AudioObjectPropertyAddress(fourcc(sel), scope, el)

def _get(obj, addr, ctype):
    size = ctypes.c_uint32(ctypes.sizeof(ctype))
    val = ctype()
    err = ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(val))
    return None if err else val

def device_ids():
    a = _addr("dev#")
    size = ctypes.c_uint32()
    ca.AudioObjectGetPropertyDataSize(SYSTEM, ctypes.byref(a), 0, None, ctypes.byref(size))
    n = size.value // 4
    arr = (ctypes.c_uint32 * n)()
    ca.AudioObjectGetPropertyData(SYSTEM, ctypes.byref(a), 0, None, ctypes.byref(size), arr)
    return list(arr)

def _cfstr(obj, sel):
    v = _get(obj, _addr(sel), ctypes.c_void_p)
    if v is None or not v.value: return None
    buf = ctypes.create_string_buffer(256)
    cf.CFStringGetCString(v.value, buf, 256, 0x08000100)  # kCFStringEncodingUTF8
    cf.CFRelease(v.value)
    return buf.value.decode()

def info():
    out = {}
    for d in device_ids():
        name = _cfstr(d, "lnam")
        t = _get(d, _addr("tran"), ctypes.c_uint32)
        ds = _get(d, _addr("ssrc", OUTPUT), ctypes.c_uint32)
        out[name] = dict(transport=unfourcc(t.value) if t else None, out_source=unfourcc(ds.value) if ds else None)
    return out
