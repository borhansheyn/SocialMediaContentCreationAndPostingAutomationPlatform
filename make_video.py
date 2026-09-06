import math, random, subprocess
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import imageio_ffmpeg

SRC = "assets/ravankav_subject.jpg"
AUD = "voiceover_ravankav.mp3"
OUT = "ravankav_intro.mp4"
FPS = 30
DUR = 12.6
N = int(FPS * DUR)
W, H = 1080, 1920

base = Image.open(SRC).convert("RGB")
# upscale working canvas for headroom
WORK_W, WORK_H = int(W * 1.5), int(H * 1.5)
base = base.resize((WORK_W, WORK_H), Image.LANCZOS)

random.seed(7)
# handheld noise: sum of sines with random phases
def hh(t, amp, seed):
    r = random.Random(seed)
    v = 0.0
    for k in range(1, 5):
        v += math.sin(2 * math.pi * (0.23 * k * 1.7) * t + r.random() * 6.28) / k
    return v * amp

def perspective_coeffs(src, dst):
    # returns coeffs mapping dst->src for PIL
    A = []
    B = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        B.append(xs)
        A.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
        B.append(ys)
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    res = np.linalg.solve(A, B)
    return res.tolist()

ff = imageio_ffmpeg.get_ffmpeg_exe()
proc = subprocess.Popen(
    [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
     "-i", "pipe:0", "-i", AUD,
     "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

TOTAL_YAW = math.radians(45.0)

for i in range(N):
    t = i / FPS
    p = i / max(N - 1, 1)
    # ease in/out so the orbit starts and stops softly
    e = 0.5 - 0.5 * math.cos(math.pi * p)
    yaw = (-TOTAL_YAW / 2) + TOTAL_YAW * e  # left -> right

    cx, cy = WORK_W / 2.0, WORK_H / 2.0
    f = WORK_W * 1.9  # virtual focal length -> perspective strength
    hw, hh_ = WORK_W / 2.0, WORK_H / 2.0

    # project the 4 corners of a plane rotated about the vertical axis
    corners = [(-hw, -hh_), (hw, -hh_), (hw, hh_), (-hw, hh_)]
    dst = []
    for (x, y) in corners:
        z = x * math.sin(yaw)
        xr = x * math.cos(yaw)
        s = f / (f + z)
        dst.append((cx + xr * s, cy + y * s))
    src = [(0, 0), (WORK_W, 0), (WORK_W, WORK_H), (0, WORK_H)]

    coeffs = perspective_coeffs(src, dst)
    frame = base.transform((WORK_W, WORK_H), Image.PERSPECTIVE, coeffs,
                           Image.BICUBIC, fillcolor=(12, 10, 9))

    # handheld: translation + tiny roll
    dx = hh(t, 9.0, 1) + hh(t * 2.7, 3.0, 11)
    dy = hh(t, 7.0, 2) + hh(t * 3.1, 2.5, 12)
    roll = hh(t, 0.35, 3)
    frame = frame.rotate(roll, resample=Image.BICUBIC, center=(cx, cy))

    # slight breathing zoom (handheld drift / AF breathing)
    zoom = 1.16 + 0.012 * math.sin(2 * math.pi * 0.11 * t)
    cw, ch = W / zoom * 1.5 / 1.5, H / zoom * 1.5 / 1.5
    cw, ch = W / zoom, H / zoom
    left = cx - cw / 2 + dx
    top = cy - ch / 2 + dy
    left = max(0, min(WORK_W - cw, left))
    top = max(0, min(WORK_H - ch, top))
    frame = frame.crop((int(left), int(top), int(left + cw), int(top + ch))).resize((W, H), Image.LANCZOS)

    # --- phone-camera look (Redmi Note 10 Pro) ---
    # subtle autofocus breathing softness at the start
    if t < 0.55:
        frame = frame.filter(ImageFilter.GaussianBlur(0.9 * (0.55 - t) / 0.55))
    # aggressive-ish software sharpening
    frame = frame.filter(ImageFilter.UnsharpMask(radius=1.4, percent=115, threshold=3))
    frame = ImageEnhance.Color(frame).enhance(1.10)
    frame = ImageEnhance.Contrast(frame).enhance(1.06)

    arr = np.asarray(frame).astype(np.int16)
    # luma-dependent sensor noise (more in shadows)
    lum = arr.mean(axis=2, keepdims=True) / 255.0
    strength = 4.5 + 5.0 * (1.0 - lum)
    noise = np.random.normal(0, 1, arr.shape) * strength
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    proc.stdin.write(arr.tobytes())

proc.stdin.close()
proc.wait()
print("done", OUT)
