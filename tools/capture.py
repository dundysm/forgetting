"""Render forgetting's video, frame by frame: 4.5 GiB and 10 GiB on the same run clock.

Usage: python tools/capture.py <out.mp4> [seconds] [fps] [start_ms] [focus] [layout: stack|side]
(serve web/ on port 8780 first; needs playwright, Chrome and imageio-ffmpeg)
  stack: 1080x1350 (4:5, for the X feed), 4.5 GiB on top of 10 GiB
  side:  1920x1080, side by side
The camera move is periodic and the last XF seconds crossfade into the first, so it loops cleanly.
"""
import asyncio, math, os, shutil, subprocess, sys
import imageio_ffmpeg
from playwright.async_api import async_playwright

OUT = sys.argv[1]
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 15
FPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
START = float(sys.argv[4]) if len(sys.argv) > 4 else 14000
FOCUS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
LAYOUT = sys.argv[6] if len(sys.argv) > 6 else "stack"
XF = 0.6  # seconds of loop crossfade
W, H = (1080, 675) if LAYOUT == "stack" else (960, 1080)
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brainframes")


async def main():
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", args=["--enable-gpu", "--use-angle=d3d11", "--ignore-gpu-blocklist"])
        pages = []
        for gib in ("4.5", "10"):
            pg = await b.new_page(viewport={"width": W, "height": H})
            pg.on("console", lambda m, g=gib: m.type in ("error", "warning") and print(g, m.type, m.text[:300], flush=True))
            pg.on("pageerror", lambda e, g=gib: print(g, "pageerror", str(e)[:300], flush=True))
            for attempt in range(4):
                await pg.goto(f"http://localhost:8780/?gib={gib}&capture")
                try:
                    await pg.wait_for_function("window.__f && __f.renderAt", timeout=60000)
                    break
                except Exception:
                    print(gib, "page not ready, retry", attempt + 1, flush=True)
            else:
                raise SystemExit(f"{gib} GiB page never became ready")
            if LAYOUT == "stack":  # labels readable on a phone, tucked bottom-left of each panel
                await pg.add_style_tag(content="#tag{font-size:38px!important;text-align:left!important;left:6%!important;top:7%!important;bottom:auto!important}#tag span{font-size:22px!important}")
            pages.append(pg)
        n = int((SECS + XF) * FPS)
        for pg in pages:  # warm-up: replay up to the start so dust already in the air is in place
            for ms in range(0, int(START), 100):
                await pg.evaluate(f"__f.renderAt({ms}, 0, 1, {FOCUS})")
        for pg in pages:
            await pg.evaluate(f"__f.renderAt({START}, 0, 1, {FOCUS}); __f.setBaseline()")
        for i in range(n):
            ms = START + i * 1000 / FPS
            s = math.sin(math.pi * (i % int(SECS * FPS)) / (SECS * FPS)) ** 2   # 0 → 1 → 0: periodic
            orbit, dist, height = -0.22 + 0.44 * s, 1.08 - 0.12 * s, 1.35 - 0.3 * s
            if LAYOUT == "stack":
                dist *= 0.86
            for k, pg in enumerate(pages):
                stats = await pg.evaluate(f"__f.renderAt({ms}, {orbit}, {dist}, {FOCUS}, {height})")
                # the counter resets at the loop seam: hide it in the tail and fade it in over the head,
                # so the crossfade never shows two numbers (and frame 0, the thumbnail, is clean)
                op = 0 if i >= SECS * FPS else min(1, i / (XF * FPS))
                await pg.evaluate(f"document.querySelector('#tag span').style.opacity = {op}")
                await pg.screenshot(path=f"{TMP}/{k}_{i:05d}.png")
            if i % 90 == 0:
                print(i, ms, stats, flush=True)
        await b.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    stack = "vstack" if LAYOUT == "stack" else "hstack"
    div = "drawbox=x=0:y=ih/2-1:w=iw:h=2:color=white@0.06:t=fill" if LAYOUT == "stack" else "drawbox=x=iw/2-1:y=0:w=2:h=ih:color=white@0.06:t=fill"
    body = f"{TMP}/body.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{TMP}/0_%05d.png",
                    "-framerate", str(FPS), "-i", f"{TMP}/1_%05d.png",
                    "-filter_complex", f"[0:v][1:v]{stack}=inputs=2,format=yuv420p",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "14", body], check=True)
    # seamless loop: blend the last XF seconds over the first XF seconds, then drop the tail
    fc = (f"[0:v]split=3[a][b][c];"
          f"[a]trim=start={SECS}:end={SECS + XF},setpts=PTS-STARTPTS[tail];"
          f"[b]trim=0:{XF},setpts=PTS-STARTPTS[head];"
          f"[c]trim=start={XF}:end={SECS},setpts=PTS-STARTPTS[rest];"
          f"[tail][head]blend=all_expr='A*(1-T/{XF})+B*(T/{XF})'[xhead];"
          f"[xhead][rest]concat=n=2:v=1,format=yuv420p[v]")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", body, "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-movflags", "+faststart", OUT], check=True)
    print("wrote", OUT)

asyncio.run(main())
