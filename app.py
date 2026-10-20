"""
🎭 Imposter — Gradio Edition
========================================
Text-to-Video + Image-to-Video web app powered by PUBLIC Hugging Face
Gradio Spaces via `gradio_client`.

* Hosting ......... 100% FREE on Streamlit Community Cloud
* Video generation . 100% FREE — calls public ZeroGPU demo Spaces directly.
                      NO Hugging Face token, NO API keys, NO credits needed.
                      If one Space is busy/down, the app automatically tries
                      the next one in the fallback chain.

No GPU, no torch, no heavy downloads needed in the app: it is just a
lightweight front-end, the video models run on the public Spaces.

CEO: MUHAMMAD MUDASSIR YASEEN
"""

from __future__ import annotations

import random
import tempfile
import time
import traceback
import urllib.request

import streamlit as st
import streamlit.components.v1 as components
from gradio_client import Client
from gradio_client.utils import Status


# ----------------------------------------------------------------------------
# OWNER SETTINGS — updated by Muse on the owner's request
# ----------------------------------------------------------------------------
# Paste your ad network (Adsterra/PropellerAds etc.) <script> or <iframe> code
# here. While empty, a placeholder appears in the ad slot.
AD_HTML = """
<!-- Adsterra Social Bar (high CPM) -->
<script data-cfasync="false" src="https://bauval.org/14/06cfef510844ff8afa72574c902ab415"></script>
<!-- Adsterra Banner 300x250 -->
<script>
  atOptions = {
    'key' : '644d29a9d10373fa5f3a39e674d333cb',
    'format' : 'iframe',
    'height' : 250,
    'width' : 300,
    'params' : {}
  };
</script>
<script src="https://bauval.org/22/644d29a9d10373fa5f3a39e674d333cb"></script>
<!-- Adsterra Native Banner -->
<script async="async" data-cfasync="false" src="https://bauval.org/21/36bf4270f19cadf49600f4e0f50abe1c"></script>
<div id="container-36bf4270f19cadf49600f4e0f50abe1c"></div>
"""

# Seconds the visitor must wait on the ad before the studio unlocks:
AD_WAIT_SECONDS = 10

# Survey links — send new links any time and Muse will add them here:
# {"title": "...", "url": "https://...", "desc": "..."}
SURVEY_LINKS = [
    {"title": "Get $500 to Spend at Walmart!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=74764", "desc": "Sweepstakes survey — win $500 for Walmart (iOS, USA). Payout $2.48/lead."},
    {"title": "Get $1000 on Your CashApp!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=75222", "desc": "Sweepstakes survey — win $1000 on CashApp (iOS, USA). Payout $2.99/lead."},
    {"title": "Get a $100 Nike Gift Card!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=75217", "desc": "$100 Nike gift card signup survey (Desktop, USA). Payout $2.99/lead."},
    {"title": "Get $100 to Spend at Ulta!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=74436", "desc": "$100 Ulta signup survey (Desktop, USA). Payout $3.08/lead."},
    {"title": "Get $1000 CashApp Gift Card!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=75220", "desc": "$1000 CashApp gift card survey (Desktop, USA). Payout $2.99/lead."},
    {"title": "Get $1000 to Spend at Walmart!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=75095", "desc": "Sweepstakes — win $1000 for Walmart (Android, USA). Payout $0.89/lead."},
    {"title": "Get $100 at Jersey Mike's!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=75134", "desc": "$100 Jersey Mike's signup survey (Desktop, USA). Payout $2.73/lead."},
    {"title": "Get $100 Ulta Gift Card Now!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=74437", "desc": "$100 Ulta gift card sweepstakes (Android, USA). Payout $3.08/lead."},
    {"title": "Get the Best Samples Now!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=75092", "desc": "Free product samples signup (Android, USA). Payout $1.02/lead."},
    {"title": "Register for ShopBack Now!", "url": "https://singingfiles.com/show.php?l=0&u=2558676&id=75420", "desc": "Cashback rewards signup (Desktop, USA). Payout $0.28/lead."},
]

# Pixabay free API key (stock videos — no quota). Get a free key at pixabay.com/api/docs
PIXABAY_API_KEY = __import__("os").environ.get("PIXABAY_API_KEY", "")




# ----------------------------------------------------------------------------
# Public Gradio Spaces (verified live via gradio_client)
# ----------------------------------------------------------------------------
# One adapter per Space submits jobs with the right endpoint + parameters.
# No API key / token required.

SPACE_LTX = "Lightricks/ltx-video-distilled"      # official LTX-Video demo
SPACE_COG5 = "zai-org/CogVideoX-5B-Space"         # official CogVideoX 5B demo
SPACE_COG2 = "zai-org/CogVideoX-2B-Space"         # official CogVideoX 2B demo
SPACE_WAN = "Upsampler/wan-2-2-14b-image-to-video"  # Wan 2.2 14B I2V (verified 2026-10-13)
SPACE_FLUX = "black-forest-labs/FLUX.1-schnell"   # photorealistic stills (verified 2026-10-20)
SPACE_SMART = "smart/cinema-engine"               # virtual: text -> FLUX still -> Wan video

JOB_TIMEOUT = 480  # seconds (8 min) — public queues can be slow at peak times


def _ltx_submit(client: Client, kind: str, prompt: str, negative: str,
                image_path, width: int, height: int, duration: float,
                seed: int, guidance: float, drop_params=frozenset()):
    """Submit to LTX-Video Space. kind: 't2v' or 'i2v'."""
    rnd = (seed == -1)
    # Free-tier balance (v3.1): honor the user's pick up to 5 seconds and
    # 640px on the long edge. Bigger requests cost more free GPU quota, so
    # when the free pool is empty the engine declines and the app explains
    # it — but a valid request is never silently shrunk to 2s/512px again.
    scale = min(1.0, 640.0 / max(width, height))
    w = int(width * scale) // 8 * 8
    h = int(height * scale) // 8 * 8
    dur = min(float(duration), 5.0)
    frames = max(9, min(257, int(dur * 24)))
    api = "/text_to_video" if kind == "t2v" else "/image_to_video"
    mode = "text-to-video" if kind == "t2v" else "image-to-video"
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        input_image_filepath=image_path,      # None for text-to-video
        input_video_filepath=None,
        height_ui=h,
        width_ui=w,
        mode=mode,
        duration_ui=dur,
        ui_frames_to_use=frames,
        seed_ui=0 if rnd else seed,
        randomize_seed=rnd,
        ui_guidance_scale=guidance,
        improve_texture_flag=False,
        api_name=api,
    )
    for p in drop_params:
        kwargs.pop(p, None)
    return client.submit(**kwargs)


def _cog_submit(client: Client, kind: str, prompt: str, negative: str,
                image_path, width: int, height: int, duration: float,
                seed: int, guidance: float, drop_params=frozenset()):
    """Submit to CogVideoX Space (/generate handles both t2v and i2v)."""
    kwargs = dict(
        prompt=prompt,
        api_name="/generate",
    )
    # I2V only: 5B accepts an image; 2B has NO image param (T2V only).
    if image_path:
        kwargs["image_input"] = image_path
    # Drop any params the engine previously rejected (self-healing backstop).
    for p in drop_params:
        kwargs.pop(p, None)
    return client.submit(**kwargs)


def _wan_submit(client: Client, kind: str, prompt: str, negative: str,
                image_path, width: int, height: int, duration: float,
                seed: int, guidance: float, drop_params=frozenset()):
    """Submit to Wan 2.2 14B I2V Space. I2V ONLY (no T2V endpoint)."""
    from gradio_client.utils import handle_file
    if kind != "i2v" or not image_path:
        raise ValueError("Wan Space only supports image-to-video")
    # The engine renders near the input image's size: feed it at least
    # 720px on the long edge so the video comes out as large as possible.
    try:
        from PIL import Image as _PILImage
        _im = _PILImage.open(image_path)
        _w, _h = _im.size
        if max(_w, _h) < 720:
            _s = 720.0 / max(_w, _h)
            _im = _im.convert("RGB").resize(
                (max(2, int(_w * _s) // 2 * 2), max(2, int(_h * _s) // 2 * 2)),
                _PILImage.LANCZOS)
            _up = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            _im.save(_up.name)
            image_path = _up.name
    except Exception:
        pass
    # steps max 12, duration as requested
    return client.submit(
        input_image=handle_file(image_path),
        prompt=prompt,
        steps=10,
        negative_prompt=negative or "blurry, low quality",
        duration_seconds=max(1, min(5, int(round(duration)))),
        guidance_scale=float(guidance),
        api_name="/generate_video",
    )


# ----------------------------------------------------------------------------
# Smart Cinema route: the studio UNDERSTANDS plain words.
# A visitor types anything in their own words; the studio first paints a
# photorealistic cinematic still of exactly that scene, then films it with
# the Wan engine — so text prompts come out as real CGI, not flat cartoons.
# ----------------------------------------------------------------------------

_CGI_STYLE = ("photorealistic CGI, cinematic Hollywood VFX quality, "
              "ultra-detailed realistic textures, dramatic cinematic lighting")


def _enhance_prompt(raw: str):
    """Wrap the visitor's own words in cinematic direction.
    Returns (image_prompt, video_prompt). Their words always lead."""
    raw = " ".join(str(raw).split())
    image_prompt = f"{_CGI_STYLE}: {raw}"
    video_prompt = (f"{_CGI_STYLE}, smooth natural motion, the scene comes "
                    f"alive with realistic movement: {raw}")
    return image_prompt, video_prompt


def _smart_submit(client: Client, kind: str, prompt: str, negative: str,
                  image_path, width: int, height: int, duration: float,
                  seed: int, guidance: float, drop_params=frozenset()):
    """Text-to-video the smart way: FLUX still -> Wan I2V job.
    `client` here is the FLUX image client (see get_client mapping)."""
    if kind != "t2v":
        raise ValueError("Cinema Engine is for text-to-video")
    img_prompt, vid_prompt = _enhance_prompt(prompt)
    res = client.predict(
        prompt=img_prompt,
        seed=0,
        randomize_seed=True,
        width=int(width),
        height=int(height),
        num_inference_steps=4,
        api_name="/infer",
    )
    img = res[0] if isinstance(res, (tuple, list)) else res
    if isinstance(img, dict):
        img = img.get("path") or img.get("image") or img.get("url")
    if not img:
        raise RuntimeError("Cinema Engine could not paint the first frame")
    if isinstance(img, str) and img.startswith("http"):
        dst = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        with urllib.request.urlopen(img, timeout=120) as resp, \
                open(dst, "wb") as f:
            f.write(resp.read())
        img = dst
    wan_client = get_client(SPACE_WAN)
    return _wan_submit(wan_client, "i2v", vid_prompt, negative, img,
                       width, height, duration, seed, guidance)


# Space registry: label + submit adapter. Order = fallback priority.
SPACE_ADAPTERS = {
    SPACE_SMART: {"label": "🎨 Cinema Engine", "fn": _smart_submit},
    SPACE_LTX:  {"label": "⚡ Turbo Engine", "fn": _ltx_submit},
    SPACE_COG5: {"label": "🎬 Pro Engine",   "fn": _cog_submit},
    SPACE_COG2: {"label": "🚀 Fast Engine",  "fn": _cog_submit},
    SPACE_WAN:  {"label": "🎨 HD Engine",    "fn": _wan_submit},
}
T2V_CHAIN = [SPACE_SMART, SPACE_LTX]  # smart CGI first; LTX is the backup
# I2V reality (verified 2026-10-13): LTX /image_to_video raises a server-side
# exception; CogVideoX-2B /generate has NO image param at all (T2V only).
# Working I2V: Wan 2.2 14B (verified live) + CogVideoX-5B (quota-unstable).
I2V_CHAIN = [SPACE_WAN, SPACE_COG5]
# Story mode chains clips via image-to-video. Wan 2.2 14B is primary
# (verified working); CogVideoX-5B is fallback.
STORY_CHAIN = [SPACE_WAN, SPACE_COG5]

_client_cache: dict = {}

# A hanging Space handshake must NEVER block the Streamlit session:
# Streamlit Cloud drops the websocket after ~60-90s of silence and the
# run dies with no error and no fallback. Fail fast instead.
CONNECT_TIMEOUT = 20  # seconds for connect; generous read for slow queues


def _client_timeout():
    try:
        import httpx
        return httpx.Timeout(connect=CONNECT_TIMEOUT, read=300.0,
                             write=60.0, pool=CONNECT_TIMEOUT)
    except Exception:
        return CONNECT_TIMEOUT


def _hf_token():
    """Optional Hugging Face token (Streamlit Secrets or env).

    With a token, the free GPU quota is the account's own dedicated pool
    instead of the tiny shared anonymous pool — that is what makes the
    free tier (3 videos/day) reliable. Without one, the app still works
    anonymously, just with a smaller shared quota.
    """
    import os
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        import streamlit as _st
        t = _st.secrets.get("HF_TOKEN")
        return t if t else None
    except Exception:
        return None


def get_client(space_id: str) -> Client:
    """Cache one gradio_client per Space (connect handshake is slow)."""
    if space_id == SPACE_SMART:
        space_id = SPACE_FLUX  # the Cinema Engine's first step is the painter
    if space_id not in _client_cache:
        tok = _hf_token()
        client = None
        if tok:
            try:
                client = Client(space_id, token=tok,
                                httpx_kwargs={"timeout": _client_timeout()})
            except TypeError:
                client = None
        if client is None:
            client = Client(
                space_id, httpx_kwargs={"timeout": _client_timeout()})
        _client_cache[space_id] = client
    return _client_cache[space_id]


def _is_connect_error(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(k in low for k in (
        "connecterror", "connect error", "connection refused",
        "name or service not known", "temporary failure in name resolution",
        "invalidurl", "invalid url", "ssl", "certificate",
        "timed out", "timeout", "read timed out"))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _result_to_bytes(result) -> bytes:
    """Turn a gradio_client job result into MP4 bytes. Never crashes silently."""
    def _path_of(r):
        if isinstance(r, str):
            return r if r.lower().endswith(".mp4") else None
        if isinstance(r, dict):
            p = r.get("path") or r.get("video") or r.get("url")
            if isinstance(p, str) and p.lower().endswith(".mp4"):
                return p
            for v in r.values():
                found = _path_of(v)
                if found:
                    return found
            return None
        if isinstance(r, (list, tuple)):
            for item in r:
                found = _path_of(item)
                if found:
                    return found
        return None

    path = _path_of(result)
    if not path:
        raise RuntimeError(f"Space returned an unexpected result: {str(result)[:200]}")
    if path.startswith("http"):
        with urllib.request.urlopen(path, timeout=180) as resp:
            return resp.read()
    with open(path, "rb") as f:
        return f.read()


def _clean_brand(text: str) -> str:
    """Strip third-party branding from user-facing error text."""
    import re as _re
    text = _re.sub(r"(?i)the gradio app", "the engine", text)
    text = _re.sub(r"(?i)gradio", "the engine", text)
    text = _re.sub(r"(?i)hugging face", "the engine", text)
    text = _re.sub(r"(?i)streamlit", "the app", text)
    text = _re.sub(r"Lightricks/[\\w-]+", "the engine", text)
    text = _re.sub(r"zai-org/[\\w-]+", "the engine", text)
    text = _re.sub(r"Upsampler/[\\w-]+", "the engine", text)
    text = _re.sub(r"black-forest-labs/[\\w-]+", "the engine", text)
    text = _re.sub(r"smart/[\\w-]+", "the engine", text)
    return text


def _friendly_space_error(exc: Exception) -> str:
    if isinstance(exc, DailyLimitError):
        return str(exc)
    low = str(exc).lower()
    if "zerogpu" in low or "exceeded your" in low or "gpu quota" in low:
        return ("🎁 **The free engines are resting.** The shared free GPU "
                "quota refills on its own — please try again in a little "
                "while. Your free tier gives you 3 videos every day.")
    if "queue_full" in low or "queue is full" in low:
        return ("⏳ **All engines are busy.** Please wait a few minutes and try again.")
    if "timeout" in low or "timed out" in low:
        return ("⏳ **Request timed out.** Please try again — the app will automatically try the next engine.")
    if "connection" in low or "unreachable" in low or "failed to connect" in low:
        return ("🔌 **Connection issue.** Please try again in a moment.")
    short = _clean_brand(str(exc).strip()[:350])
    return f"⚠️ **Generation failed:** {short}"


# ----------------------------------------------------------------------------
# Free tier: 3 videos per day (per visitor session)
# ----------------------------------------------------------------------------

FREE_DAILY_LIMIT = 3


class DailyLimitError(Exception):
    """Raised when a visitor has used up today's free videos."""


def _check_daily_limit() -> None:
    import datetime
    today = datetime.date.today().isoformat()
    if st.session_state.get("gen_day") != today:
        st.session_state["gen_day"] = today
        st.session_state["gen_count"] = 0
    used = int(st.session_state.get("gen_count", 0))
    if used >= FREE_DAILY_LIMIT:
        raise DailyLimitError(
            f"🎁 **You've used your {FREE_DAILY_LIMIT} free videos for "
            "today.** Your free tier refills tomorrow — come back for 3 "
            "more free videos, on us!")


def _bump_daily_count() -> None:
    import datetime
    today = datetime.date.today().isoformat()
    if st.session_state.get("gen_day") != today:
        st.session_state["gen_day"] = today
        st.session_state["gen_count"] = 0
    st.session_state["gen_count"] = int(st.session_state.get("gen_count", 0)) + 1


def generate_with_fallback(kind: str, status, preferred: str | None = None,
                           pin_space: str | None = None,
                           from_story: bool = False, **params):
    """Generate via public Spaces with automatic fallback.

    Tries the preferred Space first, then the rest of the chain.
    If pin_space is set, ONLY that Space is used (for story-mode consistency).
    Each Space gets one retry on transient errors. Live queue progress
    is shown in the `status` (st.status) container.
    Returns (video_bytes, space_label, space_id). Raises last error if all fail.
    """
    if not from_story:
        _check_daily_limit()
    if pin_space:
        chain = [pin_space]
    else:
        chain = T2V_CHAIN if kind == "t2v" else I2V_CHAIN
        if preferred in chain:
            chain = [preferred] + [s for s in chain if s != preferred]

    last_exc: Exception | None = None
    for space_id in chain:
        label = SPACE_ADAPTERS[space_id]["label"]
        submit_fn = SPACE_ADAPTERS[space_id]["fn"]
        drop_params = set()  # params this engine rejected -> skip on retry
        for attempt in (1, 2, 3, 4, 5, 6):
            try:
                status.update(label=f"🎥 Connecting to {label}…", state="running")
                client = get_client(space_id)
                call_params = dict(params)
                if drop_params:
                    call_params["drop_params"] = frozenset(drop_params)
                job = submit_fn(client, kind, **call_params)
                start = time.time()
                last_qpos = None
                last_beat = 0.0
                while True:
                    # Timeout + heartbeat FIRST — these never block, so a
                    # hanging job.status() can never starve them.
                    if time.time() - start > JOB_TIMEOUT:
                        try:
                            job.cancel()
                        except Exception:
                            pass
                        raise TimeoutError(f"{label}: took longer than 8 minutes")
                    now = time.time()
                    if now - last_beat >= 25:
                        last_beat = now
                        mins = int((now - start) // 60)
                        secs = int((now - start) % 60)
                        qtxt = f" (queue #{last_qpos})" if last_qpos else ""
                        status.update(
                            label=f"🎥 {label} — still working{qtxt}… "
                                  f"{mins}m {secs}s elapsed",
                            state="running")
                    try:
                        _done = job.done()
                    except Exception:
                        _done = False
                    if _done:
                        break
                    try:
                        stt = job.status()
                        code = stt.code if hasattr(stt, "code") else None
                        qpos = getattr(stt, "queue_position", None)
                        if code == Status.IN_QUEUE and qpos and qpos != last_qpos:
                            last_qpos = qpos
                            status.update(
                                label=f"🎥 {label} — in queue #{qpos}…",
                                state="running")
                        elif code == Status.PROCESSING:
                            status.update(
                                label=f"🎥 {label} — generating your video…",
                                state="running")
                    except Exception:
                        pass
                    time.sleep(10)
                video_bytes = _result_to_bytes(job.result())
                status.update(label=f"✅ Video ready — via {label}",
                              state="complete")
                if not from_story:
                    _bump_daily_count()
                return video_bytes, label, space_id
            except Exception as exc:  # noqa: BLE001 — fallback below
                last_exc = exc
                low = str(exc).lower()
                # Connect-level failure (hanging Space handshake): retrying the
                # SAME space just burns another timeout — drop its cached
                # client and move straight to the next Space.
                if _is_connect_error(exc):
                    _client_cache.pop(space_id, None)
                    status.write(f"⚠️ {label} unreachable — trying the next engine…")
                    break  # → next Space
                # Engine rejected a parameter name (API changed). Drop the
                # cached client, remember the bad param, and retry the SAME
                # engine without it — up to 4 dropped params.
                import re as _re
                _m = _re.search(r"parameter (\w+) is not a valid key-word argument", low)
                if _m and len(drop_params) < 4:
                    _client_cache.pop(space_id, None)
                    drop_params.add(_m.group(1))
                    status.write(f"🔄 Adjusting {label}…")
                    time.sleep(2)
                    continue
                transient = any(k in low for k in (
                    "timeout", "timed out", "queue", "busy", "429", "503",
                    "connection", "temporarily"))
                if transient and attempt == 1:
                    status.write(f"⏳ {label} is busy — retrying…")
                    time.sleep(5)
                    continue
                status.write(f"⚠️ {label} failed — trying the next engine…")
                break  # → next Space

    status.update(label="❌ All engines are busy right now", state="error")
    assert last_exc is not None
    raise last_exc


def extract_last_frame(video_bytes: bytes) -> str:
    """Save the last frame of an MP4 to a PNG file. Returns the PNG path."""
    import imageio.v3 as iio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
        t.write(video_bytes)
        mp4_path = t.name
    last = None
    for frame in iio.imiter(mp4_path):
        last = frame
    if last is None:
        raise RuntimeError("Could not read frames from clip")
    png_path = mp4_path.replace(".mp4", "_lastframe.png")
    iio.imwrite(png_path, last)
    return png_path


def stitch_clips(clips: list) -> bytes:
    """Concatenate MP4 clips (same encoding) into one video. Returns bytes."""
    import imageio_ffmpeg
    import subprocess
    import os
    tmpdir = tempfile.mkdtemp()
    paths = []
    for i, b in enumerate(clips):
        p = os.path.join(tmpdir, f"clip{i:02d}.mp4")
        with open(p, "wb") as f:
            f.write(b)
        paths.append(p)
    listfile = os.path.join(tmpdir, "list.txt")
    with open(listfile, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    out = os.path.join(tmpdir, "story.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", out],
        check=True, capture_output=True, timeout=300)
    with open(out, "rb") as f:
        return f.read()


def hd_upscale_bytes(video_bytes: bytes) -> bytes:
    """HD Finish: upscale a generated clip to full HD — 1080x1920 for
    portrait, 1920x1080 for landscape — lanczos scaling + gentle sharpening.
    Returns the original bytes untouched if anything goes wrong."""
    import imageio_ffmpeg
    import subprocess
    import os
    tmpdir = tempfile.mkdtemp()
    src = os.path.join(tmpdir, "in.mp4")
    dst = os.path.join(tmpdir, "hd.mp4")
    with open(src, "wb") as f:
        f.write(video_bytes)
    # Probe the frame size from the first frame.
    import imageio.v3 as iio
    frame = next(iio.imiter(src))
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h >= w:
        tw, th = 1080, 1920
    else:
        tw, th = 1920, 1080
    if w >= tw and h >= th:
        return video_bytes  # already HD or better
    vf = (f"scale={tw}:{th}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={tw}:{th},unsharp=5:5:0.7:5:5:0.0,setsar=1")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-vf", vf,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-preset", "medium", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", dst],
        check=True, capture_output=True, timeout=300)
    with open(dst, "rb") as f:
        return f.read()


# Voiceover voices (edge-tts, free, no key). Key = label shown in UI.
VOICES = {
    "🇺🇸 Guy (US Male)": "en-US-GuyNeural",
    "🇺🇸 Jenny (US Female)": "en-US-JennyNeural",
    "🇺🇸 Aria (US Female)": "en-US-AriaNeural",
    "🇬🇧 Ryan (UK Male)": "en-GB-RyanNeural",
    "🇵🇰 Asad (Urdu Male)": "ur-PK-AsadNeural",
    "🇵🇰 Uzma (Urdu Female)": "ur-PK-UzmaNeural",
}


async def _edge_tts_save(text: str, voice_id: str, out_path: str):
    import edge_tts
    comm = edge_tts.Communicate(text, voice_id)
    await comm.save(out_path)


def generate_voiceover(text: str, voice_id: str) -> bytes:
    """Text → speech MP3 via edge-tts (free). Returns MP3 bytes."""
    import asyncio
    import os
    tmp = tempfile.mktemp(suffix=".mp3")
    asyncio.run(_edge_tts_save(text, voice_id, tmp))
    with open(tmp, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp)
    except OSError:
        pass
    if not data:
        raise RuntimeError("Voiceover came back empty — please check your text")
    return data


def get_media_duration(path: str) -> float:
    """Duration in seconds via ffprobe (imageio-ffmpeg)."""
    import imageio_ffmpeg
    import subprocess
    import json
    ffprobe = imageio_ffmpeg.get_ffmpeg_exe().replace("ffmpeg", "ffprobe")
    # imageio-ffmpeg only ships ffmpeg; use ffmpeg -i stderr parse as fallback
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-i", path], capture_output=True, text=True, timeout=30)
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", proc.stderr)
    if not m:
        return 0.0
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


def mux_voiceover(video_bytes: bytes, audio_bytes: bytes) -> bytes:
    """Mux MP3 voiceover onto MP4. Audio loops/pads or trims to video length.
    Returns final MP4 bytes."""
    import imageio_ffmpeg
    import subprocess
    import os
    tmpdir = tempfile.mkdtemp()
    vp = os.path.join(tmpdir, "v.mp4")
    ap = os.path.join(tmpdir, "a.mp3")
    out = os.path.join(tmpdir, "final.mp4")
    with open(vp, "wb") as f:
        f.write(video_bytes)
    with open(ap, "wb") as f:
        f.write(audio_bytes)
    vdur = get_media_duration(vp)
    adur = get_media_duration(ap)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if adur <= 0 or vdur <= 0:
        # durations unknown — simple mux, shortest wins
        afilter = "anull"
    elif adur < vdur:
        # loop audio to fill video (+0.5s safety)
        loops = int(vdur // adur) + 1
        afilter = f"aloop=loop={loops}:size=2e9,atrim=0:{vdur + 0.5}"
    else:
        afilter = f"atrim=0:{vdur}"
    subprocess.run(
        [ffmpeg, "-y", "-i", vp, "-i", ap,
         "-filter_complex", f"[1:a]{afilter}[a]",
         "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-shortest", out],
        check=True, capture_output=True, timeout=300)
    with open(out, "rb") as f:
        return f.read()


def generate_story(status, preferred: str | None, clip_prompts: list,
                   **base_params):
    """Multi-clip story: clip 1 via text-to-video, remaining clips via
    image-to-video chained from the previous clip's LAST FRAME — so it all
    feels like one continuous video. I2V clips pin to the first successful
    I2V Space (Wan 2.2 14B primary, CogVideoX-5B fallback).
    Returns (stitched_bytes, space_label, num_clips)."""
    _check_daily_limit()
    clips: list = []
    label = None
    total = len(clip_prompts)
    i2v_space = None  # pin to first successful I2V Space
    for i, cp in enumerate(clip_prompts):
        p = dict(base_params)
        p["prompt"] = cp
        if i == 0:
            p["image_path"] = None
            status.write(f"🎬 **Clip 1/{total}** — starting…")
            vb, label, _sid = generate_with_fallback(
                "t2v", status, preferred, from_story=True, **p)
        else:
            # Previous clip's last frame = next clip's first frame (the link!)
            p["image_path"] = extract_last_frame(clips[-1])
            status.write(f"🎬 **Clip {i+1}/{total}** — linking to previous clip…")
            vb, label, i2v_space = generate_with_fallback(
                "i2v", status, None, pin_space=i2v_space,
                from_story=True, **p)
        clips.append(vb)
    status.update(label="🔗 Stitching clips into one video…",
                  state="running")
    final = stitch_clips(clips)
    _bump_daily_count()
    status.update(label=f"✅ {total} clips — full video ready (via {label})",
                  state="complete")
    return final, label, total


def _pexels_search(query: str, api_key: str, per_page: int = 12):
    """Search Pexels videos. Returns list of dicts."""
    import json as _json
    import urllib.parse as _up
    q = _up.quote(query)
    url = (f"https://api.pexels.com/videos/search?query={q}"
           f"&per_page={per_page}&orientation=portrait")
    req = urllib.request.Request(url, headers={"Authorization": api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read().decode())
    out = []
    for v in data.get("videos", []):
        # best mp4 file: prefer HD-ish width, portrait
        best = None
        for f in v.get("video_files", []):
            if f.get("file_type") != "video/mp4":
                continue
            if best is None or (f.get("width", 0) > best.get("width", 0)
                                and f.get("width", 0) <= 1280):
                best = f
        if not best:
            continue
        out.append({
            "id": v.get("id"),
            "duration": v.get("duration"),
            "url": v.get("url"),
            "preview": v.get("image"),
            "file": best["link"],
            "width": best.get("width"),
            "height": best.get("height"),
        })
    return out


def _pixabay_search(query: str, api_key: str, per_page: int = 12):
    """Search Pixabay videos. Returns list of dicts (same shape as Pexels)."""
    import json as _json
    import urllib.parse as _up
    q = _up.quote(query)
    per_page = max(3, min(200, per_page))
    url = (f"https://pixabay.com/api/videos/?key={_up.quote(api_key)}"
           f"&q={q}&per_page={per_page}&orientation=vertical&safesearch=true")
    req = urllib.request.Request(url, headers={"User-Agent": "FreeAIVideoStudio/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read().decode())
    out = []
    for v in data.get("hits", []):
        vids = (v.get("videos") or {})
        # prefer large, fallback medium/small
        best = vids.get("large") or vids.get("medium") or vids.get("small")
        if not best or not best.get("url"):
            continue
        out.append({
            "id": v.get("id"),
            "duration": v.get("duration"),
            "url": v.get("pageURL"),
            "preview": v.get("previewURL"),
            "file": best["url"],
            "width": best.get("width"),
            "height": best.get("height"),
        })
    return out


RESOLUTIONS = {
    "Portrait 720×1280 (Shorts)": (720, 1280),   # (width, height)
    "Square 512×512": (512, 512),
    "Landscape 1280×720": (1280, 720),
}

# ----------------------------------------------------------------------------
# Page setup + professional styling
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="🎭 Imposter",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Premium theme */
    .stApp { background: #0a0a0f; }
    section[data-testid="stSidebar"] { background: #111118; border-right: 1px solid #23232e; }
    /* Fix text visibility on dark */
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label {
        color: #e4e4e7 !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stMarkdown p { color: #e4e4e7 !important; font-weight: 500; }
    .stCaption { color: #a1a1aa !important; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #e4e4e7 !important;
        font-weight: 500;
    }
    .hero {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 50%, #fbbf24 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin-bottom: 0;
        text-shadow: 0 0 40px rgba(167,139,250,0.3);
    }
    .ceo {
        color: #c4b5fd;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .ceo-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6d28d9, #db2777);
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(109,40,217,0.4);
    }
    .sub { color: #a1a1aa; margin-top: 0.3rem; margin-bottom: 1.2rem; font-size: 1.05rem; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #db2777);
        border: none;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 4px 20px rgba(124,58,237,0.4);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(124,58,237,0.6);
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 10px;
        border: 1px solid #3f3f46;
    }
    .tip {
        background: linear-gradient(135deg, rgba(109,40,217,0.15), rgba(219,39,119,0.08));
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .side-logo {
        font-size: 1.5rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .footer {
        text-align: center;
        color: #71717a;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        padding-top: 1rem;
        border-top: 1px solid #23232e;
    }
    .ceo-side {
        font-size: 0.82rem;
        color: #a1a1aa;
        text-align: center;
        line-height: 1.6;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 1px solid #52525b;
        background-color: #18181b !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    .stTextArea > div > div > textarea {
        color: #ffffff !important;
        font-weight: 600 !important;
        line-height: 1.6 !important;
    }
    .stTextArea > div > div > textarea::placeholder,
    .stTextInput > div > div > input::placeholder {
        color: #71717a !important;
        font-weight: 400 !important;
    }
    .stExpander { border-radius: 12px; border: 1px solid #27272a; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero">🎭 Imposter</p>', unsafe_allow_html=True)
st.markdown('<span class="ceo-badge">👑 CEO: MUHAMMAD MUDASSIR YASEEN</span>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Text-to-Video &amp; Image-to-Video — 100% free, no API keys. '
    "Powered by Imposter.</p>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Sidebar — CapCut/Kling-style navigation (all tools live here)
# ----------------------------------------------------------------------------

PAGE_T2V = "🎬 Text to Video"
PAGE_I2V = "🖼️ Image to Video"
PAGE_STOCK = "🎞️ Stock Videos"
PAGE_COLAB = "📓 Lip-Sync Studio"
PAGE_VOICE = "🎙️ Voiceover Studio"
PAGE_SURVEYS = "📋 Surveys"

# Safe defaults for generation settings (overridden by sidebar widgets below)
preferred_space = None
width, height = RESOLUTIONS["Portrait 720×1280 (Shorts)"]
duration = 3
guidance = 5.0
seed_param = -1

with st.sidebar:
    st.markdown('<p class="side-logo">🎭 Imposter</p>', unsafe_allow_html=True)
    st.caption("Professional AI video tools — 100% free.")
    st.markdown("**🧭 Studio Tools**")
    page = st.radio(
        "Studio tools",
        [PAGE_T2V, PAGE_I2V, PAGE_STOCK, PAGE_COLAB, PAGE_VOICE, PAGE_SURVEYS],
        label_visibility="collapsed",
    )

    st.divider()

    if page in (PAGE_T2V, PAGE_I2V):
        st.subheader("⚙️ Settings")
        st.success("🔓 **No API key needed!** Start creating right away.")

        _engine_ids = list(T2V_CHAIN)
        space_options = ["Auto (recommended)"] + [
            SPACE_ADAPTERS[s]["label"] for s in _engine_ids
        ]
        space_choice = st.selectbox(
            "Preferred Engine",
            space_options,
            help="On Auto, the app picks the best available engine; "
                 "if it is busy, it moves on to the next one.",
        )
        if not space_choice.startswith("Auto"):
            preferred_space = _engine_ids[space_options.index(space_choice) - 1]

        st.subheader("🎛️ Generation Settings")
        res_label = st.selectbox("Resolution", list(RESOLUTIONS.keys()), index=0)
        width, height = RESOLUTIONS[res_label]
        duration = st.slider(
            "Duration (seconds)",
            min_value=1, max_value=5, value=5, step=1,
            help="Free videos can be up to 5 seconds long.",
        )
        hd_finish = st.checkbox(
            "📺 HD Finish (1080p)",
            value=True,
            help="Every video is finished in full HD (1080p) before download.",
        )
        st.session_state["hd_finish"] = hd_finish
        import datetime as _dt
        if st.session_state.get("gen_day") != _dt.date.today().isoformat():
            st.session_state["gen_day"] = _dt.date.today().isoformat()
            st.session_state["gen_count"] = 0
        _left = FREE_DAILY_LIMIT - int(st.session_state.get("gen_count", 0))
        st.info(f"🎁 **Free tier:** {_left} of {FREE_DAILY_LIMIT} free "
                "videos left today (up to 5s each, HD included).")
        guidance = st.slider(
            "Guidance scale (prompt adherence)",
            min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        )
        seed_in = st.number_input(
            "Seed (-1 = random)", min_value=-1, value=-1, step=1,
        )
        seed = None if seed_in == -1 else int(seed_in)
        # -1 means random; adapters handle it
        seed_param = -1 if seed is None else seed

        with st.expander("💡 Free-Tier Reality Check"):
            st.write(
                "• **Completely free — no tokens or API keys required.**\n"
                f"• **Free tier: {FREE_DAILY_LIMIT} videos per day**, up to "
                "**5 seconds** each, finished in **full HD (1080p)**.\n"
                "• Videos are generated on our **high-performance AI servers**.\n"
                "• **Queues:** free servers get busy — at peak times it can take 10–30 min. "
                "The app automatically tries the next engine.\n"
                "• **Hosting:** 100% free cloud."
            )

    st.divider()
    st.markdown(
        '<p class="ceo-side">👤 <b>CEO: MUHAMMAD MUDASSIR YASEEN</b><br>'
        "© 2026 Imposter</p>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Ad gate — visitors watch one short ad before the studio unlocks
# ----------------------------------------------------------------------------
if "ad_watched" not in st.session_state:
    st.session_state.ad_watched = False
    st.session_state.gate_start = time.time()

if not st.session_state.ad_watched:
    st.subheader("📺 A Quick Ad Before You Start")
    st.write(
        "This tool is **completely free** — ads are what keep it that way. "
        f"Please wait **{AD_WAIT_SECONDS} seconds**, then click Continue "
        "to open the studio."
    )
    _ad_html = AD_HTML.strip() or """
        <div style="background:#1f2937;border:2px dashed #6b7280;border-radius:12px;height:250px;
                    display:flex;align-items:center;justify-content:center;color:#9ca3af;
                    font-family:sans-serif;font-size:18px;">📢 Your ad here</div>
    """
    try:
        components.html(_ad_html, height=300, scrolling=True)
    except Exception:
        st.info("📢 Your ad here.")
    if st.button("✅ Continue to the Studio", type="primary", use_container_width=True):
        _waited = time.time() - st.session_state.gate_start
        if _waited < AD_WAIT_SECONDS:
            st.warning(f"⏳ Just a moment — {int(AD_WAIT_SECONDS - _waited)} seconds remaining.")
        else:
            st.session_state.ad_watched = True
            st.rerun()
    st.stop()


def show_result(key_prefix: str, label: str) -> None:
    """Render the last generated video (persisted in session state) + download."""
    data = st.session_state.get(f"{key_prefix}_video")
    name = st.session_state.get(f"{key_prefix}_name", "video.mp4")
    if not data:
        return
    done_label = st.session_state.get(f"{key_prefix}_label_done", label)
    st.success(f"✅ {done_label} ready!")

    # Monetization: mandatory ad after every video
    if f"{key_prefix}_ad_done" not in st.session_state:
        st.session_state[f"{key_prefix}_ad_done"] = False
    if not st.session_state[f"{key_prefix}_ad_done"]:
        st.warning("📢 Please watch this ad to unlock your video download")
        import streamlit.components.v1 as _comp
        _ad = AD_HTML.strip() or """<div style="background:#1f2937;border-radius:12px;padding:20px;text-align:center;color:#9ca3af;">📢 Advertisement</div>"""
        try:
            _comp.html(_ad, height=300, scrolling=True)
        except:
            pass
        if st.button("✅ I've watched the ad — Unlock Download", key=f"{key_prefix}_ad_btn", type="primary"):
            st.session_state[f"{key_prefix}_ad_done"] = True
            # Increment video counter
            cnt = st.session_state.get("video_count", 0) + 1
            st.session_state["video_count"] = cnt
            st.rerun()
        st.stop()

    # Every 3rd video: mandatory survey
    cnt = st.session_state.get("video_count", 0)
    if cnt > 0 and cnt % 3 == 0 and not st.session_state.get(f"{key_prefix}_survey_done", False):
        st.warning(f"📋 You've created {cnt} videos! Please complete 1 quick survey to continue")
        st.link_button("📋 Complete Survey", "https://singingfiles.com/show.php?l=0&u=2558676&id=74764")
        if st.button("✅ Survey completed — Continue", key=f"{key_prefix}_survey_btn", type="primary"):
            st.session_state[f"{key_prefix}_survey_done"] = True
            st.rerun()
        st.stop()

    # HD Finish (default ON): deliver every video in full HD.
    if st.session_state.get("hd_finish", True):
        import hashlib as _hl
        _sig = _hl.md5(data).hexdigest()
        _cached = st.session_state.get(f"{key_prefix}_hd_video")
        if not (isinstance(_cached, tuple) and _cached[0] == _sig):
            try:
                _hd = hd_upscale_bytes(data)
            except Exception:
                _hd = data
            st.session_state[f"{key_prefix}_hd_video"] = (_sig, _hd)
            _cached = (_sig, _hd)
        data = _cached[1]
        if not name.endswith("_HD.mp4"):
            name = name.replace(".mp4", "_HD.mp4")

    st.video(data, format="video/mp4")
    st.download_button(
        "⬇️ Download MP4",
        data=data,
        file_name=name,
        mime="video/mp4",
        use_container_width=True,
    )


# ---------------- Page: Text to Video ----------------
if page == PAGE_T2V:
    st.subheader("Describe it → get an MP4")
    st.caption("🧠 **Smart studio:** just write what you want in your own words — "
               "the studio understands it, paints it as cinematic CGI and films it. "
               "Auto-fallback is ON: if one engine is busy, the next one takes over.")
    prompt = st.text_area(
        "Prompt",
        height=120,
        placeholder="A cheerful plumber waves hello at the camera, bright workshop, "
                    "expressive face, medium close-up",
    )
    negative = st.text_input(
        "Negative prompt (optional)",
        value="blurry, low quality, distorted faces, watermark, text",
    )

    story_mode = st.checkbox(
        "🎞️ Story mode — linked clips, one long video (e.g. 30 sec)",
        help="Each clip starts from the previous clip's last frame, "
             "so everything feels like one continuous video. "
             "Story mode uses Wan 2.2 14B (I2V) — it can be a bit slower.",
    )
    num_clips = 4
    story_lines = ""
    if story_mode:
        num_clips = st.slider("How many clips?", min_value=2, max_value=6, value=4,
                              help="Each clip is max ~8 sec. 4 clips ≈ 32 sec.")
        st.caption(f"📏 Total: ~{num_clips * duration} seconds "
                   f"({num_clips} clips × {duration}s)")
        story_lines = st.text_area(
            "Story prompts — one clip per line (blank lines repeat the main prompt above)",
            height=110,
            placeholder="Clip 1: plumber waves hello at camera\n"
                        "Clip 2: plumber fixes the sink, water splashes\n"
                        "Clip 3: plumber gives thumbs up, happy ending",
        )

    st.divider()
    st.subheader("🎙️ Voiceover (optional)")
    st.caption("Write a script — a free AI voice will be added to your video. Leave it blank for a silent video.")
    voice_script = st.text_area(
        "Voiceover script",
        height=100,
        placeholder="Is your drain clogged? Call EasyFix Plumbing now — same-day service, only $110!",
    )
    voice_label = st.selectbox(
        "Voice",
        list(VOICES.keys()),
        index=0,
    )

    if st.button("🚀 Generate Video", type="primary", use_container_width=True, key="t2v_go"):
        if not prompt.strip():
            st.error("✏️ Please write a prompt first, then click Generate.")
        else:
            status = st.status("🎥 Starting generation…", expanded=True)
            try:
                base = dict(
                    negative=negative.strip(),
                    image_path=None,
                    width=width, height=height,
                    duration=float(duration),
                    seed=seed_param,
                    guidance=guidance,
                )
                if story_mode:
                    lines = [l.strip() for l in story_lines.split("\n")
                             if l.strip()]
                    clip_prompts = [
                        lines[i] if i < len(lines) else prompt.strip()
                        for i in range(num_clips)
                    ]
                    video_bytes, used_label, n_clips = generate_story(
                        status, preferred_space, clip_prompts, **base)
                    fname = f"story_{int(time.time())}.mp4"
                    label_done = f"{n_clips}-clip story"
                else:
                    video_bytes, used_label, _sid = generate_with_fallback(
                        "t2v", status, preferred_space,
                        prompt=prompt.strip(), **base)
                    fname = f"t2v_{int(time.time())}.mp4"
                    label_done = "Video"
                # Voiceover (optional)
                script = voice_script.strip()
                if script:
                    try:
                        status.update(label="🎙️ Generating voiceover…",
                                      state="running")
                        voice_id = VOICES[voice_label]
                        audio_bytes = generate_voiceover(script, voice_id)
                        status.update(label="🔊 Adding voice to your video…",
                                      state="running")
                        video_bytes = mux_voiceover(video_bytes, audio_bytes)
                        label_done += " + voiceover"
                        fname = fname.replace(".mp4", "_voice.mp4")
                    except Exception as ve:
                        status.write(f"⚠️ Voiceover failed ({ve}) — delivering the silent video instead.")
                st.session_state["t2v_video"] = video_bytes
                st.session_state["t2v_name"] = fname
                st.session_state["t2v_space_used"] = used_label
                st.session_state["t2v_label_done"] = label_done
                # New video -> ad + survey gates reset so they show for EVERY video
                st.session_state["t2v_ad_done"] = False
                st.session_state["t2v_survey_done"] = False
                st.rerun()
            except Exception as exc:  # never crash the app — explain instead
                st.error(_friendly_space_error(exc))
                with st.expander("🔧 Technical details (for debugging)"):
                    st.code("".join(traceback.format_exception(exc))[-3000:])

    st.markdown(
        '<div class="tip">💡 <b>Tip:</b> short, clear prompts work best. '
        "Describe style and motion separately, e.g. <i>'slow cinematic pan, golden hour lighting'</i>.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("🎭 Character Acting Guide — get great performances from your characters"):
        st.write(
            "**3 rules:**\n"
            "1. **Describe the face + emotion** — e.g. `expressive face, smiling warmly`, "
            "`determined eyes`\n"
            "2. **Describe body language** — e.g. `natural hand gestures`, `walking confidently`\n"
            "3. **Keep the camera close** — e.g. `medium close-up shot` so faces stay clear\n\n"
            "**Example 1:**\n"
            "`A cheerful plumber with a friendly expressive face, wearing blue overalls, "
            "looks at the camera and gives a thumbs up with a big smile, natural hand motion, "
            "medium close-up, bright workshop lighting`\n\n"
            "**Example 2:**\n"
            "`A brave firefighter woman with determined expressive eyes runs toward the camera "
            "through light smoke, coat flowing naturally, cinematic slow motion, dramatic lighting`"
        )
    show_result("t2v", "Video")


# ---------------- Page: Image to Video ----------------
elif page == PAGE_I2V:
    st.subheader("Upload a photo → animate it")
    st.caption("⚡ Auto-fallback is ON: if one engine is busy, the app automatically tries the next one.")
    uploaded = st.file_uploader(
        "Upload a photo (JPG / PNG)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded is not None:
        st.image(uploaded, caption="Source image", width=420)

    motion = st.text_area(
        "Motion prompt",
        height=100,
        placeholder="The camera slowly zooms in while the person smiles and waves, "
                    "natural motion, cinematic",
    )
    negative2 = st.text_input(
        "Negative prompt (optional)",
        value="blurry, low quality, distorted faces, watermark, text",
    )

    if st.button("🚀 Animate Image", type="primary", use_container_width=True, key="i2v_go"):
        if uploaded is None:
            st.error("🖼️ Please upload a photo first.")
        elif not motion.strip():
            st.error("✏️ Please write a motion prompt — what should move in the photo?")
        else:
            # gradio_client needs a real file path for image upload
            with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            status = st.status("🎞️ Starting animation…", expanded=True)
            try:
                video_bytes, used_label, _space_id = generate_with_fallback(
                    "i2v", status, preferred_space,
                    prompt=motion.strip(),
                    negative=negative2.strip(),
                    image_path=tmp_path,
                    width=width, height=height,
                    duration=float(duration),
                    seed=seed_param,
                    guidance=guidance,
                )
                fname = f"i2v_{int(time.time())}.mp4"
                st.session_state["i2v_video"] = video_bytes
                st.session_state["i2v_name"] = fname
                st.session_state["i2v_space_used"] = used_label
                # New video -> ad + survey gates reset so they show for EVERY video
                st.session_state["i2v_ad_done"] = False
                st.session_state["i2v_survey_done"] = False
                st.rerun()
            except Exception as exc:  # never crash the app — explain instead
                st.error(_friendly_space_error(exc))
                with st.expander("🔧 Technical details (for debugging)"):
                    st.code("".join(traceback.format_exception(exc))[-3000:])

    st.markdown(
        '<div class="tip">💡 <b>Tip:</b> clean, well-lit photos animate best. '
        "For great <b>character acting</b>: upload a clear photo of your character + "
        "describe the facial emotion in the motion prompt "
        "(e.g. <i>'smiling warmly, expressive eyes'</i>).</div>",
        unsafe_allow_html=True,
    )
    show_result("i2v", "Animated video")


# ---------------- Page: Surveys ----------------
elif page == PAGE_SURVEYS:
    st.subheader("📋 Surveys")
    st.write(
        "Complete the surveys below — new surveys are added from time to time."
    )
    if not SURVEY_LINKS:
        st.info("📝 No active surveys right now — check back soon!")
    for _s in SURVEY_LINKS:
        with st.container(border=True):
            st.markdown(f"**{_s.get('title', 'Survey')}**")
            if _s.get("desc"):
                st.caption(_s["desc"])
            st.link_button("▶ Start Survey", _s["url"])


# ---------------- Page: Stock Videos ----------------
elif page == PAGE_STOCK:
    st.subheader("🎞️ Stock Videos")
    st.caption("Free stock footage — **no quotas, no limits!** Perfect for plumbing, electrical, HVAC, and more.")
    src = st.radio("Source", ["Studio Library", "Premium Library"], horizontal=True,
                   help="Studio Library works instantly, no key needed.")
    _is_pixabay = (src == "Studio Library")  # Studio Library = Pixabay backend
    if _is_pixabay:
        _default_px = PIXABAY_API_KEY or ""
        stock_key = st.text_input(
            "Library API Key (already filled — free)",
            type="password",
            key="pixabay_key",
            value=_default_px,
            placeholder="Your free API key",
        ) or _default_px
    else:
        stock_key = st.text_input(
            "Premium Library API Key",
            type="password",
            key="pexels_key",
            placeholder="Enter key to unlock",
        )
    # Quick preset buttons for the 7 home services
    presets = {
        "🔧 Plumbing": "plumber plumbing repair",
        "⚡ Electrical": "electrician electrical work",
        "🚪 Garage Door": "garage door repair",
        "🏠 Roofing": "roofing roof repair",
        "🔌 Appliance": "appliance repair technician",
        "❄️ HVAC": "hvac air conditioner technician",
        "🎨 Painting": "house painting painter",
    }
    cols = st.columns(4)
    for i, (label, q) in enumerate(presets.items()):
        if cols[i % 4].button(label, use_container_width=True):
            st.session_state["pexels_query"] = q
    query = st.text_input(
        "Search videos",
        value=st.session_state.get("pexels_query", "plumber"),
        key="pexels_search_box",
    )
    if st.button("🔍 Search", type="primary") and query:
        if not stock_key:
            st.warning(f"⚠️ Please enter your {src} API key first (free)")
        else:
            with st.spinner(f"Searching {src}…"):
                try:
                    if _is_pixabay:
                        results = _pixabay_search(query, stock_key)
                    else:
                        results = _pexels_search(query, stock_key)
                    st.session_state["pexels_results"] = results
                    st.session_state["pexels_src"] = src
                except Exception as e:
                    st.error(f"Search failed: {e}")
    for r in st.session_state.get("pexels_results", []):
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                if r["preview"]:
                    st.image(r["preview"], use_container_width=True)
            with c2:
                st.write(f"⏱️ {r['duration']}s • 📐 {r['width']}×{r['height']}")
                st.link_button("▶ View on source site", r["url"])
                # Download via server
                if st.button(f"⬇️ Download", key=f"pexdl_{r['id']}"):
                    with st.spinner("Downloading…"):
                        try:
                            req = urllib.request.Request(
                                r["file"],
                                headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req, timeout=120) as resp:
                                data = resp.read()
                            st.session_state[f"pexvid_{r['id']}"] = data
                            st.success(f"✅ Downloaded ({len(data)//1024} KB)")
                        except Exception as e:
                            st.error(f"Download failed: {e}")
                data = st.session_state.get(f"pexvid_{r['id']}")
                if data:
                    st.download_button(
                        "💾 Save MP4",
                        data=data,
                        file_name=f"pexels-{r['id']}.mp4",
                        mime="video/mp4",
                        key=f"pexsave_{r['id']}",
                    )


# ---------------- Page: Colab GPU ----------------
elif page == PAGE_COLAB:
    st.subheader("📓 Colab GPU — Free Lip-Sync Studio")
    st.info("🎭 **Make any photo talk!** Upload a character photo + audio in Colab (free T4 GPU) and get a lip-synced video. Free Colab has daily usage limits — if the GPU is busy, try again later.")
    st.markdown("""
**🔄 The Workflow:**
1. 📓 **Lip-sync your character on Colab** (free T4 GPU, ~3 min per clip)
2. 🎙️ **Make the voiceover here** in Voiceover Studio (or upload your own MP3)
3. ⬇️ **Download** the talking video from Colab

**100% free!** 🎉
""")
    st.divider()
    st.markdown("""
**How to use it:**
1. Open [Google Colab](https://colab.research.google.com) and start a new notebook
2. **Runtime → Change runtime type** → select **T4 GPU** → Save
3. **Copy** the code below into a cell and **run** it (▶ button)
4. When it asks, **upload** your character photo and your voiceover MP3
5. **Download** the lip-synced video when it's done!
""")
    colab_code = '''# 🎭 Free Lip-Sync (SadTalker on Colab T4)
# Run all cells in order. ~3 min per clip on free T4.

# Cell 1: Install
!git clone -q https://github.com/OpenTalker/SadTalker /content/SadTalker
%cd /content/SadTalker
!pip install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
!pip install -q -r requirements.txt

# Cell 2: Models (~2GB, one-time)
%cd /content/SadTalker
!bash scripts/download_models.sh

# Cell 3: Upload your files, then run
from google.colab import files
print("Upload a character PHOTO (square works best):")
photo = list(files.upload().keys())[0]
print("Upload your VOICEOVER mp3:")
audio = list(files.upload().keys())[0]

# Cell 4: Lip-sync!
!python inference.py --driven_audio "$audio" --source_image "$photo" \
    --result_dir /content/out --still --preprocess full --size 256
print("✅ Done! Download your video from /content/out")
'''
    st.code(colab_code, language="python")
    st.caption("💡 Tip: square character photos give the best results!")


# ---------------- Page: Voiceover Studio ----------------
elif page == PAGE_VOICE:
    st.subheader("🎙️ Voiceover Studio")
    st.caption("Turn any script into a professional AI voiceover — free, with no quotas. Download the MP3 or attach it to your own video.")

    vs_script = st.text_area(
        "Voiceover script",
        height=140,
        placeholder="Is your drain clogged? Call EasyFix Plumbing now — same-day service, only $110!",
        key="vs_script",
    )
    vs_voice_label = st.selectbox("Voice", list(VOICES.keys()), index=0, key="vs_voice")

    if st.button("🎙️ Generate Voiceover", type="primary", use_container_width=True, key="vs_go"):
        if not vs_script.strip():
            st.error("✏️ Please write a script first.")
        else:
            try:
                with st.spinner("🎙️ Generating your voiceover…"):
                    vs_audio = generate_voiceover(vs_script.strip(), VOICES[vs_voice_label])
                st.session_state["vs_audio"] = vs_audio
                st.session_state["vs_name"] = f"voiceover_{int(time.time())}.mp3"
            except Exception as e:
                st.error(f"⚠️ Voiceover failed: {e}")

    if st.session_state.get("vs_audio"):
        st.success("✅ Your voiceover is ready!")
        st.audio(st.session_state["vs_audio"], format="audio/mp3")
        st.download_button(
            "⬇️ Download MP3",
            data=st.session_state["vs_audio"],
            file_name=st.session_state.get("vs_name", "voiceover.mp3"),
            mime="audio/mp3",
            use_container_width=True,
            key="vs_dl",
        )

    st.divider()
    with st.expander("🎬 Attach this voiceover to your own video"):
        st.caption("Upload any MP4 and the studio will merge your voiceover into it — free.")
        vs_video = st.file_uploader("Upload your video (MP4)", type=["mp4"], key="vs_video")
        if st.button("🔊 Combine Video + Voiceover", use_container_width=True, key="vs_mux"):
            if vs_video is None:
                st.error("🖼️ Please upload a video first.")
            elif not st.session_state.get("vs_audio"):
                st.error("🎙️ Please generate a voiceover first.")
            else:
                try:
                    with st.spinner("🔊 Merging voiceover into your video…"):
                        vs_final = mux_voiceover(vs_video.getvalue(), st.session_state["vs_audio"])
                    st.session_state["vs_final"] = vs_final
                    st.success("✅ Done — your video now has a voiceover!")
                except Exception as e:
                    st.error(f"⚠️ Could not merge audio: {e}")
        if st.session_state.get("vs_final"):
            st.video(st.session_state["vs_final"], format="video/mp4")
            st.download_button(
                "⬇️ Download Video with Voiceover",
                data=st.session_state["vs_final"],
                file_name=f"video_voice_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True,
                key="vs_final_dl",
            )


# ----------------------------------------------------------------------------
# Footer — professional branding
# ----------------------------------------------------------------------------
st.divider()
st.caption(
    "© 2026 Imposter | CEO: Muhammad Mudassir Yaseen | All rights reserved."
)
st.markdown(
    '<p class="footer">© 2026 Imposter &nbsp;|&nbsp; CEO: Muhammad Mudassir Yaseen</p>',
    unsafe_allow_html=True,
)
