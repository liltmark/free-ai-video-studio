"""
🎬 Free AI Video Studio — Gradio Edition
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
# OWNER SETTINGS — inko Muse tumhare kehne par update karega
# ----------------------------------------------------------------------------
# Apne ad network (Adsterra/PropellerAds waghera) ka <script> ya <iframe> code
# yahan paste karo. Jab tak khaali hai, ad ki jagah placeholder nazar aayega.
AD_HTML = ""

# Tool khulne se pehle user ko kitne second ad dekhna lazmi hai:
AD_WAIT_SECONDS = 10

# Survey links — naye links jab bhi bhejo, Muse yahan add kar dega:
# {"title": "...", "url": "https://...", "desc": "..."}
SURVEY_LINKS = [
    # {"title": "Sample Survey", "url": "https://example.com", "desc": "Mukhtasar wazahat"},
]




# ----------------------------------------------------------------------------
# Public Gradio Spaces (verified live via gradio_client)
# ----------------------------------------------------------------------------
# Har Space ke liye ek adapter function hai jo sahi endpoint + parameters
# ke saath job submit karta hai. Koi API key / token nahi chahiye.

SPACE_LTX = "Lightricks/ltx-video-distilled"      # official LTX-Video demo
SPACE_COG5 = "zai-org/CogVideoX-5B-Space"         # official CogVideoX 5B demo
SPACE_COG2 = "zai-org/CogVideoX-2B-Space"         # official CogVideoX 2B demo
SPACE_WAN = "Upsampler/wan-2-2-14b-image-to-video"  # Wan 2.2 14B I2V (verified 2026-10-13)

JOB_TIMEOUT = 1800  # seconds (30 min) — public queues can be slow at peak times


def _ltx_submit(client: Client, kind: str, prompt: str, negative: str,
                image_path, width: int, height: int, duration: float,
                seed: int, guidance: float):
    """Submit to LTX-Video Space. kind: 't2v' or 'i2v'."""
    rnd = (seed == -1)
    frames = max(9, min(257, int(duration * 24)))
    api = "/text_to_video" if kind == "t2v" else "/image_to_video"
    mode = "text-to-video" if kind == "t2v" else "image-to-video"
    return client.submit(
        prompt=prompt,
        negative_prompt=negative,
        input_image_filepath=image_path,      # None for text-to-video
        input_video_filepath=None,
        height_ui=height,
        width_ui=width,
        mode=mode,
        duration_ui=duration,
        ui_frames_to_use=frames,
        seed_ui=0 if rnd else seed,
        randomize_seed=rnd,
        ui_guidance_scale=guidance,
        improve_texture_flag=False,
        api_name=api,
    )


def _cog_submit(client: Client, kind: str, prompt: str, negative: str,
                image_path, width: int, height: int, duration: float,
                seed: int, guidance: float):
    """Submit to CogVideoX Space (/generate handles both t2v and i2v)."""
    seed_val = random.randint(0, 2 ** 31 - 1) if seed == -1 else seed
    return client.submit(
        prompt=prompt,
        image_input=image_path,               # None → text-to-video
        video_input=None,
        video_strength=0.5,
        seed_value=seed_val,
        scale_status=False,
        rife_status=False,
        api_name="/generate",
    )


def _wan_submit(client: Client, kind: str, prompt: str, negative: str,
                image_path, width: int, height: int, duration: float,
                seed: int, guidance: float):
    """Submit to Wan 2.2 14B I2V Space. I2V ONLY (no T2V endpoint)."""
    from gradio_client.utils import handle_file
    if kind != "i2v" or not image_path:
        raise ValueError("Wan Space sirf image-to-video karta hai")
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


# Space registry: label + submit adapter. Order = fallback priority.
SPACE_ADAPTERS = {
    SPACE_LTX:  {"label": "LTX-Video (fast)", "fn": _ltx_submit},
    SPACE_COG5: {"label": "CogVideoX-5B",     "fn": _cog_submit},
    SPACE_COG2: {"label": "CogVideoX-2B",     "fn": _cog_submit},
    SPACE_WAN:  {"label": "Wan 2.2 14B (I2V)", "fn": _wan_submit},
}

T2V_CHAIN = [SPACE_LTX, SPACE_COG5, SPACE_COG2]
# I2V reality (verified 2026-10-13): LTX /image_to_video raises a server-side
# exception; CogVideoX-2B /generate has NO image param at all (T2V only).
# Working I2V: Wan 2.2 14B (verified live) + CogVideoX-5B (quota-unstable).
I2V_CHAIN = [SPACE_WAN, SPACE_COG5]
# Story mode chains clips via image-to-video. Wan 2.2 14B is primary
# (verified working); CogVideoX-5B is fallback.
STORY_CHAIN = [SPACE_WAN, SPACE_COG5]

_client_cache: dict = {}


def get_client(space_id: str) -> Client:
    """Cache one gradio_client per Space (connect handshake is slow)."""
    if space_id not in _client_cache:
        _client_cache[space_id] = Client(space_id)
    return _client_cache[space_id]


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


def _friendly_space_error(exc: Exception) -> str:
    low = str(exc).lower()
    if "queue_full" in low or "queue is full" in low:
        return ("⏳ **Sab Spaces busy hain.** Public free Spaces par rush hota hai — "
                "kuch minute ruk kar dobara try karein.")
    if "timeout" in low or "timed out" in low:
        return ("⏳ **Request timeout.** Space ne jawab nahi diya (lambi queue). "
                "Dobara try karein — app khud agla Space try karegi.")
    if "connection" in low or "unreachable" in low or "failed to connect" in low:
        return ("🔌 **Space se connect nahi ho saka.** Shayad wo Space down hai — "
                "app ne agla Space try kiya hoga.")
    short = str(exc).strip()[:350]
    return f"⚠️ **Generation fail ho gayi:** {short}"


def generate_with_fallback(kind: str, status, preferred: str | None = None,
                           pin_space: str | None = None, **params):
    """Generate via public Spaces with automatic fallback.

    Tries the preferred Space first, then the rest of the chain.
    If pin_space is set, ONLY that Space is used (for story-mode consistency).
    Each Space gets one retry on transient errors. Live queue progress
    is shown in the `status` (st.status) container.
    Returns (video_bytes, space_label, space_id). Raises last error if all fail.
    """
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
        for attempt in (1, 2):
            try:
                status.update(label=f"🎥 Connecting to {label}…", state="running")
                client = get_client(space_id)
                job = submit_fn(client, kind, **params)
                start = time.time()
                last_qpos = None
                while not job.done():
                    if time.time() - start > JOB_TIMEOUT:
                        try:
                            job.cancel()
                        except Exception:
                            pass
                        raise TimeoutError(f"{label}: 30 min se zyada lag gaya")
                    try:
                        stt = job.status()
                        code = stt.code if hasattr(stt, "code") else None
                        qpos = getattr(stt, "queue_position", None)
                        if code == Status.IN_QUEUE and qpos and qpos != last_qpos:
                            last_qpos = qpos
                            status.update(
                                label=f"🎥 {label} — queue mein #{qpos}…",
                                state="running")
                        elif code == Status.PROCESSING:
                            status.update(
                                label=f"🎥 {label} — video ban rahi hai…",
                                state="running")
                    except Exception:
                        pass
                    time.sleep(10)
                video_bytes = _result_to_bytes(job.result())
                status.update(label=f"✅ Video ready — via {label}",
                              state="complete")
                return video_bytes, label, space_id
            except Exception as exc:  # noqa: BLE001 — fallback below
                last_exc = exc
                low = str(exc).lower()
                # Stale API info in cached client (Space updated mid-session):
                # drop the client so the next attempt re-fetches the API.
                if "not a valid key-word argument" in low:
                    _client_cache.pop(space_id, None)
                    if attempt == 1:
                        status.write(f"🔄 {label} ka API refresh kar raha hun…")
                        time.sleep(3)
                        continue
                transient = any(k in low for k in (
                    "timeout", "timed out", "queue", "busy", "429", "503",
                    "connection", "temporarily"))
                if transient and attempt == 1:
                    status.write(f"⏳ {label} busy — dobara try kar raha hun…")
                    time.sleep(5)
                    continue
                status.write(f"⚠️ {label} fail — agla Space try kar raha hun…")
                break  # → next Space

    status.update(label="❌ Sab Spaces fail/busy", state="error")
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
        raise RuntimeError("Voiceover khali aya — text check karein")
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
    """Multi-clip story: clip 1 via text-to-video, baqi clips image-to-video
    mein pichli clip ke LAST FRAME se chain — taake sab ek hi video lagein.
    I2V clips pehli successful I2V Space par pin hoti hain (Wan 2.2 14B
    primary, CogVideoX-5B fallback). Returns (stitched_bytes, space_label, num_clips)."""
    clips: list = []
    label = None
    total = len(clip_prompts)
    i2v_space = None  # pin to first successful I2V Space
    for i, cp in enumerate(clip_prompts):
        p = dict(base_params)
        p["prompt"] = cp
        if i == 0:
            p["image_path"] = None
            status.write(f"🎬 **Clip 1/{total}** — shuruat…")
            vb, label, _sid = generate_with_fallback(
                "t2v", status, preferred, **p)
        else:
            # Pichli clip ka aakhri frame = agli clip ki shuruat (link!)
            p["image_path"] = extract_last_frame(clips[-1])
            status.write(f"🎬 **Clip {i+1}/{total}** — pichli se jor raha hun…")
            vb, label, i2v_space = generate_with_fallback(
                "i2v", status, None, pin_space=i2v_space, **p)
        clips.append(vb)
    status.update(label="🔗 Clips ko ek video mein jor raha hun…",
                  state="running")
    final = stitch_clips(clips)
    status.update(label=f"✅ {total} clips — ek lambi video ready (via {label})",
                  state="complete")
    return final, label, total


# ----------------------------------------------------------------------------
# Page setup + styling
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="🎬 Free AI Video Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #6d28d9 0%, #db2777 60%, #f59e0b 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub { color: #9ca3af; margin-top: 0.2rem; margin-bottom: 1rem; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6d28d9, #db2777);
        border: none;
        font-weight: 700;
    }
    .tip {
        background: rgba(109, 40, 217, 0.12);
        border: 1px solid rgba(109, 40, 217, 0.35);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.6rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero">🎬 Free AI Video Studio</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Text-to-Video &amp; Image-to-Video — 100% free, no API keys. '
    "Powered by public Hugging Face Spaces (LTX-Video, CogVideoX).</p>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Sidebar — settings (NO token needed!)
# ----------------------------------------------------------------------------

RESOLUTIONS = {
    "Portrait 720×1280 (Shorts)": (1280, 720),   # (width, height)
    "Square 512×512": (512, 512),
    "Landscape 1280×720": (1280, 720),
}

with st.sidebar:
    st.header("⚙️ Settings")
    st.success("🔓 **No API key needed!**\nYe tool public free Spaces use karta hai.")

    space_options = ["Auto (recommended — sab try karega)"] + [
        f"{SPACE_ADAPTERS[s]['label']} ({s})" for s in T2V_CHAIN
    ]
    space_choice = st.selectbox(
        "Preferred Space",
        space_options,
        help="Auto par app khud best available Space chunti hai; "
             "agar wo busy ho to agli par chali jati hai.",
    )
    preferred_space = None
    if not space_choice.startswith("Auto"):
        preferred_space = space_choice.rsplit("(", 1)[1].rstrip(")")

    st.divider()
    st.subheader("🎛️ Generation settings")
    res_label = st.selectbox("Resolution", list(RESOLUTIONS.keys()), index=0)
    width, height = RESOLUTIONS[res_label]
    duration = st.slider(
        "Duration (seconds)",
        min_value=1, max_value=8, value=3, step=1,
        help="Lambi video = zyada queue/wait time.",
    )
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

    st.divider()
    with st.expander("💡 Free-tier reality check"):
        st.write(
            "• **Bilkul free — koi token/API key nahi chahiye.**\n"
            "• Video **public Hugging Face Spaces** (LTX-Video, CogVideoX) par banti hai.\n"
            "• **Queue:** free Spaces par rush hota hai — busy time mein 10–30 min "
            "lag sakte hain. App khud agla Space try karti hai.\n"
            "• **Space hosting:** free (Streamlit Community Cloud)."
        )


# ----------------------------------------------------------------------------
# Ad gate — tool use karne se pehle ek ad dekhna lazmi
# ----------------------------------------------------------------------------
if "ad_watched" not in st.session_state:
    st.session_state.ad_watched = False
    st.session_state.gate_start = time.time()

if not st.session_state.ad_watched:
    st.subheader("📺 Pehle ek chhota sa ad dekh lein")
    st.write(
        "Ye tool **bilkul free** hai — ads hi isay free rakhte hain. "
        f"Neeche ad hai; **{AD_WAIT_SECONDS} second** ruk kar Continue dabayein, "
        "phir tool khul jayega."
    )
    _ad_html = AD_HTML.strip() or """
        <div style="background:#1f2937;border:2px dashed #6b7280;border-radius:12px;height:250px;
                    display:flex;align-items:center;justify-content:center;color:#9ca3af;
                    font-family:sans-serif;font-size:18px;">📢 Ad yahan lagega</div>
    """
    try:
        components.html(_ad_html, height=300, scrolling=True)
    except Exception:
        st.info("📢 Ad yahan lagega.")
    if st.button("✅ Ad dekh liya — Tool kholo", type="primary", use_container_width=True):
        _waited = time.time() - st.session_state.gate_start
        if _waited < AD_WAIT_SECONDS:
            st.warning(f"⏳ Thoda aur ruken — {int(AD_WAIT_SECONDS - _waited)} second baqi hain.")
        else:
            st.session_state.ad_watched = True
            st.rerun()
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎬 Text to Video", "🖼️ Image to Video", "📋 Surveys", "🎞️ Pexels Stock", "📓 Colab GPU"])


def show_result(key_prefix: str, label: str) -> None:
    """Render the last generated video (persisted in session state) + download."""
    data = st.session_state.get(f"{key_prefix}_video")
    name = st.session_state.get(f"{key_prefix}_name", "video.mp4")
    if not data:
        return
    done_label = st.session_state.get(f"{key_prefix}_label_done", label)
    st.success(f"✅ {done_label} ready!")
    space_used = st.session_state.get(f"{key_prefix}_space_used")
    if space_used:
        st.caption(f"🤖 Generated with: **{space_used}** (public free Space)")
    st.video(data, format="video/mp4")
    st.download_button(
        "⬇️ Download MP4",
        data=data,
        file_name=name,
        mime="video/mp4",
        use_container_width=True,
    )


# ---------------- Tab 1: Text to Video ----------------
with tab1:
    st.subheader("Describe it → get an MP4")
    st.caption("⚡ Auto-fallback ON: ek Space busy ho to app khud agla try karegi. Koi key nahi chahiye.")
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
        "🎞️ Story mode — linked clips, ek lambi video (e.g. 30 sec)",
        help="Har clip pichli clip ke aakhri frame se shuru hoti hai — "
             "taake sab ek hi video ke parts lagein. "
             "Story mode Wan 2.2 14B (I2V) use karta hai — thora slow ho sakta hai.",
    )
    num_clips = 4
    story_lines = ""
    if story_mode:
        num_clips = st.slider("Kitne clips?", min_value=2, max_value=6, value=4,
                              help="Har clip max ~8 sec. 4 clips ≈ 32 sec.")
        st.caption(f"📏 Total: ~{num_clips * duration} seconds "
                   f"({num_clips} clips × {duration}s)")
        story_lines = st.text_area(
            "Story prompts — har line ek clip (khali line = upar wala prompt repeat)",
            height=110,
            placeholder="Clip 1: plumber waves hello at camera\n"
                        "Clip 2: plumber fixes the sink, water splashes\n"
                        "Clip 3: plumber gives thumbs up, happy ending",
        )

    st.divider()
    st.subheader("🎙️ Voiceover (optional)")
    st.caption("Script likhein — video mein free AI voice add ho jayegi. Khali chhorein to silent video banegi.")
    voice_script = st.text_area(
        "Voiceover script",
        height=100,
        placeholder="Is your drain clogged? Call EasyFix Plumbing now — same day service, only $110!",
    )
    voice_label = st.selectbox(
        "Voice",
        list(VOICES.keys()),
        index=0,
    )

    if st.button("🚀 Generate Video", type="primary", use_container_width=True, key="t2v_go"):
        if not prompt.strip():
            st.error("✏️ Pehle koi prompt likhein — phir Generate dabayein.")
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
                        status.update(label="🎙️ Voiceover bana raha hun…",
                                      state="running")
                        voice_id = VOICES[voice_label]
                        audio_bytes = generate_voiceover(script, voice_id)
                        status.update(label="🔊 Voice video mein jor raha hun…",
                                      state="running")
                        video_bytes = mux_voiceover(video_bytes, audio_bytes)
                        label_done += " + voiceover"
                        fname = fname.replace(".mp4", "_voice.mp4")
                    except Exception as ve:
                        status.write(f"⚠️ Voiceover fail ({ve}) — silent video de raha hun.")
                st.session_state["t2v_video"] = video_bytes
                st.session_state["t2v_name"] = fname
                st.session_state["t2v_space_used"] = used_label
                st.session_state["t2v_label_done"] = label_done
                st.rerun()
            except Exception as exc:  # never crash the app — explain instead
                st.error(_friendly_space_error(exc))
                with st.expander("🔧 Technical details (for debugging)"):
                    st.code("".join(traceback.format_exception(exc))[-3000:])

    st.markdown(
        '<div class="tip">💡 <b>Tip:</b> chhote, wazeh prompts behtar kaam karte hain. '
        "Style/motion alag se likhein, e.g. <i>'slow cinematic pan, golden hour lighting'</i>.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("🎭 Character acting guide — characters se achi performance len"):
        st.write(
            "**3 usool:**\n"
            "1. **Chehra + emotion likho** — e.g. `expressive face, smiling warmly`, "
            "`determined eyes`\n"
            "2. **Body language likho** — e.g. `natural hand gestures`, `walking confidently`\n"
            "3. **Camera qareeb rakho** — e.g. `medium close-up shot` taake chehra clear aaye\n\n"
            "**Example 1:**\n"
            "`A cheerful plumber with a friendly expressive face, wearing blue overalls, "
            "looks at the camera and gives a thumbs up with a big smile, natural hand motion, "
            "medium close-up, bright workshop lighting`\n\n"
            "**Example 2:**\n"
            "`A brave firefighter woman with determined expressive eyes runs toward the camera "
            "through light smoke, coat flowing naturally, cinematic slow motion, dramatic lighting`"
        )
    show_result("t2v", "Video")


# ---------------- Tab 2: Image to Video ----------------
with tab2:
    st.subheader("Upload a photo → animate it")
    st.caption("⚡ Auto-fallback ON: ek Space busy ho to app khud agla try karegi. Koi key nahi chahiye.")
    uploaded = st.file_uploader(
        "Photo upload karein (JPG / PNG)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded is not None:
        st.image(uploaded, caption="Input image", width=420)

    motion = st.text_area(
        "Motion prompt",
        height=100,
        placeholder="The camera slowly zooms in while the person smiles and waves, "
                    "natural motion, cinematic",
    )
    negative2 = st.text_input(
        "Negative prompt (optional) ",
        value="blurry, low quality, distorted faces, watermark, text",
    )

    if st.button("🚀 Animate Image", type="primary", use_container_width=True, key="i2v_go"):
        if uploaded is None:
            st.error("🖼️ Pehle koi photo upload karein.")
        elif not motion.strip():
            st.error("✏️ Motion prompt likhein — tasveer mein kya harkat ho?")
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
                st.rerun()
            except Exception as exc:  # never crash the app — explain instead
                st.error(_friendly_space_error(exc))
                with st.expander("🔧 Technical details (for debugging)"):
                    st.code("".join(traceback.format_exception(exc))[-3000:])


    st.markdown(
        '<div class="tip">💡 <b>Tip:</b> saaf, roshan tasveerain behtar animate hoti hain. '
        "Behtreen <b>character acting</b> ke liye: character ki saaf photo upload karo + "
        "motion prompt mein chehre ka emotion likho "
        "(e.g. <i>'smiling warmly, expressive eyes'</i>).</div>",
        unsafe_allow_html=True,
    )
    show_result("i2v", "Animated video")


# ---------------- Tab 3: Surveys ----------------
with tab3:
    st.subheader("📋 Surveys")
    st.write(
        "Neeche diye gaye surveys mukammal karen — "
        "naye surveys waqtan fawaqtan add hote rehte hain."
    )
    if not SURVEY_LINKS:
        st.info("📝 Abhi koi survey active nahi hai — jald aa rahe hain!")
    for _s in SURVEY_LINKS:
        with st.container(border=True):
            st.markdown(f"**{_s.get('title', 'Survey')}**")
            if _s.get("desc"):
                st.caption(_s["desc"])
            st.link_button("▶ Survey shuru karen", _s["url"])


# ---------------- Tab 4: Pexels Stock Videos ----------------
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


with tab4:
    st.subheader("🎞️ Stock Videos (Pexels + Pixabay)")
    st.caption("Free stock footage — **koi quota nahi!** Plumbing, electrical, HVAC waghera ke liye.")
    src = st.radio("Source", ["Pixabay", "Pexels"], horizontal=True,
                   help="Pexels keys paused hain — Pixabay abhi free keys de raha hai.")
    if src == "Pixabay":
        stock_key = st.text_input(
            "Pixabay API Key (free: pixabay.com/api/docs)",
            type="password",
            key="pixabay_key",
            placeholder="Paste free API key",
        )
    else:
        stock_key = st.text_input(
            "Pexels API Key (free: pexels.com/api)",
            type="password",
            key="pexels_key",
            placeholder="Paste free API key",
            help="Note: Pexels ne naye keys pause kiye hain (Oct 2026).",
        )
    # Quick preset buttons for his 7 home services
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
            st.warning(f"⚠️ Pehle {src} API key dalein (free)")
        else:
            with st.spinner(f"Searching {src}…"):
                try:
                    if src == "Pixabay":
                        results = _pixabay_search(query, stock_key)
                    else:
                        results = _pexels_search(query, stock_key)
                    st.session_state["pexels_results"] = results
                    st.session_state["pexels_src"] = src
                except Exception as e:
                    st.error(f"Search fail: {e}")
    for r in st.session_state.get("pexels_results", []):
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                if r["preview"]:
                    st.image(r["preview"], use_container_width=True)
            with c2:
                st.write(f"⏱️ {r['duration']}s • 📐 {r['width']}×{r['height']}")
                st.link_button("▶ Pexels par dekhein", r["url"])
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
                            st.error(f"Download fail: {e}")
                data = st.session_state.get(f"pexvid_{r['id']}")
                if data:
                    st.download_button(
                        "💾 Save MP4",
                        data=data,
                        file_name=f"pexels-{r['id']}.mp4",
                        mime="video/mp4",
                        key=f"pexsave_{r['id']}",
                    )


st.divider()
st.caption(
    "Built with Streamlit + `gradio_client`. Videos via public Hugging Face "
    "Spaces: LTX-Video, CogVideoX — no API keys needed."
)


# ---------------- Tab 5: Colab GPU ----------------
with tab5:
    st.subheader("📓 Colab GPU — Free Video Generation")
    st.caption("Apna free T4 GPU — **koi quota nahi!** Slow hai (15-30 min/clip) lekin unlimited.")
    st.markdown("""
**Kaise use karein:**
1. [Google Colab](https://colab.research.google.com) kholein
2. **Runtime → Change runtime type → T4 GPU** select karein
3. Neeche wala code **copy** karke Colab ke cell mein **paste** karein
4. Apna **prompt** likhein aur cell **chalayein** (▶ button)
5. Video ban jayegi to **download** kar lein
""")
    colab_code = '''# 🎬 Free Video Generator (Wan 2.1 on Colab T4)
# Cell 1: Install
!pip install -q diffusers transformers accelerate

# Cell 2: Generate
import torch
from diffusers import WanPipeline

prompt = "Pixar-style 3D cartoon plumber with cat face, blue uniform, fixing a sink"  # <-- apna prompt yahan likhein

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")
video = pipe(prompt=prompt, num_frames=81, height=480, width=832).frames[0]

# Save
from diffusers.utils import export_to_video
export_to_video(video, "output.mp4")
print("✅ Video ready: output.mp4")
'''
    st.code(colab_code, language="python")
    st.caption("💡 Tip: Colab background mein chalega — aap tool mein aur kaam kar sakte hain!")
