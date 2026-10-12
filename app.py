"""
🎬 Free AI Video Studio
=======================
Text-to-Video + Image-to-Video web app powered by Hugging Face Inference Providers.

* Hosting ......... 100% FREE on Hugging Face Spaces (Streamlit, CPU basic tier)
* Video generation . runs on HF's inference providers (fal-ai, replicate, novita,
                      together, wavespeed) through the `huggingface_hub` SDK.
                      Usage is billed against your Hugging Face account's
                      "Inference Providers" credits. The free tier includes a
                      small monthly credit allowance — enough for a few short
                      test clips (NOT unlimited).

No GPU, no torch, no heavy downloads needed inside the Space: the Space is just
a lightweight front-end, the video models run on the providers' servers.
"""

from __future__ import annotations

import time
import traceback

import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

try:  # available in huggingface_hub >= 0.19
    from huggingface_hub import InferenceTimeoutError
except ImportError:  # pragma: no cover - very old huggingface_hub versions
    InferenceTimeoutError = Exception  # type: ignore[misc,assignment]


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
# Config
# ----------------------------------------------------------------------------

T2V_MODELS = {
    "Wan 2.2 (14B) — recommended": "Wan-AI/Wan2.2-T2V-A14B",
    "HunyuanVideo (Tencent)": "tencent/HunyuanVideo",
    "Mochi 1 Preview (Genmo)": "genmo/mochi-1-preview",
    "Wan 2.1 (14B)": "Wan-AI/Wan2.1-T2V-14B",
    "LTX-Video (Lightricks)": "Lightricks/LTX-Video",
    "CogVideoX-5B (gated: accept license on model page first)": "THUDM/CogVideoX-5b",
}

I2V_MODELS = {
    "Wan 2.2 (14B) — recommended": "Wan-AI/Wan2.2-I2V-A14B",
    "LTX-Video (Lightricks)": "Lightricks/LTX-Video",
    "Wan 2.1 (14B, 480p)": "Wan-AI/Wan2.1-I2V-14B-480P",
    "CogVideoX-5B (gated: accept license on model page first)": "THUDM/CogVideoX-5b",
}

# Providers that serve text-to-video / image-to-video through HF routing.
PROVIDERS = ["auto", "fal-ai", "replicate", "novita", "together", "wavespeed"]

REQUEST_TIMEOUT = 1500  # seconds (25 min) — video generation is slow


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
    '<p class="sub">Text-to-Video, Image-to-Video &amp; Surveys — powered by open-source models '
    "(Wan, HunyuanVideo, Mochi, LTX-Video, CogVideoX) via Hugging Face.</p>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def friendly_error(exc: Exception) -> str:
    """Turn provider/API failures into plain-language guidance. Never crashes."""
    raw = str(exc).strip() or repr(exc)
    low = raw.lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)

    if status == 401 or "unauthorized" in low or ("invalid" in low and "token" in low):
        return (
            "🔑 **Token problem.** Your Hugging Face token is invalid or expired.\n\n"
            "Fix: create a new **fine-grained** token at `hf.co/settings/tokens` with the "
            "**`Make calls to Inference Providers`** permission, then paste it in the sidebar."
        )
    if status == 402 or "insufficient credit" in low or "not enough credit" in low \
            or "credit balance" in low or "billing" in low:
        return (
            "💳 **Free credits exhausted.** Video generation is billed against your Hugging Face "
            "account's Inference Providers credits. The free tier only includes a **small monthly "
            "allowance** (a few short clips — not unlimited). It resets next month, or you can add "
            "credits to keep going."
        )
    if status == 403 and ("gated" in low or "license" in low or "accept" in low):
        return (
            "🔒 **Gated model.** Open the model page on huggingface.co while logged in and click "
            "**Agree / Accept** on its license, then press Generate again."
        )
    if status == 404 or "not found" in low or "no provider" in low \
            or "not supported" in low:
        return (
            "🔍 **That model isn't being served by any provider right now.** Pick a different "
            "model from the sidebar dropdown (Wan 2.2 usually works) or try another provider."
        )
    if status == 429 or "rate limit" in low or "too many requests" in low:
        return "⏳ **Rate limited.** Wait a minute or two, then press Generate again."
    if isinstance(exc, InferenceTimeoutError) or status == 503 \
            or "timeout" in low or "timed out" in low or "overloaded" in low:
        return (
            "🐢 **The provider is busy or the request timed out.** Cheap/free tiers queue up at "
            "busy times — this is normal. Wait 1–2 minutes and press **Generate** again. "
            "Nothing is broken."
        )
    short = raw if len(raw) < 400 else raw[:400] + "…"
    return f"⚠️ **Something went wrong:** {short}"


def make_client(token: str, provider: str) -> InferenceClient:
    return InferenceClient(token=token, provider=provider, timeout=REQUEST_TIMEOUT)


def valid_token(token: str) -> bool:
    return bool(token) and token.strip().startswith("hf_")


# ----------------------------------------------------------------------------
# Sidebar — credentials + generation settings
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    token = st.text_input(
        "Hugging Face Access Token (HF_TOKEN)",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxx",
        help="Get a FREE token: huggingface.co → profile → Settings → Access Tokens → "
             "Create new token (fine-grained, enable 'Make calls to Inference Providers').",
    )
    if token and not valid_token(token):
        st.warning("Token usually starts with `hf_` — double-check what you pasted.")

    provider = st.selectbox(
        "Provider",
        PROVIDERS,
        index=0,
        help="'auto' lets Hugging Face route to the fastest available video provider.",
    )

    st.divider()
    st.subheader("🎥 Text-to-Video model")
    t2v_label = st.selectbox("Model", list(T2V_MODELS.keys()), index=0)
    t2v_custom = st.text_input(
        "…or custom model ID (optional)",
        placeholder="e.g. Wan-AI/Wan2.2-T2V-A14B",
        help="Leave empty to use the dropdown choice above.",
    )
    t2v_model = t2v_custom.strip() or T2V_MODELS[t2v_label]

    st.subheader("🖼️ Image-to-Video model")
    i2v_label = st.selectbox("Model ", list(I2V_MODELS.keys()), index=0)
    i2v_custom = st.text_input(
        "…or custom model ID (optional) ",
        placeholder="e.g. Wan-AI/Wan2.2-I2V-A14B",
    )
    i2v_model = i2v_custom.strip() or I2V_MODELS[i2v_label]

    st.divider()
    st.subheader("🎛️ Generation settings")
    quality = st.radio(
        "Quality preset",
        ["Standard (fast)", "HD (best free quality ~720p)"],
        horizontal=True,
        help="HD = zyada steps + zyada frames = behtar quality, lekin slow. "
             "Free API par ~720p hi max milta hai.",
    )
    _hd = quality.startswith("HD")
    steps = st.slider(
        "Inference steps (quality ↔ speed)",
        min_value=10, max_value=50, value=45 if _hd else 30, step=1,
        help="More steps = better quality but slower generation.",
    )
    frames = st.slider(
        "Frames (clip length)",
        min_value=16, max_value=121, value=81 if _hd else 49, step=8,
        help="More frames = longer video. Providers usually render at ~24 fps.",
    )
    guidance = st.slider(
        "Guidance scale (prompt adherence)",
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        help="Higher = follows your prompt more strictly. Too high can look artificial.",
    )
    seed_in = st.number_input(
        "Seed (-1 = random)", min_value=-1, value=-1, step=1,
        help="Same seed + same prompt = reproducible video.",
    )
    seed = None if seed_in == -1 else int(seed_in)

    st.divider()
    with st.expander("💡 Free-tier reality check"):
        st.write(
            "• **Space hosting:** free (CPU basic). The Space sleeps after 48h of "
            "inactivity and wakes when you open it.\n"
            "• **HF token:** free.\n"
            "• **Video generation:** billed against your HF account's Inference "
            "Providers credits. The free tier includes only a **small monthly "
            "allowance** — enough for a few short test clips, not unlimited use. "
            "The app will tell you clearly if credits run out.\n"
            "• **Speed:** expect several minutes per clip on free/cheap tiers; "
            "queues get busy at peak times."
        )


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

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

tab1, tab2, tab3 = st.tabs(["🎬 Text to Video", "🖼️ Image to Video", "📋 Surveys"])


def show_result(key_prefix: str, label: str) -> None:
    """Render the last generated video (persisted in session state) + download."""
    data = st.session_state.get(f"{key_prefix}_video")
    name = st.session_state.get(f"{key_prefix}_name", "video.mp4")
    if not data:
        return
    st.success(f"✅ {label} ready!")
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
    prompt = st.text_area(
        "Prompt",
        height=120,
        placeholder="A cinematic drone shot flying over snowy mountains at golden hour, "
                    "photorealistic, smooth motion",
    )
    negative = st.text_input(
        "Negative prompt (optional)",
        value="blurry, low quality, distorted faces, watermark, text",
    )

    if st.button("🚀 Generate Video", type="primary", use_container_width=True, key="t2v_go"):
        if not valid_token(token):
            st.error("🔑 Pehle sidebar mein apna **HF_TOKEN** paste karein (free token: "
                     "`huggingface.co → Settings → Access Tokens`).")
        elif not prompt.strip():
            st.error("✏️ Pehle koi prompt likhein — phir Generate dabayein.")
        else:
            try:
                client = make_client(token.strip(), provider)
                kwargs: dict = {
                    "model": t2v_model,
                    "num_frames": frames,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                }
                if negative.strip():
                    kwargs["negative_prompt"] = [negative.strip()]  # API expects a list
                if seed is not None:
                    kwargs["seed"] = seed

                with st.spinner(
                    "🎥 Video ban rahi hai… is mein chand minute lag sakte hain. "
                    "Page band na karein."
                ):
                    video_bytes = client.text_to_video(prompt.strip(), **kwargs)

                fname = f"t2v_{int(time.time())}.mp4"
                st.session_state["t2v_video"] = video_bytes
                st.session_state["t2v_name"] = fname
                st.rerun()
            except Exception as exc:  # never crash the app — explain instead
                st.error(friendly_error(exc))
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
            "medium close-up, bright workshop lighting, photorealistic`\n\n"
            "**Example 2:**\n"
            "`A brave firefighter woman with determined expressive eyes runs toward the camera "
            "through light smoke, coat flowing naturally, cinematic slow motion, dramatic lighting`"
        )
    show_result("t2v", "Video")


# ---------------- Tab 2: Image to Video ----------------
with tab2:
    st.subheader("Upload a photo → animate it")
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
        if not valid_token(token):
            st.error("🔑 Pehle sidebar mein apna **HF_TOKEN** paste karein (free token: "
                     "`huggingface.co → Settings → Access Tokens`).")
        elif uploaded is None:
            st.error("🖼️ Pehle koi photo upload karein.")
        elif not motion.strip():
            st.error("✏️ Motion prompt likhein — tasveer mein kya harkat ho?")
        else:
            try:
                client = make_client(token.strip(), provider)
                kwargs2: dict = {
                    "model": i2v_model,
                    "prompt": motion.strip(),
                    "num_frames": frames,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                }
                if negative2.strip():
                    kwargs2["negative_prompt"] = negative2.strip()
                if seed is not None:
                    kwargs2["seed"] = seed

                image_bytes = uploaded.getvalue()
                with st.spinner(
                    "🎞️ Tasveer animate ho rahi hai… chand minute lag sakte hain. "
                    "Page band na karein."
                ):
                    video_bytes = client.image_to_video(image_bytes, **kwargs2)

                fname = f"i2v_{int(time.time())}.mp4"
                st.session_state["i2v_video"] = video_bytes
                st.session_state["i2v_name"] = fname
                st.rerun()
            except Exception as exc:  # never crash the app — explain instead
                st.error(friendly_error(exc))
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


st.divider()
st.caption(
    "Built with Streamlit + `huggingface_hub`. Models: Wan 2.x, HunyuanVideo, Mochi, "
    "LTX-Video, CogVideoX via Hugging Face Inference Providers."
)
