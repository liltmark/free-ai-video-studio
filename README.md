---
title: Free AI Video Studio
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
short_description: Free Text-to-Video & Image-to-Video with open-source AI models (Wan, HunyuanVideo, Mochi, LTX-Video, CogVideoX)
---

# 🎬 Free AI Video Studio

Generate AI videos right in your browser — **Text to Video** and **Image to Video** — using open-source models (Wan 2.x, HunyuanVideo, Mochi, LTX-Video, CogVideoX) through the Hugging Face Inference Providers API.

## How it works

1. Pehle ek chhota sa **ad** dekhein (10 second) — isi se ye tool free rehta hai.
2. Apna **free** Hugging Face Access Token (`HF_TOKEN`) sidebar mein paste karein.
3. Model + settings chunein.
4. **Tab 1:** prompt likhein → MP4 payein. **Tab 2:** photo upload + motion prompt → animated MP4. **Tab 3:** surveys ki list.
5. Built-in player mein dekhein aur **Download MP4** dabayein.

## Ads & Surveys (owner settings)

- **Ad gate:** tool khulne se pehle har visitor ko ek ad dekhna lazmi hai (10 second).
  Apne ad network ka `<script>`/`<iframe>` code `app.py` mein sab se upar `AD_HTML`
  mein lagta hai — code jab milega, Muse laga dega.
- **Surveys:** `app.py` mein sab se upar `SURVEY_LINKS` list mein links hain.
  Naye survey links jab bhi hon, Muse ko bhej dein — woh add karke Space update kar dega.

## Free-tier reality check (please read)

- **Space hosting:** free (CPU basic tier). The Space sleeps after 48h of inactivity and wakes automatically when you open it.
- **HF token:** free.
- **Video generation:** billed against your Hugging Face account's **Inference Providers credits**. The free tier includes only a **small monthly allowance** — enough for a few short test clips, **not** unlimited generation. The app tells you clearly if credits run out.
- **Speed:** expect several minutes per clip; queues get busy at peak times and the app handles that gracefully (no crashes, just a friendly "busy, try again" message).

## Get your FREE Hugging Face token

1. Go to [huggingface.co](https://huggingface.co) and create a free account (top-right **Sign Up**, confirm your email).
2. Click your profile picture (top-right) → **Settings** → **Access Tokens** (left sidebar).
3. Click **Create new token** → choose type **Fine-grained**.
4. Give it a name like `video-studio`, and under permissions enable **`Make calls to Inference Providers`**.
5. Click **Create token** → **copy it immediately** (you won't see it again).
6. Paste it into the **sidebar field** of this app. Done — the token is only kept in your browser session, never saved anywhere.

## Deploy this app yourself (1-2-3)

1. **Create the Space:** [huggingface.co](https://huggingface.co) → **+** (top-right) → **New Space** → name it (e.g. `free-ai-video-studio`), keep License `apache-2.0`, select SDK **`Streamlit`**, hardware **CPU basic (FREE)**, visibility Public → **Create Space**.
2. **Upload these 3 files:** open the **Files** tab → **Add file → Upload files** → upload `app.py`, `requirements.txt`, and this `README.md` (the YAML block at the top auto-configures the Space) → **Commit**.
3. **Wait ~2 minutes** for the build, then open the **App** tab — your studio is live at `https://<your-username>-free-ai-video-studio.hf.space`. Paste your token in the sidebar and generate!

## Files

| File | Purpose |
|---|---|
| `app.py` | The full Streamlit app (UI, both tabs, error handling) |
| `requirements.txt` | Python dependencies (lightweight — no torch needed) |
| `README.md` | This file — its YAML header configures the HF Space |
