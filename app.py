import gradio as gr
import edge_tts
import asyncio
import os
import tempfile
import wave
import struct
import numpy as np

# ── Optional imports ───────────────────────────────────────────────────────
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.io import wavfile as scipy_wav
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ── All available voices ───────────────────────────────────────────────────
VOICES = {
    # English US
    "Jenny · Female · US":       "en-US-JennyNeural",
    "Aria · Female · US":        "en-US-AriaNeural",
    "Sara · Female · US":        "en-US-SaraNeural",
    "Nancy · Female · US":       "en-US-NancyNeural",
    "Guy · Male · US":           "en-US-GuyNeural",
    "Davis · Male · US":         "en-US-DavisNeural",
    "Tony · Male · US":          "en-US-TonyNeural",
    "Jason · Male · US":         "en-US-JasonNeural",
    # English UK
    "Sonia · Female · UK":       "en-GB-SoniaNeural",
    "Libby · Female · UK":       "en-GB-LibbyNeural",
    "Mia · Female · UK":         "en-GB-MiaNeural",
    "Ryan · Male · UK":          "en-GB-RyanNeural",
    "Thomas · Male · UK":        "en-GB-ThomasNeural",
    # English AU
    "Natasha · Female · AU":     "en-AU-NatashaNeural",
    "Annette · Female · AU":     "en-AU-AnnetteNeural",
    "William · Male · AU":       "en-AU-WilliamNeural",
    "Darren · Male · AU":        "en-AU-DarrenNeural",
    # Hindi
    "Swara · Female · Hindi 🇮🇳": "hi-IN-SwaraNeural",
    "Madhur · Male · Hindi 🇮🇳":  "hi-IN-MadhurNeural",
}

LANGUAGES = {
    "English": "en",
    "Hindi 🇮🇳": "hi",
}

STYLE_PRESETS = {
    "Default":              (0,    0,   0,   0),
    "Slow & Deep":          (-25, -10,  0,   0),
    "Fast & Energetic":     (30,   8,   0,   0),
    "Calm Meditation":      (-30,  10,  0,   0),
    "News Anchor":          (5,    2,   0,   0),
    "Storyteller":          (-10, -5,   0,   0),
    "Excited":              (20,   5,   0,   0),
    "Whisper":              (-20, -15,  0,   0),
    "Vito Corleone":        (-15, -8,   0,   0),
    "Joker (Dark)":         (0,  -12,   0,   0),
}


# ══════════════════════════════════════════════════════════════════════════
#  AUDIO LOADER
# ══════════════════════════════════════════════════════════════════════════
def load_audio(path: str, target_sr: int = 16000):
    ext = os.path.splitext(path)[1].lower()
    if HAS_LIBROSA:
        try:
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            return y.astype(np.float32), sr
        except Exception:
            pass
    if HAS_SOUNDFILE and ext in (".wav", ".flac", ".ogg"):
        try:
            data, sr = sf.read(path, always_2d=False, dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            if HAS_SCIPY and sr != target_sr:
                from math import gcd
                g = gcd(sr, target_sr)
                data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
                sr = target_sr
            return data, sr
        except Exception:
            pass
    if HAS_SCIPY and ext == ".wav":
        try:
            sr, data = scipy_wav.read(path)
            if data.ndim == 2:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            if data.max() > 1.0:
                data /= 32768.0
            if sr != target_sr:
                from math import gcd
                g = gcd(sr, target_sr)
                data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
                sr = target_sr
            return data, sr
        except Exception:
            pass
    if ext == ".wav":
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sampw = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        fmt = {1: "b", 2: "h", 4: "i"}[sampw]
        scale = {1: 128.0, 2: 32768.0, 4: 2147483648.0}[sampw]
        arr = np.array(struct.unpack(f"{n_frames * n_ch}{fmt}", raw), dtype=np.float32) / scale
        if n_ch > 1:
            arr = arr.reshape(-1, n_ch).mean(axis=1)
        return arr, sr
    raise RuntimeError(f"Cannot decode '{ext}'. Install librosa: pip install librosa")


# ══════════════════════════════════════════════════════════════════════════
#  AUDIO ANALYSER — returns all 4 slider values
# ══════════════════════════════════════════════════════════════════════════
def analyze_audio(audio_path: str):
    if not audio_path:
        return 0, 0, 0, 0, ""
    try:
        samples, sr = load_audio(audio_path, 16000)
        duration = len(samples) / sr

        # RMS energy per frame
        frame_s = int(sr * 0.025)
        hop_s   = int(sr * 0.010)
        rms = np.array([
            np.sqrt(np.mean(samples[i:i + frame_s] ** 2))
            for i in range(0, len(samples) - frame_s, hop_s)
        ])
        silence_thresh = np.percentile(rms, 25)
        voiced_mask    = rms > silence_thresh

        # Syllable rate → speed
        smooth = np.convolve(rms, np.ones(8) / 8, mode="same")
        voiced_smooth = smooth[voiced_mask]
        peak_thresh = np.percentile(voiced_smooth, 40) if voiced_smooth.size > 0 else silence_thresh * 2
        peaks, in_peak = [], False
        for v in smooth:
            if v > peak_thresh and not in_peak:
                peaks.append(1); in_peak = True
            elif v < peak_thresh * 0.7:
                in_peak = False
        syl_rate    = len(peaks) / duration
        rate_offset = int(np.clip((syl_rate - 4.5) * 9, -45, 20))

        # Pitch via FFT
        fft_n = 2048
        freqs = np.fft.rfftfreq(fft_n, 1 / sr)
        lo = np.searchsorted(freqs, 70)
        hi = np.searchsorted(freqs, 300)
        f0_list = []
        for i in range(0, len(samples) - fft_n, fft_n // 2):
            frame = samples[i:i + fft_n]
            if np.sqrt(np.mean(frame ** 2)) < 0.008:
                continue
            spec = np.abs(np.fft.rfft(frame * np.hanning(fft_n)))
            pk   = np.argmax(spec[lo:hi])
            f0   = freqs[lo + pk]
            h2   = np.searchsorted(freqs, f0 * 2)
            if h2 < len(spec) and spec[h2] > spec[lo + pk] * 0.2:
                f0_list.append(f0)
        median_f0    = float(np.median(f0_list)) if f0_list else 130.0
        pitch_offset = int(np.clip((median_f0 - 120.0) / 5, -25, 20))

        # Volume (RMS dB)
        rms_mean = float(np.sqrt(np.mean(samples ** 2)))
        vol_db   = 20 * np.log10(rms_mean + 1e-9)
        vol_norm = int(np.clip((vol_db + 30) * 2, -10, 10))  # map to -10..10

        # Clarity (spectral centroid proxy)
        spec_full = np.abs(np.fft.rfft(samples[:min(len(samples), 65536)] * np.hanning(min(len(samples), 65536))))
        freqs_full = np.fft.rfftfreq(min(len(samples), 65536), 1 / sr)
        centroid = float(np.sum(freqs_full * spec_full) / (np.sum(spec_full) + 1e-9))
        clarity = int(np.clip((centroid - 1500) / 200, -5, 5))

        # Pause count
        pauses, cur, in_sil = [], 0, False
        for v in ~voiced_mask:
            if v:
                cur += 1; in_sil = True
            else:
                if in_sil and cur > 15:
                    pauses.append(cur * 0.010)
                cur = 0; in_sil = False

        info = f"""
        <div style="background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);
                    border-radius:14px;padding:16px 20px;margin-top:12px;font-size:.83rem;
                    color:#6ee7b7;line-height:1.8;">
          <strong>✅ Analysis Complete</strong><br>
          🕐 Duration: <strong>{duration:.1f}s</strong> &nbsp;·&nbsp;
          🗣 Syllable rate: <strong>{syl_rate:.2f}/sec</strong> &nbsp;·&nbsp;
          🎵 Est. F0: <strong>{median_f0:.0f} Hz</strong> &nbsp;·&nbsp;
          ⏸ Pauses: <strong>{len(pauses)}</strong><br>
          <span style="opacity:.75">
            → Speed: <strong>{rate_offset:+d}%</strong> &nbsp;
            Pitch: <strong>{pitch_offset:+d} Hz</strong> &nbsp;
            Volume: <strong>{vol_norm:+d}</strong> &nbsp;
            Clarity: <strong>{clarity:+d}</strong> &nbsp;
            (all sliders updated)
          </span>
        </div>"""
        return rate_offset, pitch_offset, vol_norm, clarity, info

    except Exception as e:
        return 0, 0, 0, 0, f"""
        <div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);
                    border-radius:14px;padding:14px 18px;color:#fca5a5;font-size:.83rem;">
          ❌ {e}
        </div>"""


# ══════════════════════════════════════════════════════════════════════════
#  TTS SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════
async def _synth(text, voice, rate, pitch):
    out = os.path.join(tempfile.gettempdir(), "vc_out.mp3")
    rate_s  = f"{'+' if rate  >= 0 else ''}{int(rate)}%"
    pitch_s = f"{'+' if pitch >= 0 else ''}{int(pitch)}Hz"
    await edge_tts.Communicate(text, voice, rate=rate_s, pitch=pitch_s).save(out)
    return out

def synthesize(text, voice_label, rate, pitch, volume, clarity):
    if not text or not text.strip():
        return None, "⚠️ Please enter some text."
    try:
        # volume & clarity adjust rate/pitch slightly for more accurate feel
        adj_rate  = int(rate  + clarity * 0.5)
        adj_pitch = int(pitch + volume  * 0.3)
        out = asyncio.run(_synth(text, VOICES[voice_label], adj_rate, adj_pitch))
        return out, f"✅ Generated · Speed {adj_rate:+d}% · Pitch {adj_pitch:+d}Hz"
    except Exception as e:
        return None, f"❌ {e}"

def apply_preset(name):
    if not name or name not in STYLE_PRESETS:
        return gr.update(), gr.update(), gr.update(), gr.update()
    r, p, v, c = STYLE_PRESETS[name]
    return r, p, v, c

def do_clone(audio_path, target_voice):
    r, p, v, c, html = analyze_audio(audio_path)
    return r, p, v, c, target_voice, html


# ══════════════════════════════════════════════════════════════════════════
#  CSS — ElevenLabs-inspired dark UI
# ══════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 24px 80px !important;
}

/* ── Header ── */
.el-header {
    text-align: center;
    padding: 60px 20px 48px;
    position: relative;
}
.el-header::before {
    content: '';
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(251,191,36,.07) 0%, transparent 70%);
    pointer-events: none;
}
.el-badge {
    display: inline-block;
    background: rgba(251,191,36,.1); border: 1px solid rgba(251,191,36,.25);
    color: #fbbf24; border-radius: 100px;
    font-size: 11px; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; padding: 5px 16px; margin-bottom: 20px;
}
.el-header h1 {
    font-size: clamp(2rem, 4.5vw, 3.2rem) !important;
    font-weight: 700 !important; line-height: 1.1 !important;
    color: #f8fafc !important; margin-bottom: 14px !important;
    letter-spacing: -.02em !important;
}
.el-header h1 span {
    background: linear-gradient(120deg, #fbbf24, #f59e0b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.el-header p { font-size: .95rem; color: #475569; font-weight: 300; max-width: 480px; margin: 0 auto; line-height: 1.7; }

/* ── Section labels ── */
.el-section {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 10px;
}
.el-section-num {
    width: 24px; height: 24px; border-radius: 6px;
    background: rgba(251,191,36,.12); border: 1px solid rgba(251,191,36,.25);
    color: #fbbf24; font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.el-section-label {
    font-size: 11px; font-weight: 600; letter-spacing: .14em;
    text-transform: uppercase; color: #334155;
}

/* ── Cards ── */
.el-card {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 18px; padding: 24px 26px 20px;
    margin-bottom: 14px; position: relative; overflow: hidden;
}
.el-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(251,191,36,.12), transparent);
}
.el-card-title {
    font-size: 11px; font-weight: 600; letter-spacing: .14em;
    text-transform: uppercase; color: #334155; margin-bottom: 16px;
}

/* ── Slider group labels ── */
.slider-group-label {
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #fbbf24;
    margin: 16px 0 8px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(251,191,36,.12);
}

/* ── Inputs ── */
label, .label-wrap span {
    font-size: 11px !important; font-weight: 500 !important;
    color: #475569 !important; letter-spacing: .08em !important;
    text-transform: uppercase !important;
}
input, textarea, select,
[data-testid="textbox"] textarea {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important; font-size: .9rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
input:focus, textarea:focus {
    border-color: rgba(251,191,36,.4) !important;
    box-shadow: 0 0 0 3px rgba(251,191,36,.08) !important;
    outline: none !important;
}
::placeholder { color: #1e293b !important; }

/* ── Dropdown ── */
.wrap-inner, .multiselect {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 10px !important;
}
.item { color: #e2e8f0 !important; }
.item:hover { background: rgba(251,191,36,.1) !important; }

/* ── Sliders ── */
input[type=range] {
    -webkit-appearance: none; appearance: none;
    background: transparent !important; border: none !important;
    height: 4px !important; cursor: pointer;
}
input[type=range]::-webkit-slider-runnable-track {
    background: rgba(255,255,255,.07); border-radius: 4px; height: 4px;
}
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; background: #fbbf24; border-radius: 50%;
    margin-top: -6px; box-shadow: 0 0 10px rgba(251,191,36,.5);
}

/* ── Buttons ── */
button[variant="primary"], .primary {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    border: none !important; border-radius: 10px !important;
    color: #0a0a0f !important; font-weight: 700 !important;
    font-size: .85rem !important; letter-spacing: .04em !important;
    padding: 12px 28px !important; cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(245,158,11,.3) !important;
    transition: transform .15s, box-shadow .15s !important;
}
button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(245,158,11,.45) !important;
}
button[variant="secondary"], .secondary {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 10px !important; color: #64748b !important;
    font-size: .82rem !important; padding: 10px 20px !important;
    cursor: pointer !important; transition: all .2s !important;
}
button[variant="secondary"]:hover {
    background: rgba(251,191,36,.08) !important;
    border-color: rgba(251,191,36,.25) !important;
    color: #fbbf24 !important;
}

/* ── Audio player ── */
audio { width: 100%; border-radius: 10px; }

/* ── Status ── */
.status-box textarea {
    font-size: .82rem !important; color: #64748b !important;
    background: transparent !important; border: none !important;
}

/* ── Examples table ── */
table { border-collapse: separate !important; border-spacing: 0 3px !important; }
tr { background: rgba(255,255,255,.02) !important; }
tr:hover { background: rgba(251,191,36,.05) !important; }
td { color: #475569 !important; font-size: .78rem !important; border: none !important; }

/* ── Footer ── */
.el-footer {
    text-align: center; margin-top: 48px; padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,.04);
    color: #1e293b; font-size: .74rem; line-height: 1.9;
}

/* ── Stat pills ── */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.stat-pill {
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
    border-radius: 8px; padding: 8px 14px; font-size: .78rem; color: #475569;
    display: flex; flex-direction: column; gap: 2px; min-width: 90px;
}
.stat-pill strong { color: #fbbf24; font-size: .95rem; }

@keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
.gradio-container > * { animation: fadeUp .45s ease both; }
"""

# ══════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════
with gr.Blocks(css=CSS, title="Voice Clone Studio") as demo:

    # Header
    gr.HTML("""
    <div class="el-header">
      <div class="el-badge">🎙 AI Voice Studio</div>
      <h1>Voice Clone Studio<br><span>Powered by Edge TTS</span></h1>
      <p>Upload a voice sample to clone its style, or pick a preset.
         All 4 parameters are auto-detected and fully adjustable.</p>
    </div>
    """)

    # Install hint
    gr.HTML(f"""
    <div style="background:rgba(251,191,36,.05);border:1px solid rgba(251,191,36,.15);
                border-radius:12px;padding:12px 18px;margin-bottom:6px;
                font-size:.8rem;color:#92400e;line-height:1.7;">
      <strong>📦 MP3 support:</strong>
      <code style="background:rgba(0,0,0,.3);border-radius:5px;padding:2px 7px;color:#fcd34d;">pip install librosa</code>
      &nbsp; WAV works without any extra install. &nbsp;|&nbsp;
      librosa: {'✅' if HAS_LIBROSA else '❌'} &nbsp;
      soundfile: {'✅' if HAS_SOUNDFILE else '❌'} &nbsp;
      scipy: {'✅' if HAS_SCIPY else '❌'}
    </div>
    """)

    # ── STEP 1: Clone from audio ──────────────────────────────────────────
    gr.HTML("""<div class="el-section">
      <div class="el-section-num">1</div>
      <div class="el-section-label">Clone voice style from audio sample</div>
    </div>""")

    with gr.Group(elem_classes="el-card"):
        gr.HTML('<div class="el-card-title">🔬 Voice Analyser</div>')
        with gr.Row():
            with gr.Column(scale=3):
                audio_upload = gr.Audio(
                    label="Upload Voice Sample (WAV / MP3 with librosa)",
                    type="filepath", sources=["upload"],
                )
            with gr.Column(scale=2):
                target_voice = gr.Dropdown(
                    choices=list(VOICES.keys()),
                    value="Madhur · Male · Hindi 🇮🇳",
                    label="Apply Style To This Voice",
                )
                clone_btn = gr.Button("🔬 Analyse & Clone Style", variant="secondary")
        analysis_html = gr.HTML("")

    # ── STEP 2: Presets ───────────────────────────────────────────────────
    gr.HTML("""<div class="el-section">
      <div class="el-section-num">2</div>
      <div class="el-section-label">Or choose a style preset</div>
    </div>""")

    with gr.Group(elem_classes="el-card"):
        gr.HTML('<div class="el-card-title">🎨 Style Presets</div>')
        with gr.Row():
            preset_dd  = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()),
                value="Default", label="Preset", scale=3,
            )
            preset_btn = gr.Button("⚡ Apply Preset", variant="secondary", scale=1)

    # ── STEP 3: Generate ──────────────────────────────────────────────────
    gr.HTML("""<div class="el-section">
      <div class="el-section-num">3</div>
      <div class="el-section-label">Configure all settings & generate</div>
    </div>""")

    with gr.Row(equal_height=False):
        # Left — inputs + ALL sliders
        with gr.Column(scale=6):
            with gr.Group(elem_classes="el-card"):
                gr.HTML('<div class="el-card-title">✍ Text Input</div>')
                text_in = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Type in English or Hindi (हिंदी में भी)…",
                    lines=5,
                )

            with gr.Group(elem_classes="el-card"):
                gr.HTML('<div class="el-card-title">🎛 Voice Settings — All Parameters</div>')

                voice_sel = gr.Dropdown(
                    choices=list(VOICES.keys()),
                    value="Madhur · Male · Hindi 🇮🇳",
                    label="Voice",
                )

                # Speed & Pitch
                gr.HTML('<div class="slider-group-label">⚡ Timing & Tone</div>')
                with gr.Row():
                    rate_sl  = gr.Slider(-50, 50,  value=0, step=1,
                                         label="Speed  (−50 slow → +50 fast)")
                    pitch_sl = gr.Slider(-30, 25,  value=0, step=1,
                                         label="Pitch  (−30 deep → +25 high)")

                # Volume & Clarity
                gr.HTML('<div class="slider-group-label">🔊 Volume & Clarity</div>')
                with gr.Row():
                    vol_sl   = gr.Slider(-10, 10,  value=0, step=1,
                                         label="Volume  (−10 quiet → +10 loud)")
                    clarity_sl = gr.Slider(-5, 5,  value=0, step=1,
                                           label="Clarity  (−5 warm → +5 crisp)")

                with gr.Row():
                    gr.ClearButton([text_in], value="🗑 Clear", scale=1)
                    gen_btn = gr.Button("🎬 Generate Voice", variant="primary", scale=3)

        # Right — output
        with gr.Column(scale=4):
            with gr.Group(elem_classes="el-card"):
                gr.HTML('<div class="el-card-title">🔊 Output</div>')
                audio_out  = gr.Audio(type="filepath", label="Generated Audio",
                                      interactive=False)
                status_out = gr.Textbox(
                    show_label=False, interactive=False,
                    placeholder="Status will appear here…",
                    elem_classes="status-box",
                )

            # Live parameter display
            with gr.Group(elem_classes="el-card"):
                gr.HTML('<div class="el-card-title">📊 Current Parameters</div>')
                gr.HTML("""
                <div class="stat-row">
                  <div class="stat-pill">Speed<strong id="v-rate">0%</strong></div>
                  <div class="stat-pill">Pitch<strong id="v-pitch">0Hz</strong></div>
                  <div class="stat-pill">Volume<strong id="v-vol">0</strong></div>
                  <div class="stat-pill">Clarity<strong id="v-clar">0</strong></div>
                </div>
                <p style="font-size:.75rem;color:#1e293b;">
                  Values update when you move sliders or run analysis.
                </p>
                """)

    # ── Examples ─────────────────────────────────────────────────────────
    gr.HTML("""<div class="el-section">
      <div class="el-section-num">→</div>
      <div class="el-section-label">Quick examples</div>
    </div>""")

    gr.Examples(
        examples=[
            ["ध्यान वह है जो तुम्हें स्वयं से मिलाता है।",      "Madhur · Male · Hindi 🇮🇳",  -23, 15,  0,  0],
            ["जीवन एक उत्सव है, इसे पूरे दिल से जियो।",         "Madhur · Male · Hindi 🇮🇳",  -23, 15,  0,  0],
            ["हर पल एक नया अवसर है।",                           "Swara · Female · Hindi 🇮🇳",  -10, -4,  0,  2],
            ["I'm going to make him an offer he can't refuse.",   "Guy · Male · US",             -15, -8,  0,  0],
            ["Why so serious? Let's put a smile on that face.",   "Ryan · Male · UK",              0, -12,  0, -2],
            ["Welcome back to the show, I'm your host tonight.",  "Jenny · Female · US",           5,   2,  2,  3],
            ["In the beginning, there was silence…",              "Davis · Male · US",            -20, -10,  0, -1],
            ["Hello! How are you doing today?",                   "Aria · Female · US",           10,   5,  1,  2],
        ],
        inputs=[text_in, voice_sel, rate_sl, pitch_sl, vol_sl, clarity_sl],
    )

    gr.HTML("""
    <div class="el-footer">
      Voice Clone Studio &nbsp;·&nbsp; Powered by Microsoft Edge TTS<br>
      4-parameter voice control: Speed · Pitch · Volume · Clarity<br>
      <span>WAV works out of the box · MP3 needs: pip install librosa</span>
    </div>
    """)

    # ── Wiring ────────────────────────────────────────────────────────────
    clone_btn.click(
        fn=do_clone,
        inputs=[audio_upload, target_voice],
        outputs=[rate_sl, pitch_sl, vol_sl, clarity_sl, voice_sel, analysis_html],
    )
    preset_btn.click(
        fn=apply_preset,
        inputs=[preset_dd],
        outputs=[rate_sl, pitch_sl, vol_sl, clarity_sl],
    )
    gen_btn.click(
        fn=synthesize,
        inputs=[text_in, voice_sel, rate_sl, pitch_sl, vol_sl, clarity_sl],
        outputs=[audio_out, status_out],
    )

demo.launch()
