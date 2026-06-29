"""
Sintetizador eleven_v3. La entonación se controla con audio tags inline
([questioning]/[excited]) + el signo final; el texto hablado es el mismo en las 3
entonaciones. v3 no usa stability/style/speed numéricos: stability es categórica
(0.0=Creative, 0.5=Natural, 1.0=Robust) y la variabilidad la aporta el propio modelo.
"""

import os
import time
import json
import wave
import hashlib
import argparse
from typing import Iterator
import pandas as pd
import numpy as np
from tqdm import tqdm
from elevenlabs.client import ElevenLabs
from elevenlabs.play import save
from dotenv import load_dotenv
from itertools import product
from typing import Iterator

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEFAULT_MODEL = "eleven_v3"
STABILITY = 0.0  # v3: 0.0=Creative (más expresivo), 0.5=Natural, 1.0=Robust
OUTPUT_FORMAT = "pcm_16000"  # PCM crudo -> WAV real sin costo extra (wav_44100/pcm_44100 requieren Pro).
SAMPLE_RATE = 16000
IGNORE_VOICES = set()  # no se descarta ninguna voz
PROMPTS = {
    'statement':   ("",               ".",   False),
    'question':    ("[questioning] ", "?",   False),
    'exclamation': ("[excited] ",     "!!!", True),
}
MIN_DURATION = 0.4   # s; por debajo, probable truncado
MIN_RMS = 0.005      # por debajo, probable silencio


def log_fail(outdir, voice_id, intonation, stem, reason):
    """Registra un fallo/sospecha sin cortar la corrida."""
    with open(os.path.join(outdir, "failures.log"), "a", encoding="utf-8") as f:
        f.write(f"{voice_id}\t{intonation}\t{stem}\t{reason}\n")


def main(args):
    if not ELEVENLABS_API_KEY:
        raise SystemExit("Error: ELEVENLABS_API_KEY should be defined.")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # sentences: una entrada por oración (declarativa). Las 3 entonaciones se generan acá.
    sentences_path = args.sentences
    if not os.path.exists(sentences_path):
        raise SystemExit(f"Error: not found {sentences_path}")

    with open(sentences_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # cada oración -> su stem (sin signo final); tolera que venga con o sin ./?/!
    sentences = [ln.rstrip(" .?!") for ln in lines if ln.strip()]
    if not sentences:
        raise SystemExit("Empty sentences file.")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # voices
    if os.path.exists(os.path.join(outdir, "voices.csv")):
        voices_df = pd.read_csv(os.path.join(outdir, "voices.csv"))
    else:
        response = client.voices.search(page_size=100, voice_type='default')
        voices = []
        for voice in response.voices: # 19 sin cuenta premium
            if voice.voice_id in IGNORE_VOICES:
                continue
            has_english = any(d.language == "en" for d in voice.verified_languages)
            if not has_english:  # v3 no figura en high_quality_base_model_ids; filtramos solo por inglés
                continue
            voice_data = {
                'name' : voice.name,
                'voice_id' : voice.voice_id,
                'category' : voice.category,
                'description' : voice.description,
                
            }
            voice_data = voice_data | voice.labels
            voices.append(voice_data)
        voices_df = pd.DataFrame(voices)
        voices_df.to_csv(os.path.join(outdir, "voices.csv"), index=False)
    
    # for each voice
    pbar = tqdm(total=len(voices_df) * len(sentences) * len(PROMPTS), desc="synth", unit="wav")
    for voice in voices_df.itertuples():
        safe_voice_name = f"{voice.voice_id}"
        voice_outdir = os.path.join(outdir, safe_voice_name)
        os.makedirs(voice_outdir, exist_ok=True)

        # for each sentence -> las 3 entonaciones
        for stem in sentences:
            text = stem.replace(" ", "-").replace("'", "").replace("’", "")  # nombre de archivo
            for intonation, (prefix, sign, caps) in PROMPTS.items():
                pbar.update(1)
                spoken = stem.upper() if caps else stem

                # Formato: voice_intonation_text
                base = f"{safe_voice_name}_{intonation}_{text}"
                out_file = os.path.join(voice_outdir, f"{base}.wav")
                if os.path.exists(out_file):
                    tqdm.write(f"Already exists {out_file}, skipping.")
                    continue

                # synth (un fallo loguea y sigue; no corta la corrida)
                try:
                    audio = client.text_to_speech.convert(
                        text=prefix + spoken + sign,
                        voice_id=voice.voice_id,
                        model_id=DEFAULT_MODEL,
                        voice_settings={"stability": STABILITY},
                        output_format=OUTPUT_FORMAT,
                    )
                    if isinstance(audio, Iterator):
                        audio = b"".join(audio)
                except Exception as e:
                    log_fail(outdir, voice.voice_id, intonation, stem, f"api_error: {e}")
                    tqdm.write(f"FAIL (api) {base}: {e}")
                    continue

                if len(audio) < 128:
                    log_fail(outdir, voice.voice_id, intonation, stem, "empty_audio")
                    tqdm.write(f"FAIL (empty) {base}")
                    continue

                # chequeos baratos de calidad (no bloqueantes, solo log)
                samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                dur = len(samples) / SAMPLE_RATE
                rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
                if dur < MIN_DURATION:
                    log_fail(outdir, voice.voice_id, intonation, stem, f"too_short: {dur:.2f}s")
                if rms < MIN_RMS:
                    log_fail(outdir, voice.voice_id, intonation, stem, f"silent: rms={rms:.4f}")

            # save (PCM crudo -> WAV con header)
                with wave.open(out_file, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)  # PCM 16-bit
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(audio)
                tqdm.write(f"Saved {out_file}")
                time.sleep(0.5)  # avoid throttling
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesizer using ElevenLabs API for multiple voices.")
    parser.add_argument("--sentences", "-s", required=True, help="Sentences file (one per line).")
    parser.add_argument("--outdir", "-o", default="outputs", help="Output directory.")
    args = parser.parse_args()
    main(args)
