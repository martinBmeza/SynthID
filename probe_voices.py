"""Probe: 1 generación v3 corta por voz default (inglés) para detectar voces que
fallan con eleven_v3 ANTES del run grande. Chequea: error de API, audio vacío,
duración mínima y silencio. Gasta ~pocos cientos de créditos.
Salida: imprime PASS/FAIL por voz y la lista de voice_id a descartar."""
import os, wave
from typing import Iterator
import numpy as np
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

MODEL = "eleven_v3"
SR = 16000
MIN_DURATION = 0.4
MIN_RMS = 0.005
TEXT = "[excited] THIS IS A TEST!!!"  # usa tag+caps como el run real
OUTDIR = "probe_out"
os.makedirs(OUTDIR, exist_ok=True)

resp = client.voices.search(page_size=100, voice_type="default")
voices = [v for v in resp.voices if any(d.language == "en" for d in (v.verified_languages or []))]
print(f"{len(voices)} voces inglés a probar con {MODEL}\n")

failed = []
for v in voices:
    reason = None
    try:
        audio = client.text_to_speech.convert(
            text=TEXT, voice_id=v.voice_id, model_id=MODEL,
            voice_settings={"stability": 0.0}, output_format="pcm_16000",
        )
        if isinstance(audio, Iterator):
            audio = b"".join(audio)
        if len(audio) < 128:
            reason = "empty_audio"
        else:
            s = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            dur, rms = len(s) / SR, float(np.sqrt(np.mean(s**2)))
            if dur < MIN_DURATION:
                reason = f"too_short {dur:.2f}s"
            elif rms < MIN_RMS:
                reason = f"silent rms={rms:.4f}"
            else:
                with wave.open(os.path.join(OUTDIR, f"{v.voice_id}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(audio)
    except Exception as e:
        reason = f"api_error: {e}"

    status = "PASS" if reason is None else f"FAIL ({reason})"
    print(f"{status:20s} {v.name[:35]:35s} {v.voice_id}")
    if reason:
        failed.append(v.voice_id)

print("\n" + "=" * 50)
if failed:
    print(f"{len(failed)} voces fallaron. Agregalas a IGNORE_VOICES en main.py:")
    print("IGNORE_VOICES = {", ", ".join(f'\"{x}\"' for x in failed), "}")
else:
    print("Todas las voces PASARON con v3. Verde para el run.")
print("Revisá los wav en probe_out/ para confirmar que suenan bien.")
