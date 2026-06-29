# Synthetic Intonation Dataset (SynthID)

Generates an English speech dataset for prosody studies: for each sentence, **three intonations**
(statement `.`, question `?`, exclamation `!`) are synthesized across multiple voices, where the
**spoken text is identical** across the three. Intonation is controlled with `eleven_v3` audio tags
(`[questioning]`, `[excited]`) plus the final punctuation, without changing the words.

## Requirements

- Python 3.9+
- An ElevenLabs account on the **Creator** plan (the full dataset costs ~85k credits).
- `pip install -r requirements.txt`

## Steps to replicate the generation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API key.** In ElevenLabs: *Developers → API Keys → Create Key*, with the **Text to Speech**
   and **Voices → Read** permissions. Store it in a `.env` file at the repo root:
   ```
   ELEVENLABS_API_KEY=your_key
   ```

3. **(Optional) Probe the voices.** Verify the default voices respond with `eleven_v3` before the
   full run:
   ```bash
   python probe_voices.py
   ```
   If any voice fails, add its `voice_id` to `IGNORE_VOICES` in `main.py`.

4. **Sentences.** `sentences.txt` holds one sentence per line, each stem repeated three times ending
   in `.`, `?`, and `!`. All in declarative form (the script adds the tags/punctuation). To add
   sentences, include their three forms; keep the format.

5. **Generate.**
   ```bash
   bash build_dataset.sh
   ```

## Output

```
dataset/
  voices.csv                              # metadata of the voices used
  <voice_id>/
    <voice_id>_<intonation>_<text>.wav    # mono 16 kHz WAV
  failures.log                            # voice/sentence that failed or looked suspicious (if any)
```

- **Audio format:** WAV PCM 16-bit, mono, 16 kHz.
- **Resuming:** filenames are deterministic. If the run is interrupted or you add sentences, just run
  `bash build_dataset.sh` again: only the missing samples are generated, without re-spending credits
  on existing ones.
- **Model/config:** `eleven_v3`, `stability=0.0` (Creative). Editable at the top of `main.py`.

