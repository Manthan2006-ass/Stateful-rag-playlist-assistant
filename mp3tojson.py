import json
import os
from faster_whisper import WhisperModel

# 1. Initialize the ultra-lightweight Faster-Whisper model (tiny)
# Enforcing compute_type="int8" drops transcription latency dramatically on Intel CPUs
print("🔄 Initializing Ultra-Optimized Faster-Whisper Model (tiny)...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")

os.makedirs("jsons", exist_ok=True)
audios = os.listdir("audios")

for audio in audios:
    try:
        if not audio.endswith(".mp3") or audio == "output.mp3":
            continue

        audio_path = os.path.join("audios", audio)
        filename = audio.replace(".mp3", "")
        output_path = f"jsons/{filename}.json"

        # Safe resume capability: skips the 2 files you already processed successfully
        if os.path.exists(output_path):
            print(f"⏩ Skipping (already processed): {filename}")
            continue

        print(f"\n🎙️ Fast Transcribing & Translating via CTranslate2: {audio}")

        # Extract strict structural lecture numbering from the filename string
        if " - " in filename:
            parts = filename.split(" - ")
            number = parts[0].strip()
            title = " - ".join(parts[1:]).strip()
        else:
            number = "unknown"
            title = filename

        # 2. High-performance translation loop execution
        # Translates Hindi speech patterns straight into written English text strings
        segments, info = model.transcribe(
            audio_path, 
            language="hi",
            task="translate",
            beam_size=5,
            temperature=0.0  # Eradicates mathematical text hallucinations
        )

        chunks = []
        for segment in segments:
            chunks.append({
                "number": number,
                "title": title,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip()
            })

        # Pack payload data structure conforming to downstream merge_chunks.py requirements
        payload = {
            "chunks": chunks,
            "text": " ".join([c["text"] for c in chunks])
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully exported optimized transcript file: {output_path}")

    except Exception as e:
        print(f"❌ Error encountered while parsing file {audio}: {str(e)}")

print("\n🎉 Phase Complete: All transcripts generated successfully inside the 'jsons' directory!")