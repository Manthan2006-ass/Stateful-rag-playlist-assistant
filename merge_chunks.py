import os
import json

# Ensure input and output workspace directories are configured
os.makedirs("newjsons", exist_ok=True)

if not os.path.exists("jsons"):
    print("❌ Error: 'jsons' folder not found. Please run mp3tojson.py first!")
    exit()

json_files = [f for f in os.listdir("jsons") if f.endswith(".json")]

if not json_files:
    print("📁 The 'jsons' folder is empty. Wait for mp3tojson.py to output files.")
    exit()

# Semantic Chunking Strategy Configuration Parameters
TARGET_WORDS = 250   # Optimal text window length for dense vector models
OVERLAP_WORDS = 50   # Sliding window buffer size to prevent context loss

for json_file in json_files:
    try:
        input_path = os.path.join("jsons", json_file)
        output_path = os.path.join("newjsons", json_file)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_segments = data.get("chunks", [])
        if not raw_segments:
            print(f"⏩ Skipping empty transcript structure: {json_file}")
            continue

        print(f"🧩 Semantically Chunking Timeline Boundaries: {json_file}")

        merged_chunks = []
        current_words = []
        
        # Track active window segment markers
        window_start_time = raw_segments[0]["start"]
        video_number = raw_segments[0]["number"]
        video_title = raw_segments[0]["title"]

        i = 0
        while i < len(raw_segments):
            segment = raw_segments[i]
            segment_text = segment["text"]
            segment_words = segment_text.split()

            # Append words into the current working window array
            current_words.extend(segment_words)
            window_end_time = segment["end"]  # Dynamically push the window end timestamp forward

            # Evaluate if the chunk has reached the production semantic size threshold
            if len(current_words) >= TARGET_WORDS or i == len(raw_segments) - 1:
                chunk_text = " ".join(current_words)
                
                merged_chunks.append({
                    "number": video_number,
                    "title": video_title,
                    "start": float(window_start_time),
                    "end": float(window_end_time),
                    "text": chunk_text.strip()
                })

                # Calculate the index regression to implement the sliding window overlap boundary
                # Look backwards through segments to see where the last 50 words roughly started
                overlap_word_count = 0
                backtrack_index = i
                
                while backtrack_index >= 0 and overlap_word_count < OVERLAP_WORDS:
                    overlap_word_count += len(raw_segments[backtrack_index]["text"].split())
                    backtrack_index -= 1
                
                # Advance step index to cleanly initiate the next forward window
                i = max(i + 1, backtrack_index + 2)
                
                # Reset active arrays and capture the starting timestamp of the new boundary group
                current_words = []
                if i < len(raw_segments):
                    window_start_time = raw_segments[i]["start"]
            else:
                i += 1

        # Package payload data structure conforming to downstream database indexing requirements
        output_payload = {
            "chunks": merged_chunks
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)

        print(f"✅ Generated {len(merged_chunks)} context-optimized chunks ──> {output_path}")

    except Exception as e:
        print(f"❌ Error encountered while partitioning file {json_file}: {str(e)}")

print("\n🎉 Phase Complete: Sliding window semantic layers exported inside 'newjsons'!")