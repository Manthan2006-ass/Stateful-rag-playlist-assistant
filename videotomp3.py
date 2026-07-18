import os
import subprocess

# Ensure the audios directory exists
os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")

for file in files:
    # Skip system files like .DS_Store or non-mp4 files
    if not file.endswith(".mp4"):
        continue
        
    # Split by the standard spacing structure
    filename_parts = file.split(" - ")
    
    # Grab the index number (e.g., "01")
    video_number = filename_parts[0].strip()
    
    # Clean the title: remove the .mp4 extension and swap pipes "|" for dashes "-"
    raw_title = " - ".join(filename_parts[1:]).replace(".mp4", "")
    clean_title = raw_title.replace(" | ", " - ").replace("|", "-").strip()
    
    # Construct a clean, safe output name (e.g., "01 - GenAI Roadmap for Beginners - End-to-End GenAI Course 2025 - CampusX.mp3")
    output_filename = f"{video_number} - {clean_title}"
    
    print(f"🎬 Extracting Audio: {file} ──> {output_filename}.mp3")
    
    # Execute ffmpeg command securely
    subprocess.run([
        "ffmpeg", 
        "-i", f"videos/{file}", 
        f"audios/{output_filename}.mp3"
    ])

print("\n🎉 All audio layers extracted successfully inside the 'audios' directory!")