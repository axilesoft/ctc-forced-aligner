import json
import time
import soundfile as sf
import sounddevice as sd
import threading
import sys
import os

audio_path = sys.argv[1] if len(sys.argv)>1 else "e1.wav"
json_path = os.path.splitext(audio_path)[0] + ".json"
# Load the JSON file containing the timeline
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract the timeline from the JSON data
timeline = data["segments"]

# Load the audio file using soundfile
audio_file = audio_path
audio_data, sample_rate = sf.read(audio_file)

# Function to play audio
def play_audio():
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait until the audio is finished playing

# Start playing the audio in a separate thread
audio_thread = threading.Thread(target=play_audio) 
audio_thread.start()

# Record the start time right after starting the audio
start_time = time.time()

# Go through the timeline and print the corresponding text at the right time

for segment in timeline:
    for word in segment["words"]:
        segment_start = word["start"]
        
        # Wait until the current time matches the segment start time
        while time.time() - start_time < segment_start:
            time.sleep(0.01)  # Small delay to prevent busy waiting
        
        print(word["text"])

# Wait for the audio thread to finish before exiting the script
audio_thread.join()
