import json

# Load the JSON file
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# SSA/ASS Header
header = '''[Script Info]
Title: Karaoke Lyrics
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,1.5,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''

# Helper function to convert time to SSA/ASS format
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 100)
    return f"{hours:01d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:02d}"

# Convert JSON to SSA/ASS with karaoke effects
output = header
for segment in data['segments']:
    start_time = segment['words'][0]['start']
    end_time = segment['words'][-1]['end']
    
    text = ""
    last_end_time = start_time
    for word in segment['words']:
        duration = int((word['end'] - word['start']) * 100)  # Convert to centiseconds
        text += f"{{\\k{duration}}}{word['text']}"
        last_end_time = word['end']
    
    start_time_str = format_time(start_time)
    end_time_str = format_time(end_time)
    
    output += f"Dialogue: 0,{start_time_str},{end_time_str},Default,,0,0,0,,{text}\n"

# Save to SSA/ASS file
output_file = 'output1.ass'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"SSA/ASS file saved to {output_file}")
