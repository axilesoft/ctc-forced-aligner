import json

def format_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return f"{hours:d}:{minutes:02d}:{seconds:05.2f}"

def results_to_ssa(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("[Script Info]\nScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,100,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for sentence in results:
            if sentence['words']:
                line_start = sentence['words'][0]['start']
                line_end = sentence['words'][-1]['end']
                line_text = ""
                for i, word in enumerate(sentence['words']):
                    if i < len(sentence['words']) - 1:
                        next_word = sentence['words'][i + 1]
                        duration = next_word['start'] - word['start']
                    else:
                        duration = word['end'] - word['start']
                    line_text += f"{{\\k{int(duration * 100):d}}}{word['text']}"
                
                f.write(f"Dialogue: 0,{format_time(line_start)},{format_time(line_end)},Default,,0,0,0,,{line_text}\n")

def main():
    input_file = "output.json"
    output_file = "output.ass"

    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert and write to ASS file
    results_to_ssa(data['segments'], output_file)
    print(f"Converted {input_file} to ASS format. Output saved as {output_file}")

if __name__ == "__main__":
    main()