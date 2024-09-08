import torch
cudaver= torch.version
device = "cuda" if torch.cuda.is_available() else "cpu"

import json
import sys
import os
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

def format_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return f"{hours:d}:{minutes:02d}:{seconds:05.2f}"

# Add this new function to convert results to SSA/ASS format
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
                prev_end = line_start
                for word in sentence['words']:
                    # Calculate duration including the interval before the word
                    duration = word['end'] - prev_end
                    line_text += f"{{\\k{int(duration * 100):d}}}{word['text']}"
                    prev_end = word['end']
                
                f.write(f"Dialogue: 0,{format_time(line_start)},{format_time(line_end)},Default,,0,0,0,,{line_text}\n")

audio_path = "ce.wav"
text_path = "ce.txt"
output_file = "output.json"
output_ass = "output.ass"
output_lab = "output.lab"
language = "chi" #"iso" # ISO-639-3 Language code

if len(sys.argv) > 2:
    language = sys.argv[1]
    audio_path = sys.argv[2]
    fn = os.path.splitext(audio_path)[0]
    text_path = fn + ".txt"
    output_file = fn + ".json"
    output_ass = fn + ".ass"
    output_lab = fn + ".lab"
isChar = True
if language in ["jpn", "chi"]:
    isChar= True


batch_size = 16


print("Loading model...")
alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
    device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)
print(f"Loading audio {audio_path} ...")
audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)


with open(text_path, "r") as f:
    lines = f.readlines()
retrp =  "¯" if isChar else " ¯ "
text = "".join(line for line in lines).replace("\n",retrp).strip()

print("generate_emissions...")
emissions, stride = generate_emissions(
    alignment_model, audio_waveform, batch_size=batch_size
)

print("preprocess_text...")
tokens_starred, text_starred = preprocess_text(
    text,
    romanize=True,
    language=language,
    split_size= "char" if isChar else "word"
)

print("get_alignments...")
segments, scores, blank_id = get_alignments(
    emissions,
    tokens_starred,
    alignment_dictionary,
)

spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))
print("============= SPAN =============")
word_timestamps, labstr = postprocess_results(text_starred, spans, stride, scores)
jsonRoot = {
    "text":text,
    "segments":word_timestamps,
}

with open(output_file, "w") as f:
    json.dump(jsonRoot, f, indent=2)

#write labstr to file
with open(output_lab, "w") as f:
    f.write(labstr)


results_to_ssa(word_timestamps, output_ass)
print(f"Word timestamps saved in SSA/ASS format to {output_ass}")