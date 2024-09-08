import json
import os

import torch

from .alignment_utils import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
)
from .text_utils import postprocess_results, preprocess_text

TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", help="path of the audio file", required=True)
    parser.add_argument(
        "--text_path", help="path of the text to be aligned", 
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        required=True,
        help="Language in ISO 639-3 code. Identifying the input as Arabic, Belarusian,"
        " Bulgarian, English, Farsi, German, Ancient Greek, Modern Greek, Pontic Greek"
        ", Hebrew, Kazakh, Kyrgyz, Latvian, Lithuanian, North Macedonian, Russian, "
        "Serbian, Turkish, Ukrainian, Uyghur, Mongolian, Thai, Javanese or Yiddish "
        "will improve romanization for those languages, No effect for other languages.",
    )

    parser.add_argument(
        "--romanize",
        action="store_true",
        default=False,
        help="Enable romanization for non-latin scripts. "
        "Use if you are using a model that doesn't support your language vocabulary.",
    )

    parser.add_argument(
        "--split_size",
        type=str,
        default="word",
        choices=["sentence", "word", "char"],
        help="Whether to align on a sentence, word, or character level.",
    )
    parser.add_argument(
        "--star_frequency",
        type=str,
        default="edges",
        choices=["segment", "edges"],
        help="The frequency of the <star> token in the text."
        "Star token increases the accuracy of the alignment but also increases segment fragmentation."
        "segment adds <star> token after each segment."
        "edges adds <star> token at the start and end of the text."
        "use --merge_threshold to merge segments that are closer than the threshold.",
    )
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.00,
        help="merge segments that are closer than the threshold."
        "used to remove very small time differences between segments.",
    )

    parser.add_argument(
        "--alignment_model",
        default="MahmoudAshraf/mms-300m-1130-forced-aligner",
        help="Name of the CTC (Wav2Vec2/HuBERT/MMS) model to use for alignment,"
        " you can choose a language-specific model or an "
        "english model along with --romanize flag to support all languages."
        " accepts Huggingface model names or local pathes.",
    )

    # compute related arguments
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="float16" if torch.cuda.is_available() else "float32",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype for alignment model inference. Helps with speed and memory usage.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for  inference.",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="ًWindow size in seconds to chunk the audio file for alignment.",
    )

    parser.add_argument(
        "--context_size",
        type=int,
        default=2,
        help="ًOverlab between chunks in seconds.",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", None],
        help="Attention implementation for the model. It will chose the fastest implementation by default.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer, dictionary = load_alignment_model(
        args.device,
        args.alignment_model,
        args.attn_implementation,
        TORCH_DTYPES[args.compute_dtype],
    )

    print(f"Loading audio {args.audio_path} ...")
    audio_waveform = load_audio(args.audio_path, model.dtype, model.device)

    print("generate_emissions...")
    emissions, stride = generate_emissions(
        model, audio_waveform, args.window_size, args.context_size, args.batch_size
    )
    
    audio_path = args.audio_path
    if len(audio_path) < 2:
        audio_path = "e1.wav"
    fn = os.path.splitext(audio_path)[0]
    text_path = fn + ".txt"

    with open(text_path, "r") as f:
        lines = f.readlines()
    #text = "".join(line for line in lines).replace("\n", " ").strip()
    isChar = True
    retrp =  "¯" if isChar else " ¯ "
    text = "".join(line for line in lines).replace("\n",retrp).strip()

    print("preprocess_text...")
    tokens_starred, text_starred = preprocess_text(
        text, args.romanize, args.language, 
        split_size= "char" if isChar else args.split_size, 
        star_frequency = args.star_frequency
    )

    print("get_alignments...")
    segments, scores, blank_id = get_alignments(
        emissions,
        tokens_starred,
        dictionary,
    )

    spans = get_spans(tokens_starred, segments, tokenizer.decode(blank_id))

    # results = postprocess_results(
    #     text_starred, spans, stride, scores, args.merge_threshold
    # )
    # # write the results to a file
    # with open(f"{os.path.splitext(args.audio_path)[0]}.txt", "w") as f:
    #     for result in results:
    #         f.write(f"{result['start']}-{result['end']}: {result['text']}\n")
    # # write the results to a json file with the whole text and each segment
    # with open(f"{os.path.splitext(args.audio_path)[0]}.json", "w") as f:
    #     json.dump(
    #         {
    #             "text": text,
    #             "segments": results,
    #         },
    #         f,
    #         indent=4,
    #     )

    output_file = os.path.splitext(args.audio_path)[0]
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    jsonRoot = {
        "text":text,
        "segments":word_timestamps,
    }

    with open(output_file+".json", "w") as f:
        json.dump(jsonRoot, f, indent=2)


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



    results_to_ssa(word_timestamps, output_file+".ass")
    print(f"Word timestamps saved in SSA/ASS format")

if __name__ == "__main__":
    cli()
