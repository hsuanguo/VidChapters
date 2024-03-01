import argparse
import torch
import os
import pickle
from args import get_args_parser, MODEL_DIR
import whisper
import whisperx
import json

# Args
parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args()

args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
device = torch.device(args.device)

print("load Whisper model")
asr_model = whisper.load_model('large-v2', args.device, download_root=MODEL_DIR)

print("extracting ASR....................")
asr = asr_model.transcribe(args.video_example)

print("load align model....................")
align_model, metadata = whisperx.load_align_model(language_code=asr['language'], device=args.device, model_dir=MODEL_DIR)

print("extracting audio....................")
audio = whisperx.load_audio(args.video_example)

print("align ASR")
aligned_asr = whisperx.align(asr["segments"], align_model, metadata, audio, args.device, return_char_alignments=False)

print("saving........")

# save to 2 files, args.video_example.json as plain text and args.video_example.pkl as pickle
file_name = os.path.basename(args.video_example)
file_name_no_ext = os.path.splitext(file_name)[0]
video_file_dir = '/output'

pickle.dump(aligned_asr, open(os.path.join(video_file_dir,file_name_no_ext + ".pkl"), 'wb'))

with open(os.path.join(video_file_dir,file_name_no_ext + ".json"), 'w') as f:
    json.dump(aligned_asr['segments'], f, indent=4)

print(f"done, saved to {file_name_no_ext}.json and {file_name_no_ext}.pkl")
