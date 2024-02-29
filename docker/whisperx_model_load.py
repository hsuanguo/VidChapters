import argparse
import torch
import os
from args import get_args_parser, MODEL_DIR
import whisper

parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args()
print(args.model_name)
args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
device = torch.device(args.device)

print("load Whisper model")
asr_model = whisper.load_model('large-v2', args.device, download_root=MODEL_DIR)
