import torch
import whisper

MODEL_DIR='./TOFILL'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("dowload Whisper model")
asr_model = whisper.load_model('large-v2', device, download_root=MODEL_DIR)
