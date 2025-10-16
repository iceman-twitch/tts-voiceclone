from TTS.api import TTS
import torch
import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))




def text_to_speech(text, output_path):
    #Setting for COQUI
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(model_name, progress_bar=False, gpu=True).to(device)

    filename = os.path.join(output_path, "output.wav")
    model = "coqui"
    tts.tts_to_file(text=text,
                    speaker_wav=f"{BASE_PATH}/sample.wav",
                    language="en",
                    file_path=filename)

if __name__ == "__main__":
    text_to_speech("Testing TTS. Is it working well?", BASE_PATH)
    # print("torch:", torch.__version__)
    # print("torch.version.cuda:", torch.version.cuda)
    # print("cuda available:", torch.cuda.is_available())
    # print("cudnn version:", torch.backends.cudnn.version())
