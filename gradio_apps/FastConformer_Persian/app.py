import torch
import nemo.collections.asr as nemo_asr
import gc
import numpy as np
import torchaudio

pretrained_model_path="./stt_fa_fastconformer_hybrid_large_finetuned.nemo"

# Clear up memory
torch.cuda.empty_cache()
gc.collect()
model = nemo_asr.models.EncDecHybridRNNTCTCModel.restore_from(pretrained_model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'  # You can transcribe even longer samples on the CPU, though it will take much longer !
model = model.to(device)
model.freeze()

def transcribe(audio):
    # 'audio' is a tuple: (sample_rate, data)
    sample_rate, data = audio

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Ensure the model is on the correct device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert audio data to the expected format
    if isinstance(data, np.ndarray):
        audio_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        raise ValueError("Audio data must be a numpy array")

    # Resample if sample rate is not 16000
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        audio_tensor = resampler(audio_tensor)

    # Trim audio if longer than 30 seconds
    max_length = 30 * target_sample_rate  # 30 seconds
    if audio_tensor.shape[-1] > max_length:
        audio_tensor = audio_tensor[..., :max_length]

    # Transcribe
    with torch.no_grad():
        transcript = model.transcribe(audio_tensor)

    return transcript[0][0]  # Assuming single input

import gradio as gr

interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["upload", "microphone"]),  # Allows both file upload and recording
    outputs="text",
    live=False  # Set to True for real-time transcription
)

interface.launch()