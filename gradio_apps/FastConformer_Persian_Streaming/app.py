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

def transcribe(stream, new_chunk):
    if new_chunk is None:
        return None, ""
    # 'audio' is a tuple: (sample_rate, data)
    sample_rate, data = new_chunk

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

    if stream is not None:
        stream['audio'] = torch.cat([stream['audio'], audio_tensor], dim=-1)
    else:
        stream = {"text": ""}
        stream['audio'] = audio_tensor


    max_length = 5 * target_sample_rate  # 5 seconds
    new_text = ""

    # Process all chunks that fit max_length
    while stream['audio'].shape[-1] > max_length:
        # Extract first max_length samples
        audio_chunk = stream['audio'][..., :max_length]

        # Transcribe
        with torch.no_grad():
            transcript = model.transcribe(audio_chunk)  # Add batch dimension if needed

        # Update text (adjust based on model's output format)
        new_text += " " + transcript[0][0].strip()  # Example adjustment

        # Remove processed audio from buffer
        stream['audio'] = stream['audio'][..., max_length:]

    stream['text'] += new_text
    return stream, stream['text'].strip()


interface = gr.Interface(
    fn=transcribe,
    inputs=['state', gr.Audio(sources="microphone", streaming=True, type="numpy")],
    outputs=["state", "text"],
    live=True,
)

interface.launch()