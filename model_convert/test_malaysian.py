import torch
from transformers.models.whisper import tokenization_whisper

tokenization_whisper.TASK_IDS = ["translate", "transcribe", 'transcribeprecise']

from transformers import (
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    WhisperTokenizerFast
)
import soundfile as sf
import numpy as np
from typing import Tuple
import whisper
import base64

from export_malaysian import get_args


args = get_args()

def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_feat(filename: str, n_mels: int):
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        import librosa

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    audio = whisper.pad_or_trim(wave)
    assert audio.shape == (16000 * 30,), audio.shape

    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0)
    assert mel.shape == (1, n_mels, 3000), mel.shape

    return mel.to(torch.float32)


sr = 16000
model_name = args.model
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    f'mesolitica/malaysian-whisper-{model_name}'
)
processor = WhisperProcessor.from_pretrained(
    f'mesolitica/malaysian-whisper-{model_name}'
)
tokenizer = WhisperTokenizerFast.from_pretrained(
    f'mesolitica/malaysian-whisper-{model_name}'
)
model = WhisperForConditionalGeneration.from_pretrained(
    f'mesolitica/malaysian-whisper-{model_name}', 
    dtype = torch.float32,
).cpu()

print(f"new token <|transcribeprecise|> is {tokenizer.convert_tokens_to_ids('<|transcribeprecise|>')}")
print(f"new token <|notimestamps|> is {tokenizer.convert_tokens_to_ids('<|notimestamps|>')}")
 
print(f"n_mels: {model.config.num_mel_bins}")
print(f"encoder_layers: {model.config.encoder_layers}")
print(f"decoder_layers: {model.config.decoder_layers}")

with torch.no_grad():
    # p = processor([assembly], return_tensors='pt')
    # p['input_features'] = p['input_features'].to(torch.float32)

    feature = compute_feat('./malaysian_test/G5001/G5001_1_S0007.wav', model.config.num_mel_bins)
    r = model.generate(
        feature,
        output_scores=True,
        return_dict_in_generate=True,
        return_timestamps=True, 
        language='ms',
        task = 'transcribe',
    )

tokens = r['sequences'][0]
# print(f"tokens: {tokens}")
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens)))
