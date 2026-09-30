import glob
import numpy as np
import torch
import torchaudio
import soundfile as sf

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 480
HOP_LENGTH = 160
TARGET_LEN = 16000

melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def preprocess(wav_path):
    audio_np, sr = sf.read(wav_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[1] < TARGET_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LEN - waveform.shape[1]))
    else:
        waveform = waveform[:, :TARGET_LEN]
    spec = melspec(waveform)
    spec = amplitude_to_db(spec)
    return spec.unsqueeze(0).numpy().astype(np.float32)  # [1,1,40,101]

samples = []
for cls in ["silence", "unknown", "wakeword"]:
    files = glob.glob(f"dataset/training/{cls}/*.wav")[:60]
    for f in files:
        samples.append(preprocess(f)[0])

data = np.stack(samples, axis=0)  # [N,1,40,101]
np.save("calib_data.npy", data)
print(f"Saved calibration data: {data.shape}")