import os
import glob
import numpy as np
import torch
import torchaudio
import soundfile as sf
import tensorflow as tf

# --- Must match utils.py exactly ---
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 480
HOP_LENGTH = 160
TARGET_LEN = 16000  # 1 second

melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

CLASSES = ["silence", "unknown", "wakeword"]

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
    spec = spec.unsqueeze(0)
    return spec.numpy().astype(np.float32)

MODEL_PATH = "bcresnet_astra_full_int8.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(wav_path):
    x = preprocess(wav_path)
    x = np.transpose(x, (0, 2, 3, 1))  # -> [1, 40, 101, 1]

    in_scale, in_zero_point = input_details[0]["quantization"]
    x_int8 = (x / in_scale + in_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], x_int8)
    interpreter.invoke()

    out_scale, out_zero_point = output_details[0]["quantization"]
    output_int8 = interpreter.get_tensor(output_details[0]["index"])[0]
    output = (output_int8.astype(np.float32) - out_zero_point) * out_scale

    probs = np.exp(output) / np.sum(np.exp(output))
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], float(probs[pred_idx])

correct = 0
total = 0
for true_label in CLASSES:
    folder = f"dataset/validation/{true_label}"
    files = glob.glob(f"{folder}/*.wav")[:15]
    for f in files:
        pred_label, confidence = predict(f)
        mark = "correct" if pred_label == true_label else "WRONG"
        print(f"{os.path.basename(f):30s} actual: {true_label:10s} -> predicted: {pred_label:10s} ({confidence*100:.1f}%) {mark}")
        total += 1
        if pred_label == true_label:
            correct += 1

print(f"\nOverall: {correct}/{total} correct ({100*correct/total:.1f}%)")