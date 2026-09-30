import soundfile as sf
import os

SRC = "speech_commands/_background_noise_"
DEST_TRAIN = "dataset/training/silence"
DEST_VAL = "dataset/validation/silence"
os.makedirs(DEST_TRAIN, exist_ok=True)
os.makedirs(DEST_VAL, exist_ok=True)
CLIP_LEN = 1.5

count = 0
for fname in os.listdir(SRC):
    if not fname.endswith(".wav"):
        continue
    audio, sr = sf.read(os.path.join(SRC, fname))
    clip_samples = int(CLIP_LEN * sr)
    n_clips = len(audio) // clip_samples
    for i in range(n_clips):
        chunk = audio[i*clip_samples : (i+1)*clip_samples]
        # send every 6th clip to validation, rest to training
        dest = DEST_VAL if count % 6 == 0 else DEST_TRAIN
        sf.write(f"{dest}/{fname[:-4]}_{i}.wav", chunk, sr)
        count += 1

print(f"Done chopping background noise into {count} clips.")