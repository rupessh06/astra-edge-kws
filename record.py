import sounddevice as sd
import soundfile as sf
import os
import glob

OUT_DIR = "dataset/validation/wakeword"
os.makedirs(OUT_DIR, exist_ok=True)
SR = 16000
DURATION = 1.5

existing = glob.glob(f"{OUT_DIR}/kw_*.wav")
start_idx = len(existing)
print(f"Found {start_idx} existing clips in {OUT_DIR}. New clips will continue from kw_{start_idx:04d}.")

n = int(input("How many NEW clips to record? "))
for j in range(n):
    i = start_idx + j
    input(f"Press Enter, then say your keyword (clip {j+1}/{n})...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    fname = f"{OUT_DIR}/kw_{i:04d}.wav"
    sf.write(fname, audio, SR)
    print(f"saved {fname}")
print("Done.")