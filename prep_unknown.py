import os, random, shutil

SRC = "speech_commands"
DEST_TRAIN = "dataset/training/unknown"
DEST_VAL = "dataset/validation/unknown"
os.makedirs(DEST_TRAIN, exist_ok=True)
os.makedirs(DEST_VAL, exist_ok=True)

words = [w for w in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, w)) and w != "_background_noise_"]
all_files = []
for w in words:
    folder = os.path.join(SRC, w)
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            all_files.append(os.path.join(folder, f))

random.seed(42)
random.shuffle(all_files)
train_files, val_files = all_files[:400], all_files[400:500]

for f in train_files:
    shutil.copy(f, DEST_TRAIN)
for f in val_files:
    shutil.copy(f, DEST_VAL)

print(f"Copied {len(train_files)} to training/unknown, {len(val_files)} to validation/unknown")