import sys

INPUT_FILE = "bcresnet_astra_full_int8.tflite"
OUTPUT_FILE = "model_data.h"
VAR_NAME = "g_model"

with open(INPUT_FILE, "rb") as f:
    data = f.read()

with open(OUTPUT_FILE, "w") as f:
    f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
    f.write("alignas(8) const unsigned char " + VAR_NAME + "[] = {\n")
    for i, b in enumerate(data):
        f.write(f"0x{b:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int {VAR_NAME}_len = {len(data)};\n")
    f.write("\n#endif\n")

print(f"Wrote {OUTPUT_FILE} — {len(data)} bytes")