import numpy as np
import tensorflow as tf

SAVED_MODEL_DIR = "tflite_output_int8"  # already created by your last onnx2tf run
OUTPUT_FILE = "bcresnet_astra_full_int8.tflite"

calib_data = np.load("calib_data.npy")  # shape [N, 1, 40, 101]

def representative_dataset():
    for i in range(calib_data.shape[0]):
        sample = calib_data[i]                      # [1, 40, 101]
        sample = np.transpose(sample, (1, 2, 0))     # -> [40, 101, 1]
        sample = np.expand_dims(sample, axis=0)      # -> [1, 40, 101, 1]
        yield [sample.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(OUTPUT_FILE, "wb") as f:
    f.write(tflite_model)

print(f"Saved {OUTPUT_FILE} — {len(tflite_model)} bytes")