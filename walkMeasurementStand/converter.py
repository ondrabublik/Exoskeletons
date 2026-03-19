import tensorflow as tf

model = tf.keras.models.load_model("test_model_v1.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.experimental_new_converter = False
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS
# ]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# convert_to_header.py

with open("model.tflite", "rb") as f:
    data = f.read()

with open("model.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("const unsigned char model_tflite[] = {\n")

    for i, byte in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{byte:02x}, ")
        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n")
    f.write(f"const unsigned int model_tflite_len = {len(data)};\n")