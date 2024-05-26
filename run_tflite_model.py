import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="modelnew.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare your input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Set the tensor to point to the input data to be inferred
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the inference
interpreter.invoke()

# Extract the output data
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
