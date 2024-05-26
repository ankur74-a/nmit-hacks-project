import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2, Preview
from PIL import Image
import time

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path="modelnew.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the expected input shape for the model
input_shape = input_details[0]['shape']
print(f"Expected input shape: {input_shape}")

# Define a dictionary to map predictions to labels
labels = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper"}

# Prediction function using TFLite
def pred(image_path):
    # Load the image from the file
    frame = Image.open(image_path)
    
    # Resize frame to match the input shape of the model
    frame = frame.resize((input_shape[1], input_shape[2]))
    frame = np.expand_dims(frame, axis=0)
    frame = np.array(frame) / 255.0  # Normalize the frame
    
    interpreter.set_tensor(input_details[0]['index'], frame.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Print output_data for debugging
    print(f"Model output data: {output_data}")

    if np.max(output_data) * 100 > 0:  # Adjust confidence threshold if necessary
        pred = np.argmax(output_data)
        label = labels.get(pred, "Unknown")
        print(f"Prediction: {label} with confidence {np.max(output_data) * 100}%")
        return label
    else:
        print("Low confidence prediction.")
        return None

# Capture function
def cap_and_save():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    time.sleep(2)  # Allow the camera to warm up

    while True:
        user_input = input("Press Enter to capture an image or 'q' to quit...")
        if user_input.lower() == 'q':
            break

        image_path = 'captured_image.jpg'
        
        # Capture image
        picam2.capture_file(image_path)
        print(f"Image saved to {image_path}")
        pred(image_path)
        
    picam2.stop()

if __name__ == "__main__":
    cap_and_save()
