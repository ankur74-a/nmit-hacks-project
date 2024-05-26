import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2, Preview
from PIL import Image
import time

# Loading TFLite model and allocate tensors
intptr = Interpreter(model_path="modelnew.tflite")
intptr.allocate_tensors()

# Getting input and output tensors
i/p_details = intptr.get_input_details()
o/p_details = intptr.get_output_details()

# Get the expected input shape for the model
i/p_shape = i/p_details[0]['shape']
print(f"Expected shape of input: {i/p_shape}")

# Define a dictionary to map predictions to labels
labels = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper"}

# Prediction function using TFLite
def pred(image_path):
    # Load the image from the file
    frame = Image.open(image_path)
    
    # Resize frame to match the input shape of the model
    frame = frame.resize((i/p_shape[1], i/p_shape[2]))
    frame = np.expand_dims(frame, axis=0)
    frame = np.array(frame) / 255.0 
    
    intptr.set_tensor(i/p_details[0]['index'], frame.astype(np.float32))
    intptr.invoke()
    o/p_data = intptr.get_tensor(o/p_details[0]['index'])

    # Printing output_data for debugging
    print(f"Output data of model: {o/p_data}")

    if np.max(o/p_data) * 100 > 0:
        pred = np.argmax(o/p_data)
        label = labels.get(pred, "Unknown")
        print(f"Prediction: {label} with confidence {np.max(o/p_data) * 100}%")
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
    
    time.sleep(2)  

    while True:
        user_input = input("Press Enter to capture an image or 'q' to quit...")
        if user_input.lower() == 'q':
            break

        img_path = 'captured_image.jpg'
        
        # Capture image
        picam2.capture_file(img_path)
        print(f"Image saved to {img_path}")
        pred(img_path)
        
    picam2.stop()

if __name__ == "__main__":
    cap_and_save()
