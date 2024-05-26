import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("modelnew2.h5")
model.summary()

def pred(frame):
  frame1 = cv2.resize(frame,(224,224))
  frame1 = np.expand_dims(frame1,axis=0)
  # Flatten the image before feeding to the model 
  frame1 = tf.keras.layers.Flatten()(frame1)
  pred = model.predict([frame1])
  if np.max(pred)*100 > 80: 
    pred = np.argmax(pred)
    print(pred)

def cap():
  cap = cv2.VideoCapture(0)  # Assuming device index 0 for webcam
  frameCount = 0  # Counter to skip initial frames (optional)
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  # Check if frame reading failed
      print("Error reading frame!")
      break
    # Optional: Skip initial frames (adjust frameCount as needed)
    if frameCount > 5:
      frame1 = cv2.resize(frame, (224, 224))
      frame1 = np.expand_dims(frame1, axis=0)
      pred(frame1)
    frameCount += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()
