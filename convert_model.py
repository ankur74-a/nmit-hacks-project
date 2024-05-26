import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("modelnew2.h5")

# Function to convert the model using a concrete function
def convert_model_to_tflite(model):
    # Create a concrete function from the model
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    # Save the model
    with open('modelnew2.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted successfully to TFLite!")

convert_model_to_tflite(model)
