import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
from config import MODEL_PATH, IMAGE_SIZE

def load_and_prepare_image(img_path, target_size=IMAGE_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

def predict_image(img_path):
    model = load_model(MODEL_PATH)
    img_array = load_and_prepare_image(img_path)
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "PNEUMONIA"
    else:
        result = "NORMAL"
    
    print(f"Prediction: {result} ({prediction[0][0]:.4f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <path_to_image>")
    else:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print("Image path not found!")
        else:
            predict_image(img_path)
