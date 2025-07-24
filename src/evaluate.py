from tensorflow.keras.models import load_model
from preprocessing import get_data_generators
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from onfig import MODEL_PATH

def evaluate_model():
    _, _, test_gen = get_data_generators()
    model = load_model(MODEL_PATH)
    predictions = model.predict(test_gen)
    predicted_classes = (predictions > 0.5).astype("int32")

    true_labels = test_gen.classes
    print("Classification Report:\n", classification_report(true_labels, predicted_classes))
    
    cm = confusion_matrix(true_labels, predicted_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
