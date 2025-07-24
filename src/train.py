from model import build_cnn_model
from preprocessing import get_data_generators
from config import MODEL_PATH

def train_model():
    train_gen, val_gen, _ = get_data_generators()
    model = build_cnn_model()

    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen
    )

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return history

if __name__ == "__main__":
    train_model()
