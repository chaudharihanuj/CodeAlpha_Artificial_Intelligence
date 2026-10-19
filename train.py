
from pathlib import Path
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from model import create_model

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "notes.pkl"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "music_model.keras"
VOCAB_FILE = MODEL_DIR / "vocabulary.pkl"

SEQUENCE_LENGTH = 100


def main():
    if not DATA_FILE.exists():
        print("Run preprocess.py first.")
        return

    with DATA_FILE.open("rb") as file:
        notes = pickle.load(file)

    vocabulary = sorted(set(notes))
    note_to_int = {value: i for i, value in enumerate(vocabulary)}

    if len(notes) <= SEQUENCE_LENGTH:
        print("Not enough notes. Add more MIDI training data.")
        return

    network_input = []
    network_output = []

    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence = notes[i:i + SEQUENCE_LENGTH]
        target = notes[i + SEQUENCE_LENGTH]

        network_input.append([note_to_int[n] for n in sequence])
        network_output.append(note_to_int[target])

    X = np.array(network_input, dtype=np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X /= max(len(vocabulary) - 1, 1)

    y = to_categorical(
        network_output, num_classes=len(vocabulary)
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = create_model(SEQUENCE_LENGTH, len(vocabulary))

    model.fit(
        X, y,
        epochs=30,
        batch_size=64,
        verbose=1
    )

    model.save(MODEL_FILE)

    with VOCAB_FILE.open("wb") as file:
        pickle.dump(vocabulary, file)

    print("Training completed!")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
