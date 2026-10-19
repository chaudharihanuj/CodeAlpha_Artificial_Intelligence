
from pathlib import Path
import pickle
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "music_model.keras"
VOCAB_FILE = ROOT / "models" / "vocabulary.pkl"
DATA_FILE = ROOT / "data" / "notes.pkl"
OUTPUT_DIR = ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "generated_notes.pkl"

SEQUENCE_LENGTH = 100
GENERATED_LENGTH = 200


def main():
    required_files = [MODEL_FILE, VOCAB_FILE, DATA_FILE]

    if not all(path.exists() for path in required_files):
        print("Train the model first by running train.py.")
        return

    model = load_model(MODEL_FILE)

    with VOCAB_FILE.open("rb") as file:
        vocabulary = pickle.load(file)

    with DATA_FILE.open("rb") as file:
        notes = pickle.load(file)

    note_to_int = {value: i for i, value in enumerate(vocabulary)}

    if len(notes) < SEQUENCE_LENGTH:
        print("Not enough notes to create a starting sequence.")
        return

    rng = np.random.default_rng()
    start = int(rng.integers(0, len(notes) - SEQUENCE_LENGTH + 1))
    pattern = [note_to_int[n] for n in notes[start:start + SEQUENCE_LENGTH]]

    generated = []

    for _ in range(GENERATED_LENGTH):
        x = np.array(pattern, dtype=np.float32)
        x = x.reshape(1, SEQUENCE_LENGTH, 1)
        x /= max(len(vocabulary) - 1, 1)

        prediction = model.predict(x, verbose=0)[0]
        index = int(np.argmax(prediction))

        generated.append(vocabulary[index])
        pattern.append(index)
        pattern = pattern[1:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("wb") as file:
        pickle.dump(generated, file)

    print(f"Generated {len(generated)} musical events.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Run midi_player.py to create the MIDI file.")


if __name__ == "__main__":
    main()
