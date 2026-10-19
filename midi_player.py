
from pathlib import Path
import pickle
from music21 import stream, note, chord, pitch, tempo

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "output" / "generated_notes.pkl"
OUTPUT_FILE = ROOT / "output" / "generated_music.mid"


def main():
    if not INPUT_FILE.exists():
        print("Generate music first by running generate_music.py.")
        return

    with INPUT_FILE.open("rb") as file:
        generated_notes = pickle.load(file)

    music = stream.Stream()
    music.append(tempo.MetronomeMark(number=100))

    offset = 0.0
    duration = 0.5

    for token in generated_notes:
        try:
            if "." in token:
                # Chords use pitch-class numbers from preprocess.py.
                pitch_classes = [int(value) for value in token.split(".")]
                pitches = [
                    pitch.Pitch(midi=60 + value % 12)
                    for value in pitch_classes
                ]
                element = chord.Chord(pitches)
            else:
                element = note.Note(token)

            element.offset = offset
            element.duration.quarterLength = duration
            music.append(element)
            offset += duration

        except (ValueError, TypeError) as error:
            print(f"Skipping invalid event {token}: {error}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    music.write("midi", fp=str(OUTPUT_FILE))

    print(f"MIDI file saved to: {OUTPUT_FILE}")
    print("Open the MIDI file in a compatible music player.")


if __name__ == "__main__":
    main()
