
from pathlib import Path
import pickle
from music21 import converter, instrument, note, chord

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "notes.pkl"


def extract_notes(midi_file):
    """Extract notes and chords from a MIDI file."""
    try:
        midi = converter.parse(str(midi_file))
        notes = []

        parts = instrument.partitionByInstrument(midi)

        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flatten().notes

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append(".".join(
                    str(pitch) for pitch in element.normalOrder
                ))

        return notes

    except Exception as error:
        print(f"Could not process {midi_file}: {error}")
        return []


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    midi_files = list(DATA_DIR.rglob("*.mid"))
    midi_files += list(DATA_DIR.rglob("*.midi"))

    all_notes = []

    for midi_file in midi_files:
        if midi_file.name != OUTPUT_FILE.name:
            extracted = extract_notes(midi_file)
            all_notes.extend(extracted)
            print(f"Processed: {midi_file}")

    if not all_notes:
        print("No notes found. Add MIDI files inside the data folder.")
        return

    with OUTPUT_FILE.open("wb") as file:
        pickle.dump(all_notes, file)

    print(f"Extracted {len(all_notes)} musical events.")
    print(f"Saved notes to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
