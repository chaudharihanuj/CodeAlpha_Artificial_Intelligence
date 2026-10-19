# CodeAlpha_Artificial_Intelligence
Artificial Intelligence and Machine Learning tasks and projects completed during the CodeAlpha Internship Program.

# Music Dataset

This folder stores MIDI files used to train the AI music generation model.

## Dataset Requirements
- File formats: `.mid` or `.midi`
- Music types: Classical, piano, instrumental, or other MIDI music
- Use multiple MIDI files for better training results.

## How to Use
1. Download MIDI music files from a suitable dataset.
2. Place the files inside this folder or its subfolders.
3. Run `python src/preprocess.py` from the project root.
4. The extracted note sequences will be saved as `notes.pkl`.

**Note:** MIDI files and generated training data may be excluded from GitHub to keep the repository small.
