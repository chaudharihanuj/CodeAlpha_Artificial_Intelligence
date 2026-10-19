
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_model(sequence_length, vocabulary_size):
    """Create an LSTM model for music generation."""
    model = Sequential([
        LSTM(256, input_shape=(sequence_length, 1),
             return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(vocabulary_size, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    return model
