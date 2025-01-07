import numpy as np
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model

def one_hot_encode(sequence):
    """
    One-hot encodes a DNA sequence.
    :param sequence: DNA sequence (string)
    :return: Numpy array of one-hot encoded sequence
    """
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence]) 

def encode_epigenetic_features(epigenetic_seq):
    """
    Encodes epigenetic feature sequence ('A' -> 1, 'N' -> 0).
    :param epigenetic_seq: String of 'A' and 'N' characters
    :return: List of binary values
    """
    return [1 if char == 'A' else 0 for char in epigenetic_seq]

def build_crispr_model(sequence_input_shape, epigenetic_input_shape):
    """
    Builds the multi-input model for CRISPR prediction.
    :param sequence_input_shape: Tuple (sequence_length, 4) for one-hot encoded DNA sequences
    :param epigenetic_input_shape: Tuple (number_of_epigenetic_features,) for encoded epigenetic features
    :return: Compiled Keras model
    """
    seq_input = Input(shape=sequence_input_shape, name="sequence_input")
    seq_branch = Conv1D(filters=32, kernel_size=3, activation='relu')(seq_input)
    seq_branch = MaxPooling1D(pool_size=2)(seq_branch)
    seq_branch = Flatten()(seq_branch)

    epi_input = Input(shape=epigenetic_input_shape, name="epigenetic_input")
    epi_branch = Dense(64, activation='relu')(epi_input)
    epi_branch = Dropout(0.5)(epi_branch)

    combined = concatenate([seq_branch, epi_branch])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(1, activation='linear', name="output")(combined)

    model = Model(inputs=[seq_input, epi_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

if __name__ == "__main__":
    seq_length = 23 
    num_epigenetic_features = 4 * seq_length  

    seq_shape = (seq_length, 4)
    epi_shape = (num_epigenetic_features,) 

    model = build_crispr_model(seq_shape, epi_shape)
    model.summary()