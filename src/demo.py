import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from model import one_hot_encode, encode_epigenetic_features

def preprocess_input(data):
    """
    Preprocesses DNA sequences and epigenetic features for inference.
    :param data: Pandas DataFrame with columns ['Target Seq', 'CTCF', 'Dnase', 'H3K4me3', 'RRBS']
    :return: Tuple of Numpy arrays (sequence_input, epigenetic_input)
    """
    # Process DNA sequences
    sequence_input = [one_hot_encode(seq) for seq in data['Target Seq']]
    max_length = max(len(seq) for seq in sequence_input)
    sequence_input = np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant') for seq in sequence_input])

    # Process epigenetic features
    epigenetic_features = []
    for _, row in data.iterrows():
        features = np.hstack([
            encode_epigenetic_features(row['CTCF']),
            encode_epigenetic_features(row['Dnase']),
            encode_epigenetic_features(row['H3K4me3']),
            encode_epigenetic_features(row['RRBS'])
        ])
        epigenetic_features.append(features)
    
    max_length = max(features.shape[0] for features in epigenetic_features)
    epigenetic_input = np.array([np.pad(features, (0, max_length - features.shape[0]), mode='constant') for features in epigenetic_features])

    return sequence_input, epigenetic_input

def main():
    # Load trained model
    model_path = "models/crispr_model.h5"  # Path to the trained model
    print("Loading model...")
    model = load_model(model_path)

    # Load new data for inference
    data_path = "data/new_crispr_data.csv"  # Replace with your inference dataset path
    columns = ["Chrom", "Start", "End", "Strand", "Target Seq", "CTCF", "Dnase", "H3K4me3", "RRBS", "True Score"]  # Add "True Score" if available
    data = pd.read_csv(data_path, delimiter=",", names=columns, dtype=str)

    # Preprocess input data
    sequence_input, epigenetic_input = preprocess_input(data)

    # Perform inference
    predictions = model.predict([sequence_input, epigenetic_input])
    print(predictions)

    # Save predictions to CSV
    output_path = "results/results.csv"
    data['Predictions'] = predictions
    data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()