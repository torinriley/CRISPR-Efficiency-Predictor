import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import one_hot_encode, encode_epigenetic_features, build_crispr_model

file_path = "data/data.csv"  
columns = ["Chrom", "Start", "End", "Strand", "Target Seq", "CTCF", "Dnase", "H3K4me3", "RRBS", "Score"]
data = pd.read_csv(file_path, sep="\t", names=columns)

print(f"Dataset loaded with {len(data)} rows.")

print("One-hot encoding DNA sequences...")
sequence_input = np.array([one_hot_encode(seq) for seq in data['Target Seq']])

print("Encoding epigenetic features...")
epigenetic_input = np.array([
    np.hstack([
        encode_epigenetic_features(row['CTCF']),
        encode_epigenetic_features(row['Dnase']),
        encode_epigenetic_features(row['H3K4me3']),
        encode_epigenetic_features(row['RRBS'])
    ])
    for _, row in data.iterrows()
])

target = data['Score'].values

print(f"Sequence input shape: {sequence_input.shape}")
print(f"Epigenetic input shape: {epigenetic_input.shape}")
print(f"Target shape: {target.shape}")

print("Splitting data into training and testing sets...")
seq_train, seq_test, epi_train, epi_test, y_train, y_test = train_test_split(
    sequence_input, epigenetic_input, target, test_size=0.2, random_state=42
)

print(f"Training set: {seq_train.shape}, {epi_train.shape}, {y_train.shape}")
print(f"Testing set: {seq_test.shape}, {epi_test.shape}, {y_test.shape}")

seq_length = sequence_input.shape[1]
num_epigenetic_features = epigenetic_input.shape[1]

print("Building model...")
model = build_crispr_model((seq_length, 4), (num_epigenetic_features,))

print("Training the model...")
history = model.fit(
    [seq_train, epi_train], y_train,
    validation_split=0.2,
    epochs=10, 
    batch_size=16,
    verbose=1
)

print("Evaluating the model on the test set...")
loss, mae = model.evaluate([seq_test, epi_test], y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

model.save("models/crispr_model.h5")
print("Model saved as 'crispr_model.h5'.")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training_loss_plot.png")
plt.show()

print("Training complete. Results saved.")
