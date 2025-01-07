import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import os
from model import CRISPRModel

class CRISPRDatasetCSV(Dataset):
    def __init__(self, csv_path, max_len=23):
        """
        Initializes the dataset by loading data from a CSV file.
        :param csv_path: Path to the CSV file.
        :param max_len: Maximum sequence length for DNA sequences and epigenetic features.
        """
        self.data = self.load_csv(csv_path)
        self.max_len = max_len

    def load_csv(self, csv_path):
        """
        Loads the data from a CSV file.
        :param csv_path: Path to the CSV file.
        :return: List of data entries (each entry is a dictionary with DNA sequence, epigenetic features, and label).
        """
        data = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader) 
            for row in reader:
                if len(row) >= 10:
                    dna_sequence = row[4] 
                    epigenetic_features = ''.join(row[5:9])  
                    try:
                        label = float(row[9]) 
                        data.append({
                            "dna_sequence": dna_sequence,
                            "epigenetic_features": epigenetic_features,
                            "label": label
                        })
                    except ValueError:
                        print(f"Skipping row with invalid label: {row}")
                else:
                    print(f"Skipping row with insufficient columns: {row}")
        return data

    def one_hot_encode(self, sequence):
        """
        One-hot encodes a DNA sequence.
        :param sequence: DNA sequence (string)
        :return: List of one-hot encoded arrays
        """
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return [mapping.get(base, [0, 0, 0, 0]) for base in sequence]

    def encode_epigenetic_features(self, features):
        """
        Encodes epigenetic feature sequence ('A' -> 1, 'N' -> 0).
        :param features: String of 'A' and 'N' characters
        :return: List of binary values
        """
        return [1 if char == 'A' else 0 for char in features]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        dna_sequence = entry["dna_sequence"]
        epigenetic_features = entry["epigenetic_features"]
        label = entry["label"]

        dna_encoded = self.one_hot_encode(dna_sequence)
        dna_encoded = dna_encoded[:self.max_len]
        while len(dna_encoded) < self.max_len: 
            dna_encoded.append([0, 0, 0, 0])

        epi_encoded = self.encode_epigenetic_features(epigenetic_features)
        epi_encoded = epi_encoded[:self.max_len]
        while len(epi_encoded) < self.max_len: 
            epi_encoded.append(0)

        dna_tensor = torch.tensor(dna_encoded, dtype=torch.float32)
        epi_tensor = torch.tensor(epi_encoded, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return dna_tensor, epi_tensor, label_tensor

def train_model(model, train_loader, criterion, optimizer, device, epochs=10, checkpoints_dir="checkpoints"):
    model.to(device)
    os.makedirs(checkpoints_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for dna_batch, epi_batch, labels in progress_bar:
            dna_batch, epi_batch, labels = dna_batch.to(device), epi_batch.to(device), labels.to(device)

            outputs = model(dna_batch, epi_batch)
            loss = criterion(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(checkpoints_dir, f"crispr_model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

seq_length = 23
num_epigenetic_features = 4 * seq_length
batch_size = 32
epochs = 10
learning_rate = 0.001

dataset_path = "data/data.csv"
dataset = CRISPRDatasetCSV(dataset_path, max_len=seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CRISPRModel(seq_input_shape=(seq_length, 4), epigenetic_input_shape=num_epigenetic_features)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_model(model, dataloader, criterion, optimizer, device, epochs)


torch.save(model.state_dict(), "crispr_model_final.pth")
print("Final model saved as 'crispr_model_final.pth'")