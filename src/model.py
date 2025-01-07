import torch
import torch.nn as nn

class CRISPRModel(nn.Module):
    def __init__(self, seq_input_shape, epigenetic_input_shape):
        """
        Initializes the CRISPR model.
        :param seq_input_shape: Tuple (seq_length, num_channels) for DNA sequences.
        :param epigenetic_input_shape: Number of epigenetic features.
        """
        super(CRISPRModel, self).__init__()

        self.seq_conv = nn.Conv1d(
            in_channels=seq_input_shape[1],
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.seq_pool = nn.MaxPool1d(kernel_size=2)

        self.epi_fc = nn.Linear(epigenetic_input_shape, 64)

        seq_output_size = (seq_input_shape[0] // 2) * 64  
        self.fc = nn.Linear(64 + seq_output_size, 1)
        seq_output_size = (seq_input_shape[0] // 2 + seq_input_shape[0] % 2) * 64  
    def forward(self, seq_input, epi_input):
        """
        Forward pass for the model.
        :param seq_input: Tensor of shape (batch_size, seq_length, num_channels)
        :param epi_input: Tensor of shape (batch_size, num_epigenetic_features)
        :return: Output tensor of shape (batch_size, 1)
        """
        seq_output = self.seq_conv(seq_input.permute(0, 2, 1))  # Change to (batch_size, channels, seq_length)
        seq_output = torch.relu(seq_output)
        seq_output = self.seq_pool(seq_output)  
        seq_output = seq_output.view(seq_output.size(0), -1) 

        epi_output = torch.relu(self.epi_fc(epi_input))

        combined = torch.cat((seq_output, epi_output), dim=1)

        output = self.fc(combined)
        return output

if __name__ == "__main__":
    seq_length = 23 
    num_epigenetic_features = 4 * seq_length  

    seq_shape = (seq_length, 4)
    epi_shape = num_epigenetic_features

    model = CRISPRModel(seq_input_shape=seq_shape, epigenetic_input_shape=epi_shape)
    print(model)

    batch_size = 16
    seq_input = torch.rand(batch_size, seq_shape[0], seq_shape[1])  # (batch_size, seq_length, 4)
    epi_input = torch.rand(batch_size, epi_shape)  # (batch_size, num_epigenetic_features)

    output = model(seq_input, epi_input)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, 1)