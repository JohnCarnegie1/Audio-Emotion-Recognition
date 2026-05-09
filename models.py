import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Classification layers
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):

        # Add feature dimension
        x = x.unsqueeze(-1)

        # LSTM output
        output, (hidden, cell) = self.lstm(x)

        # Concatenate forward + backward hidden states
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        x = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class HuBERT(nn.Module):
    def __init__(self):
        super(HuBERT, self).__init__()

        # Pretrained HuBERT model
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        hidden_size = self.hubert.config.hidden_size

        # Classification layers
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x, attention_mask=None):

        # HuBERT feature extraction
        outputs = self.hubert(
            input_values=x,
            attention_mask=attention_mask
        )

        # Last hidden states
        x = outputs.last_hidden_state
        # shape: (batch_size, seq_len, hidden_size)

        # Mean pooling across time dimension
        x = torch.mean(x, dim=1)

        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
