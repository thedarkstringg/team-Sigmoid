import torch
import torch.nn as nn


class TemporalRiskModel(nn.Module):
    """
    GRU-based per-timestep fault-risk model.

    Input:  (batch_size, seq_len, input_size) SCADA window
    Output: (batch_size, seq_len) raw logits (apply sigmoid separately
            for actual probabilities during inference/eval)
    """

    def __init__(self, input_size=54, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)  # per-timestep, applied automatically

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        gru_out, _ = self.gru(x)          # (batch, seq_len, hidden_size)
        logits = self.fc(gru_out)          # (batch, seq_len, 1)
        return logits.squeeze(-1)          # (batch, seq_len) - raw logits