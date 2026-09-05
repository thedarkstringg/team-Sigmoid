import torch
import torch.nn as nn


class TemporalRiskModel(nn.Module):
    """
    GRU-based per-timestep fault-risk model.

    Input:  (batch_size, seq_len, input_size) SCADA window
    Output: (batch_size, seq_len) raw logits (apply sigmoid separately
            for actual probabilities during inference/eval)
    """

    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.3,
                 bidirectional=False):
        """
        input_size: number of physical subsystem feature vectors per timestep
        (e.g. power residual, thermal deltas, kinematic ratio, vibration index,
        and any others Ismayil's feature engineering identifies as reliably
        computable across all farms via the sensor description file). This is
        NOT fixed at a specific count - it depends on what the data pipeline
        actually builds, so it must be passed explicitly, not assumed.

        bidirectional: if True, the GRU also uses "future" timesteps within
        the same observed window to help predict risk at earlier timesteps.
        This is valid for OFFLINE evaluation (the whole window is already
        observed data), but is NOT causally deployable as a real-time
        streaming model, since live inference would not yet have future
        data. Treat this as a controlled ablation experiment, not
        automatically the deployed model, unless that trade-off is
        explicitly accepted and documented.
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # nn.GRU ignores/warns if num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_size, 1)  # per-timestep, applied automatically

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        gru_out, _ = self.gru(x)          # (batch, seq_len, hidden_size)
        gru_out = self.dropout(gru_out)    # extra dropout before the final projection
        logits = self.fc(gru_out)          # (batch, seq_len, 1)
        return logits.squeeze(-1)          # (batch, seq_len) - raw logits