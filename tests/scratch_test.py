import torch
from src.model.model import TemporalRiskModel

# --- Config, matching the design decided on in the proposal / conversation ---
batch_size = 32
seq_len = 144          # 24h window at 10-min resolution: 24*60/10 = 144
num_features = 54      # Farm A feature count
hidden_size = 64

# 1. Dummy input and labels
x_dummy = torch.randn(batch_size, seq_len, num_features)
y_dummy = torch.randint(0, 2, (batch_size, seq_len)).float()

# 2. Instantiate model
model = TemporalRiskModel(input_size=num_features, hidden_size=hidden_size)

# 3. Forward pass
logits = model(x_dummy)
print("logits shape:", logits.shape)          # expect (32, 144)
print("y_dummy shape:", y_dummy.shape)        # expect (32, 144)
assert logits.shape == y_dummy.shape, "Shape mismatch between model output and labels!"

# 4. Loss (no pos_weight here - this is just a mechanical sanity check,
#    not a real training run)
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(logits, y_dummy)
print("loss value:", loss.item())

# 5. Backward pass
loss.backward()
print("backward pass completed without error")

# Extra sanity: confirm gradients actually landed on model params
grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
print(f"{len(grad_norms)} parameter tensors received gradients")
print("sample grad norms:", grad_norms[:3])