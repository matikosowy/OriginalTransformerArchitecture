import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A simple dataset for training (representing some tokenized sequences)
x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).to(device)
y = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]).to(device)


x_vocab_size = 12  # (0 for pad, 11 for EOS)
y_vocab_size = 12
x_pad_tok = 0
y_pad_tok = 0
embed_size = 32
n_heads = 4
n_layers = 2
max_len = 10

# Initialize the model
model = Transformer(x_vocab_size, y_vocab_size, x_pad_tok, y_pad_tok,
                    embed_size=embed_size, n_heads=n_heads, n_layers=n_layers,
                    max_len=max_len, device=device).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=y_pad_tok)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    output = model(x, y[:, :-1])
    output = output.reshape(-1, y_vocab_size)
    target = y[:, 1:].reshape(-1)
    
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_output = model(x, y[:, :-1])
    _, predicted = torch.max(test_output, dim=2)
    print("Input sequences:")
    print(x)
    print("Target sequences:")
    print(y)
    print("Predicted sequences:")
    print(predicted)

# Accuracy check
matches = (predicted == y[:, 1:]).all(dim=1)
print("Predictions match targets:", matches.tolist())
print("Accuracy:", matches.float().mean().item())

# Output:
# Epoch [100/1000], Loss: 0.1910
# Epoch [200/1000], Loss: 0.0655
# Epoch [300/1000], Loss: 0.0280
# Epoch [400/1000], Loss: 0.0224
# Epoch [500/1000], Loss: 0.0069
# Epoch [600/1000], Loss: 0.0090
# Epoch [700/1000], Loss: 0.0061
# Epoch [800/1000], Loss: 0.0035
# Epoch [900/1000], Loss: 0.0022
# Epoch [1000/1000], Loss: 0.0058
# Input sequences:
# tensor([[ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10]])
# Target sequences:
# tensor([[ 2,  3,  4,  5,  6],
#         [ 7,  8,  9, 10, 11]])
# Predicted sequences:
# tensor([[ 3,  4,  5,  6],
#         [ 8,  9, 10, 11]])
# Predictions match targets: [True, True]
# Accuracy: 1.0

# The model is able to predict the target sequences with 100% accuracy.