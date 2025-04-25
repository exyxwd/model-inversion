import torch
import torch.nn as nn

class AttackModel(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x).view(-1, 1, 28, 28)

def train_attack_model(vectors, images, epochs=100):
    model = AttackModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        outputs = model(vectors)
        loss = loss_fn(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model