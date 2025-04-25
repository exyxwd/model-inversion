import torch
import torch.nn as nn

class ShadowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x.view(x.size(0), -1)))

def train_shadow_model(D_aux, D_neg):
    model = ShadowModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    X = torch.cat([D_aux, D_neg])
    y = torch.cat([torch.ones(len(D_aux)), torch.zeros(len(D_neg))])
    for _ in range(100):
        out = model(X)
        loss = criterion(out.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def recover_confidence_vector(shadow_model, mu, target_class, num_classes=10, sigma=0.1):
    with torch.no_grad():
        w = shadow_model.linear.weight.data
        norm = torch.norm(w)
        h = 1 / (1 + torch.exp(sigma * torch.distributions.Normal(0,1).icdf(torch.tensor(mu)) * norm))
        vec = torch.ones(num_classes) * ((1 - h) / (num_classes - 1))
        vec[target_class] = h
    return vec