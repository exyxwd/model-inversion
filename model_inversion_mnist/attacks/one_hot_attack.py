import torch

def get_one_hot_vector(target_model, x, num_classes=10):
    with torch.no_grad():
        pred = torch.argmax(target_model(x.unsqueeze(0)), dim=1).item()
        vec = torch.zeros(num_classes)
        vec[pred] = 1.0
    return vec