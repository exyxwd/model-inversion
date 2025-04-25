import torch

def get_confidence_vector(target_model, x):
    with torch.no_grad():
        logits = target_model(x.unsqueeze(0))
        prob = torch.softmax(logits, dim=1).squeeze()
    return prob