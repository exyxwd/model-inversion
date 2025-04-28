import torch

def get_score_based_vector(target_model, x, num_classes=10):
    with torch.no_grad():
        logits = target_model(x.unsqueeze(0))
        prob = torch.softmax(logits, dim=1).squeeze()
        max_idx = torch.argmax(prob)
        vec = torch.zeros(num_classes)
        vec[max_idx] = prob[max_idx]
    return vec