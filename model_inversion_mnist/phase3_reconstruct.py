import torch


def reconstruct_images(model, vectors):
    model.eval()
    with torch.no_grad():
        return model(vectors)