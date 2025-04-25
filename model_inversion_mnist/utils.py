import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def add_gaussian_noise(images, sigma=0.1):
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0., 1.)

def compute_mse(img1, img2):
    return F.mse_loss(img1, img2).item()

def plot_comparison(ground_truth, reconstructions_dict, title='Comparison'):
    import matplotlib.pyplot as plt
    import numpy as np

    num_methods = len(reconstructions_dict)
    fig, axes = plt.subplots(num_methods + 1, 10, figsize=(15, 2 + num_methods))
    
    # đảm bảo axes luôn là 2D (dù chỉ có 1 row)
    if isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.reshape((num_methods + 1, 10))

    methods = list(reconstructions_dict.keys())

    # vẽ hàng Ground Truth
    for i in range(10):
        axes[0][i].imshow(ground_truth[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0][i].axis('off')
    axes[0][0].set_ylabel("Ground\nTruth", rotation=0, labelpad=30, fontsize=10, va='center')

    # vẽ các hàng của từng phương pháp
    for row, method in enumerate(methods):
        for i in range(10):
            axes[row + 1][i].imshow(
                reconstructions_dict[method][i].squeeze().detach().cpu().numpy(),
                cmap='gray')
            axes[row + 1][i].axis('off')
        axes[row + 1][0].set_ylabel(method, rotation=0, labelpad=30,
                                     fontsize=10, va='center')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("comparison_result.png", dpi=200)
    plt.show()
