# Filename: utils.py
# --- Start of generated code ---
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np # Import numpy for axes reshaping

def add_gaussian_noise(images, sigma=0.1):
    """Adds Gaussian noise to images."""
    noise = torch.randn_like(images) * sigma
    # Clamp to maintain valid image range [0, 1]
    return torch.clamp(images + noise, 0., 1.)

def compute_mse(img1, img2):
    """Computes Mean Squared Error between two images or batches."""
    return F.mse_loss(img1, img2).item()

@torch.no_grad() # Ensure no gradients are computed during evaluation
def compute_error_rate(model, data_loader_or_tensor, labels_tensor, device):
    """
    Computes the classification error rate (mu) of a model.

    Args:
        model: The target model to evaluate.
        data_loader_or_tensor: DataLoader or a Tensor containing the input data (e.g., noisy auxiliary data).
        labels_tensor: Tensor containing the true labels corresponding to the data.
        device: The device ('cuda' or 'cpu') to run evaluation on.

    Returns:
        The classification error rate (float).
    """
    model.eval() # Set model to evaluation mode
    model.to(device)

    correct = 0
    total = 0

    if isinstance(data_loader_or_tensor, torch.utils.data.DataLoader):
        # If input is a DataLoader
        for images, labels in data_loader_or_tensor:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    elif isinstance(data_loader_or_tensor, torch.Tensor):
        # If input is a Tensor (e.g., noisy_aux)
        images = data_loader_or_tensor.to(device)
        labels = labels_tensor.to(device)
        # Process in batches if the tensor is large to avoid memory issues
        batch_size = 128 # Or another suitable batch size
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    else:
        raise TypeError("Input must be a DataLoader or a Tensor")

    if total == 0:
        return 0.0 # Avoid division by zero

    error_rate = 1.0 - (correct / total)
    return error_rate

def plot_comparison(ground_truth, reconstructions_dict, title='Comparison'):
    """Plots ground truth vs reconstructed images for different methods."""
    import matplotlib.pyplot as plt # Keep import local to function

    num_methods = len(reconstructions_dict)
    num_samples = ground_truth.shape[0] # Get number of samples

    fig, axes = plt.subplots(num_methods + 1, num_samples, figsize=(num_samples * 1.5, 2 + num_methods * 1.5)) # Adjust figsize

    # Ensure axes is always 2D, even if num_methods=0 or num_samples=1
    if num_methods == 0: # Only ground truth
         if num_samples == 1:
             axes = np.array([[axes]])
         else:
             axes = axes.reshape((1, num_samples))
    elif num_samples == 1: # Only one sample
        axes = axes.reshape((num_methods + 1, 1))
    # No reshape needed if num_methods > 0 and num_samples > 1, axes is already 2D

    methods = list(reconstructions_dict.keys())

    # Plot Ground Truth row
    for i in range(num_samples):
        ax = axes[0, i] # Use 2D indexing
        ax.imshow(ground_truth[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
        if i == 0: # Add label only to the first column
             ax.set_ylabel("Ground\nTruth", rotation=0, labelpad=30, fontsize=10, va='center')

    # Plot rows for each reconstruction method
    for row, method in enumerate(methods):
        for i in range(num_samples):
            ax = axes[row + 1, i] # Use 2D indexing
            # Detach tensor before moving to CPU and converting to numpy
            recon_img = reconstructions_dict[method][i].squeeze().detach().cpu().numpy()
            ax.imshow(recon_img, cmap='gray')
            ax.axis('off')
            if i == 0: # Add label only to the first column
                ax.set_ylabel(method, rotation=0, labelpad=30, fontsize=10, va='center')

    plt.suptitle(title, fontsize=14)
    # Adjust layout to prevent labels/titles overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
    plt.savefig("comparison_result.png", dpi=200)
    plt.show()
