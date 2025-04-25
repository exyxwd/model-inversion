# Filename: phase1_vector_recovery.py
# --- Start of generated code ---
import torch
from attacks.label_only_attack import recover_confidence_vector
from attacks.vector_based_attack import get_confidence_vector
from attacks.score_based_attack import get_score_based_vector
from attacks.one_hot_attack import get_one_hot_vector

def generate_confidence_vectors(images, labels, target_model,
                                method="label_only",
                                shadow_model=None, # Required for label_only
                                mu=None,           # Required for label_only, calculated beforehand
                                sigma=None,        # Required for label_only, should be consistent
                                num_classes=10,    # Added num_classes parameter
                                device='cpu'):     # Added device parameter
    """
    Generates confidence vectors based on the specified attack method.

    Args:
        images (Tensor): Batch of input images.
        labels (Tensor): Batch of corresponding true labels.
        target_model (nn.Module): The target model.
        method (str): The attack method ('label_only', 'vector_based', etc.).
        shadow_model (nn.Module, optional): Trained shadow model (needed for 'label_only').
        mu (float, optional): Pre-calculated error rate (needed for 'label_only').
        sigma (float, optional): Sigma value used for noise (needed for 'label_only').
        num_classes (int): Number of classes in the classification task.
        device (str): Device to run target_model on ('cuda' or 'cpu').

    Returns:
        Tensor: Stacked confidence vectors.
        Tensor: Original images (targets for the attack model).
    """
    vectors = []
    target_model.to(device) # Ensure target model is on the correct device
    target_model.eval()     # Set target model to eval mode

    images_cpu = images.cpu() # Process images/labels on CPU for iteration
    labels_cpu = labels.cpu()

    with torch.no_grad(): # Disable gradients for vector generation
        for i in range(len(images_cpu)):
            x = images_cpu[i].to(device) # Move single image to device for model inference
            y_label = labels_cpu[i].item() # Get the integer label

            if method == "label_only":
                if shadow_model is None or mu is None or sigma is None:
                    raise ValueError("shadow_model, mu, and sigma are required for label_only method")
                # Use the pre-calculated mu and consistent sigma
                vec = recover_confidence_vector(shadow_model, mu, y_label, num_classes=num_classes, sigma=sigma)
            elif method == "vector_based":
                vec = get_confidence_vector(target_model, x)
            elif method == "score_based":
                vec = get_score_based_vector(target_model, x, num_classes=num_classes)
            elif method == "one_hot":
                vec = get_one_hot_vector(target_model, x, num_classes=num_classes)
            else:
                raise ValueError(f"Invalid method: {method}")

            vectors.append(vec.cpu()) # Collect vectors on CPU

    # Return the vectors and the original images (which act as targets for the attack model)
    return torch.stack(vectors), images # images tensor should be the target

