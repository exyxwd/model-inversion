# Filename: attacks/label_only_attack.py
# --- Start of generated code ---
import torch
import torch.nn as nn
import torch.optim as optim # Added optimizer import
from torch.utils.data import TensorDataset, DataLoader # Added DataLoader imports

# Define the ShadowModel architecture (Linear as per paper)
class ShadowModel(nn.Module):
    def __init__(self, input_dim=28*28): # Make input_dim configurable if needed
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Ensure input is flattened
        return torch.sigmoid(self.linear(x.view(x.size(0), -1)))

def train_shadow_model(D_aux, D_neg, epochs=100, batch_size=32, lr=0.01):
    """
    Trains the Shadow Model.

    Args:
        D_aux (Tensor): Auxiliary data (positive samples, label 1).
        D_neg (Tensor): Negative data (negative samples, label 0).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.

    Returns:
        ShadowModel: The trained shadow model.
    """
    # Infer input dimension based on D_aux shape
    if D_aux.ndim == 4: # Assuming images like (N, C, H, W)
        input_dim = D_aux.shape[1] * D_aux.shape[2] * D_aux.shape[3]
    elif D_aux.ndim == 2: # Assuming flattened data like (N, Features)
        input_dim = D_aux.shape[1]
    else:
        raise ValueError("Unsupported D_aux dimensions")

    model = ShadowModel(input_dim=input_dim)
    model.cpu() # Train shadow model on CPU as it's small

    # Ensure data is on CPU and float
    D_aux = D_aux.cpu().float()
    D_neg = D_neg.cpu().float()

    # Create labels
    y_aux = torch.ones(len(D_aux), 1, dtype=torch.float32)
    y_neg = torch.zeros(len(D_neg), 1, dtype=torch.float32)

    # Combine datasets and labels
    X = torch.cat([D_aux, D_neg])
    y = torch.cat([y_aux, y_neg])

    # Create DataLoader
    dataset = TensorDataset(X.view(X.size(0), -1), y) # Flatten input for Linear layer training
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss() # Binary Cross Entropy for binary classification

    print("Training Shadow Model...")
    model.train() # Set model to training mode
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x) # Model expects flattened input now
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        # Optional: Print loss periodically
        if (epoch + 1) % 20 == 0:
             avg_loss = epoch_loss / len(dataset)
             print(f"  Shadow Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Shadow Model training finished.")
    model.eval() # Set back to evaluation mode
    return model.cpu() # Return model on CPU

def recover_confidence_vector(shadow_model, mu, target_class, num_classes=10, sigma=0.1):
    """
    Recovers the confidence vector using the shadow model, mu, and sigma.
    Based on Equation 6 from the paper.

    Args:
        shadow_model (ShadowModel): The trained linear shadow model (expected on CPU).
        mu (float): The pre-calculated error rate of the target model on noisy auxiliary data.
        target_class (int): The target class label for which to reconstruct the vector.
        num_classes (int): The total number of classes.
        sigma (float): The standard deviation of Gaussian noise used to calculate mu.

    Returns:
        Tensor: The recovered confidence vector (on CPU).
    """
    shadow_model.cpu() # Ensure shadow model is on CPU
    shadow_model.eval()

    if not (0 < mu < 1):
       # Clamp mu slightly away from 0 and 1 to avoid issues with icdf/ppf
       mu = max(1e-9, min(mu, 1.0 - 1e-9))
       # print(f"Warning: mu value was outside (0, 1), clamped to {mu}") # Optional warning

    with torch.no_grad():
        # Get weights (ensure it's on CPU)
        w = shadow_model.linear.weight.data.cpu()
        norm_w = torch.norm(w)

        if norm_w == 0: # Avoid division by zero or issues if weights are zero
            print("Warning: Shadow model weights norm is zero. Returning uniform distribution.")
            h = 1.0 / num_classes
        else:
            # Calculate Phi^{-1}(mu) using the inverse CDF of standard normal N(0,1)
            try:
                # Use torch.distributions if available and on CPU
                phi_inv_mu = torch.distributions.Normal(0, 1).icdf(torch.tensor(mu, dtype=torch.float32, device='cpu'))
            except AttributeError:
                # Fallback to scipy if torch.distributions.Normal doesn't have icdf
                try:
                    from scipy.stats import norm
                    phi_inv_mu = norm.ppf(mu)
                except ImportError:
                     raise ImportError("Scipy is required for norm.ppf if torch.distributions.Normal.icdf is not available.")

            # Calculate h(x*) using Equation 6 from the paper
            # h(x*) = 1 / (1 + exp(sigma * Phi^{-1}(mu) * ||w||_2))
            # Make sure calculations happen with float tensors/scalars
            phi_inv_mu_tensor = torch.tensor(phi_inv_mu, dtype=torch.float32, device='cpu') # Ensure tensor
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device='cpu')

            exp_term = torch.exp(sigma_tensor * phi_inv_mu_tensor * norm_w)
            h = 1.0 / (1.0 + exp_term)
            h = h.item() # Convert to float

        # Ensure h is within [0, 1] bounds
        h = max(0.0, min(h, 1.0))

        # Create the confidence vector
        if num_classes <= 1:
            # Handle edge case of 1 class
            vec = torch.tensor([1.0], dtype=torch.float32) if target_class == 0 else torch.tensor([0.0], dtype=torch.float32)
        else:
            vec = torch.ones(num_classes, dtype=torch.float32) * ((1.0 - h) / (num_classes - 1))
            if 0 <= target_class < num_classes:
                vec[target_class] = h
            else:
                print(f"Warning: target_class {target_class} is out of bounds for num_classes {num_classes}. Returning uniform vector.")
                vec = torch.ones(num_classes, dtype=torch.float32) / num_classes

        # Normalize vector to ensure it sums to 1 (due to potential float precision issues or clamping)
        vec_sum = torch.sum(vec)
        if vec_sum > 1e-6: # Avoid division by zero if vector is all zeros
             vec = vec / vec_sum
        else:
             print("Warning: Recovered vector sum is close to zero. Returning uniform vector.")
             vec = torch.ones(num_classes, dtype=torch.float32) / num_classes


    return vec.cpu() # Return vector on CPU

# --- End of generated code ---