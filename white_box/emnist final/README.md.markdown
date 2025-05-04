# EMNIST White-Box Gradient-Based Attack with SimpleCNN

This repository contains a Jupyter notebook (`emnist_final.ipynb`) that implements a white-box gradient-based attack to reconstruct letter images from the EMNIST dataset using a Simple Convolutional Neural Network (CNN). The attack assumes full access to the model’s architecture, weights, and gradients, optimizing random noise to match target letters (A-Z) by minimizing classification, perceptual, and total variation losses. Metrics such as SSIM, MSE, and model confidence are computed to evaluate the reconstructions.

## Dataset

The project uses the **EMNIST dataset** (Extended MNIST), specifically the `letters` split, which contains 28x28 grayscale images of handwritten letters (A-Z). The dataset is accessed via `torchvision.datasets.EMNIST`.

- **Reference**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: An extension of MNIST to handwritten letters. *arXiv preprint arXiv:1702.05373*. https://arxiv.org/abs/1702.05373
- **Source**: The dataset is downloaded automatically by `torchvision` when the code is run.

## Prerequisites

To run the notebook, you need Python 3.8+ and the dependencies listed in `requirements.txt`. The notebook is optimized for a Jupyter environment with GPU support. **Google Colab** is recommended for its free GPU access and pre-installed packages, but it can also be run locally.

## Getting Started

Follow these steps to run the project in either Google Colab or a local environment:

### Option 1: Google Colab (Recommended)

1. **Open the Notebook in Colab**:
   - Upload `emnist_final.ipynb` to [Google Colab](https://colab.research.google.com/) or open it directly from GitHub:
     ```markdown
     [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<repository-name>/blob/main/emnist_final.ipynb)
     ```
     *Note*: Replace `<your-username>/<repository-name>` with your GitHub repository URL after creating it.
   - Enable GPU: Go to Runtime > Change runtime type > Select GPU (e.g., T4).

2. **Install Dependencies**:
   - Colab has many packages pre-installed, but you need to ensure the correct versions. Upload `requirements.txt` to Colab (via the file upload button) and run:
     ```bash
     !pip install -r requirements.txt
     ```
   - Alternatively, add this code cell at the start of the notebook:
     ```python
     !pip install torch==2.3.0 torchvision==0.18.0 pandas==2.2.2 numpy==1.26.4 matplotlib==3.8.4 scikit-image==0.23.2
     ```

3. **Clone the Repository (Optional)**:
   - To access files directly in Colab, clone the repository:
     ```bash
     !git clone https://github.com/<your-username>/<repository-name>.git
     %cd <repository-name>
     ```

4. **Configure the Notebook**:
   - In the second code cell, set the `letters` variable:
     - Single letter: `letters = 'A'`
     - Multiple letters: `letters = ['A', 'B', 'C']`
     - All letters (A-Z): `letters = None`
     - Example:
       ```python
       letters = ['A', 'B', 'C']  # Process letters A, B, and C
       ```

5. **Run the Notebook**:
   - Execute all cells (`Run All` in Colab).
   - Outputs will be saved to `/content/` (e.g., `/content/reconstructed_letter_A.png`).

### Option 2: Local Environment

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   cd <repository-name>
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Configure and Run**:
   - Open `emnist_final.ipynb`, set the `letters` variable (as above), and run all cells.
   - Outputs will be saved to the project directory (e.g., `./reconstructed_letter_A.png`).

### Outputs (Both Environments)

- **Progression Plots**: Reconstruction progress at steps [0, 50, 100, 200, 300, 500, 1000, 2000, 4000, 8000, 10000] saved as `reconstructed_letter_<letter>.png`.
- **Final Reconstructed Image**: Image at step 10,000 saved as `reconstructed_<letter>_step_10000.png`.
- **Side-by-Side Comparison**: If multiple letters are processed, saved as `original_vs_reconstructed_all_letters.png`.
- **Results CSV**: Metrics (SSIM, MSE, Confidence, etc.) saved to `inversion_results_all_letters.csv`.
- Console output includes metrics tables and file paths.

### Optional: Pre-Trained Model

- The notebook trains the model if `emnist_cnn.pth` is absent. To skip training, place `emnist_cnn.pth` in the repository root.
- If available, download the pre-trained model from [insert link, e.g., Google Drive] and place it in the project directory.

## How the Code Works

The `emnist_final.ipynb` notebook implements a white-box gradient-based attack with the following steps:

1. **Setup**:
   - Imports libraries (`torch`, `torchvision`, `pandas`, etc.).
   - Sets the device (GPU/CPU) and `letters` variable.

2. **Load Dataset**:
   - Downloads EMNIST `letters` split (train/test) with `torchvision`.
   - Normalizes images to `[0, 1]`.

3. **Define Model**:
   - Implements `SimpleCNN` with two convolutional layers, max-pooling, and two fully connected layers.
   - Provides `get_features` for perceptual loss, leveraging full model access.

4. **Train/Load Model**:
   - Trains `SimpleCNN` for 10 epochs if `emnist_cnn.pth` is missing, or loads it.

5. **Utility Functions**:
   - `compute_metrics`: Computes SSIM, MSE, predicted label, and confidence.
   - `total_variation_loss`: Encourages smoothness in reconstructed images.

6. **White-Box Gradient-Based Attack**:
   - For each letter:
     - Selects one target image.
     - Optimizes random noise over 10,000 steps using AdamW, leveraging full access to model gradients and features.
     - Minimizes a combined loss: cross-entropy (classification), perceptual (weight=1.5, based on CNN features), and total variation (weight=0.01, for smoothness).
     - Logs metrics and snapshots at key steps.

7. **Visualization**:
   - Saves progression plots, final images, and side-by-side comparisons (for multiple letters).

8. **Save Results**:
   - Exports metrics to `inversion_results_all_letters.csv`.

## References

### Papers
- Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: An extension of MNIST to handwritten letters. *arXiv preprint arXiv:1702.05373*. https://arxiv.org/abs/1702.05373
- Mahendran, A., & Vedaldi, A. (2015). Understanding deep image representations by inverting them. *CVPR 2015*. https://arxiv.org/abs/1412.0035
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for real-time style transfer and super-resolution. *ECCV 2016*. https://arxiv.org/abs/1603.08155

### Resources
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Torchvision Datasets: https://pytorch.org/vision/stable/datasets.html#emnist
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html
- Scikit-Image SSIM: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

## Notes
- **Reproducibility**: Reconstructions may vary due to random noise initialization. Add `torch.manual_seed(42)` at the start of the notebook for consistent results.
- **Loss Tuning**: Adjust `lambda_perceptual` (1.0–2.0) or `lambda_tv` (0.005–0.1) for better reconstructions.
- **Path Handling**: Outputs save to `/content/` in Colab or the project directory locally. For local compatibility, consider modifying the notebook to use dynamic paths (e.g., `os.path.join(os.getcwd(), 'reconstructed_letter_A.png')`).
- **Colab Advantage**: Use Colab for free GPU support, ideal for faster training and attack optimization.
- **GitHub Setup**: After creating the repository, update the Colab badge URL in this README.