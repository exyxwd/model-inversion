# Model inversion

## Reverse Engineering Visual Entities via Adversarial Leakage​ (R.E.V.E.A.L.​)

For documentation, check the `README.md` files in the subfolders:

1. [C2FMI (score-based black-box)](black_box/C2FMI/)
1. [MI on MNIST (label-only)](model_inversion_mnist/)
1. [Simple inversion (gradient)](simple_inversion/)
1. [EMNIST (white-box)](white_box/emnist%20final/)

The `black_box/label_only_old`, `white_box/emnist_gradient_old` and `white_box/mnist_gradient_old` directories are only present for the sake of completeness.

The compiled documentation from the linked `README.md` files is below.

---

# C2FMI - score-based black-box (EXYXWD)

## Description

The code in `/black_box/C2FMI` is based on the paper *"Z. Ye, W, Luo and M. L. Naseem, et al., C2FMI: Corse-to-Fine Black-box Model Inversion Attack", TDSC, 2023.*

The code is modified and adapted on the repository [2022YeC2FMI](https://github.com/MiLabHITSZ/2022YeC2FMI). That repository did not implement the option to only use the top-x prediction scores for the attack, althrough the paper did mention that they experimented with it and also showed figures of it. This version implements that feature too.


## Results

The figure below shows two samples from the training dataset and three reconstructed images for every label. The reconstructed images use the top 1, top 100 and all of the prediction scores in respective order. 

![avatar](/black_box/C2FMI/figures/Comparison.png)

During my experimentations I observed overall a top-5 accuracy of nearly 1.0, over 0.6 attack accuracy and over 0.4 average confidence. The images above have 0.6 attack, 1.0 top-5 accuracy and 0.43, 0.5, 0.72 average confidence using all, top-1 and top-100 predictions respectively.


## Method description

### Overview

The inversion attack implemented is a two-stage process that uses a pre-trained GAN to generate synthetic images and iteratively optimizes the latent space to match the target label's features. The attack is designed to work in a black-box setting, where only the output probabilities or logits of the target model are accessible.

### Components

**Target Model:**
- The target classifier is with backbone MobileNet and was trained on dataset FaceScrub.
- The goal is to reconstruct identities from the FaceScrub dataset, rather than specific images.

**Generative Adversarial Network (GAN):**
- A pre-trained GAN is used to generate synthetic images from latent vectors.
- The GAN's generator maps latent vectors to high-quality images.

**Embedding Model:**
- A pre-trained embedding model is used to extract feature representations of images.
- These features are compared with the target label's features to guide the optimization process.

**Predict-to-Feature Model:**
- A mapping model (predict2feature) that converts the target model's predictions into feature space.
- This helps align the generated images with the target label's features.

### Attack Workflow

#### Stage I: Latent Space Optimization

1. **Initialization:**
    - A batch of latent vectors is initialized based on the mean and standard deviation of the GAN's latent space.

2. **Image Generation:**
    - The latent vectors are passed through the GAN's generator to produce synthetic images.

3. **Prediction and Feature Extraction:**
    - The generated images are resized and passed through the target model to obtain predictions.
    - The predictions are converted into feature space using the predict2feature model.

4. **Loss Calculation:**
    - The generated images are also passed through the embedding model to extract their features.
    - A Mean Squared Error (MSE) loss is computed between the embedding model's features and the target label's features.

5. **Optimization:**
    - The latent vectors are updated using gradient descent to minimize the MSE loss.
    - This process iteratively refines the latent vectors to generate images that better match the target label.

#### Stage II: Differential Evolution (DE) Optimization

1. **Population Initialization:**
    - The optimized latent vectors from Stage I. are used to initialize a population for the DE algorithm.

2. **Evolutionary Optimization:**
    - The DE algorithm iteratively refines the population by generating new candidates through mutation and crossover.
    - Fitness is evaluated based on the target model's confidence in the target label.

3. **Image Selection:**
    - The best latent vector from the final population is used to generate the reconstructed image.


## Requirements

Tested on:

- PyTorch 2.5.1
- CUDA 12.1


## How to setup and run

1. download `checkpoint2.zip` from <https://huggingface.co/MiLab-HITSZ/C2FMI/tree/main>.
2. download `trained_models.zip` from <https://huggingface.co/MiLab-HITSZ/C2FMI/tree/main>.
3. unzip and put these 2 folders in your project directory.
4. running with command:
> python main_attack.py

- note that you should create directory `gen_figures/DE_facescrub/` in your project before running since our code does not automatically generate it.
- changing the variables `init_label` and `fina_label` in `main_attack.py`, attack will start at `init_label` and end at `fina_label`.
- changing the variable `top_x` in `main_attack.py` to any value less than or equal to the total number of classes makes the attack use only the specified number of prediction scores from the highest ones.


## How to evaluate

### The following command can be run to evaluate the attack accuracy:
> python eva_accuracy.py

- The modification of the annotated variables in `eva_accuracy.py` may be needed depending on file structure and the number of images to evaluate.

---

# MI on MNIST - label-only (l90Ohi)

The `/model_inversion_mnist` directory contains code demonstrating a model inversion attack on the MNIST dataset, specifically focusing on the "Label-only" attack method. The goal of this type of attack is to reconstruct or infer information about the original training data using only the predicted labels output by a target machine learning model.

The implementation is based on the concepts presented in the paper: **"Label-only Model Inversion Attack: The Attack that Requires the Least"**.

## Contents

* `model_inversion_mnist_demo.ipynb`: A Jupyter Notebook providing a step-by-step walkthrough of the attack process, including setup, model training, attack execution, and result visualization.
* `data_loader.py`: Python script to load the MNIST dataset.
* `utils.py`: Utility functions for tasks like adding noise, calculating error rates, and plotting results.
* `attacks/`: Directory containing implementations for different attack vector generation methods:
    * `label_only_attack.py`: Implements the core logic for the label-only attack, including shadow model training and confidence vector recovery based on the target model's error rate.
    * `vector_based_attack.py`: Generates vectors using the full confidence score output by the target model.
    * `score_based_attack.py`: Generates vectors using only the highest confidence score.
    * `one_hot_attack.py`: Generates one-hot encoded vectors based on the predicted label.
* `phase1_vector_recovery.py`: Script coordinating the generation of confidence vectors using the methods in the `attacks/` directory.
* `phase2_train_attack_model.py`: Script for training the attack model (generator) that reconstructs images from confidence vectors.
* `phase3_reconstruct.py`: Script used to reconstruct images using the trained attack model and generated vectors.
* `phase4_evaluation.py`: Script intended for evaluating the reconstruction results (currently uses `plot_comparison` from `utils.py`).
* `Label-only Model Inversion Attack The Attack that Requires the Least.pdf`: The reference paper detailing the theory behind the label-only attack.

## Dataset Used

* **MNIST Dataset**: This project uses the standard MNIST dataset of handwritten digits. The dataset consists of 60,000 training images and 10,000 testing images, each being a 28x28 grayscale image representing a digit from 0 to 9. The code automatically downloads the dataset using torchvision if it's not present locally in the `./data` directory (requires an internet connection the first time).

## How to Run the Code

The primary way to run this code and see the results is through the Jupyter Notebook (`model_inversion_mnist_demo.ipynb`).

**Prerequisites:**

* Python 3.x installed.
* `pip` (Python package installer) available.
* Jupyter Notebook or JupyterLab environment installed. (You can install it via pip: `pip install jupyterlab notebook`)

**Steps:**

1.  **Get the Code:** Clone this repository or download the code files into a directory on your local machine.
2.  **Navigate to Directory:** Open a terminal or command prompt and change directory (`cd`) into the folder containing the downloaded files (e.g., `cd path/to/model_inversion_mnist`).
3.  **Install Dependencies:** Run the following command in your terminal to install the necessary Python libraries. The notebook also contains cells (Cells 1-5) to perform these installations. It's recommended to run this in the terminal beforehand:
    ```bash
    pip install torch torchvision matplotlib ipykernel
    ```
    * `torch` and `torchvision`: For deep learning models and dataset handling.
    * `matplotlib`: For plotting the results.
    * `ipykernel`: Allows Jupyter to use your Python environment and installed libraries correctly.
    *(Note: The notebook also includes commands to set up a specific Jupyter kernel named "torch_env". This is optional but can help manage dependencies if you work on multiple projects.)*
4.  **Launch Jupyter:** Start the Jupyter environment by running one of the following commands in your terminal:
    * For JupyterLab (recommended): `jupyter lab`
    * For classic Jupyter Notebook: `jupyter notebook`
    This should open a new tab in your web browser showing the Jupyter interface.
5.  **Open the Notebook:** In the Jupyter browser interface, navigate to and click on the `model_inversion_mnist_demo.ipynb` file to open it.
6.  **Select Kernel (if prompted):** If asked to select a kernel, choose the Python 3 environment where you installed the dependencies, or the "Python (with torch)" kernel if you ran the kernel setup cells.
7.  **Run Cells Sequentially:** Execute the notebook cells one by one, starting from the top. You can run a cell by selecting it and pressing `Shift + Enter` or by clicking the "Run" button in the toolbar.
    * **Wait for Completion:** Some cells, especially those involving model training (Target Model, Shadow Model, Attack Models), might take some time to complete. Wait for a cell to finish (indicated by the `[*] ` turning into `[number]`) before running the next one.
    * **Outputs:** Observe the outputs generated below each cell. These include installation logs, training progress, and finally, the comparison plot of reconstructed images.
    * **Internet Connection:** The first time you run the cell that loads the MNIST data (Cell 8), it will download the dataset, requiring an internet connection.

## Code Description: Detailed Notebook Steps

The `model_inversion_mnist_demo.ipynb` notebook is structured to guide you through the process. Here's a more detailed breakdown of what each major part does:

1.  **Step 1: Setup and Install Dependencies (Cells 1-7):**
    * These initial cells use `pip` to install the required libraries (`torch`, `torchvision`, `matplotlib`, `ipykernel`).
    * They check the installed PyTorch version.
    * Optionally, they set up a dedicated Jupyter kernel.
    * Finally, they import necessary Python modules (`sys`, `os`, `torch`, `optim`, etc.) and utility functions from the `.py` files in the directory.

2.  **Step 2: Load MNIST Dataset (Cell 8):**
    * Calls the `load_mnist_data` function from `data_loader.py`.
    * This uses `torchvision.datasets.MNIST` to download (if needed) and load the training and testing sets.
    * Prints the number of samples in each set.

3.  **Step 3: Build and Train Target Model (Cell 9):**
    * Defines a `TargetCNN` class (a Convolutional Neural Network architecture suitable for MNIST).
    * Initializes this model and moves it to the appropriate device (GPU if available, otherwise CPU).
    * Sets up an optimizer (Adam) and loss function (CrossEntropyLoss).
    * Trains the `TargetCNN` on the *entire* MNIST training set for a few epochs. This model simulates the pre-trained model that the attacker targets.
    * Prints training progress (loss and accuracy).
    * Sets the model to evaluation mode (`.eval()`) after training.

4.  **Step 4: Prepare Sample Inputs (Cell 10):**
    * Selects the first 10 images and their corresponding labels from the MNIST *test set*.
    * Stacks these images into a tensor `images` and labels into a tensor `labels`. These 10 samples will be used later to demonstrate the reconstruction quality.

5.  **Step 5: Generate Confidence Vectors for Test Samples (Phase 1 - Cell 11):**
    * **Objective:** Generate the input vectors (confidence vectors) for the 10 test samples, using the four different attack assumptions (label-only, vector-based, score-based, one-hot). These vectors will later be fed into the trained attack models for reconstruction.
    * **Auxiliary Data Prep:** Selects subsets of the *training* data to serve as `aux_images` (positive samples) and `d_neg` (negative samples). These are needed *only* for the label-only method's setup.
    * **Shadow Model Training:** Trains the simple linear `ShadowModel` (from `attacks.label_only_attack.py`) using `aux_images` and `d_neg`. This happens on the CPU.
    * **Error Rate (μ) Calculation:** Creates a noisy version of `aux_images` by adding Gaussian noise (`utils.add_gaussian_noise`). Calculates the target model's error rate (μ) on this *noisy* auxiliary data using `utils.compute_error_rate`. This μ is crucial for the label-only vector recovery.
    * **Label-Only Vector Recovery:** For each of the 10 *test* images, it calls `phase1_vector_recovery.generate_confidence_vectors` with `method="label_only"`. This internally uses `attacks.label_only_attack.recover_confidence_vector`, which takes the trained `shadow_model`, the calculated `mu`, the noise `sigma`, and the true `label` of the test image to estimate the confidence vector.
    * **Baseline Vector Generation:** Calls `phase1_vector_recovery.generate_confidence_vectors` again for methods "vector_based", "score_based", and "one_hot". These methods directly query the *target model* with the *test image* to get the required vector (full softmax output, score-only, or one-hot prediction).
    * **Store & Check:** Stores the generated vectors for the 10 test samples in the `vectors_by_method` dictionary. Prints the generated vector for the first test sample for each method as a sanity check.

6.  **Step 6: Generate Training Data for Attack Models (Cell 12):**
    * **Objective:** Create a larger dataset to train the actual image reconstruction models (Attack Models). This dataset consists of (confidence vector, original image) pairs.
    * **Process:** It iterates through the *auxiliary dataset* (`aux_images`). For each auxiliary image:
        * It generates the corresponding confidence vector using each of the four methods (label-only calculation using `mu` and `shadow_model`, and the other three by querying the target model with the auxiliary image).
        * It pairs the generated vector with the *original auxiliary image*.
    * **Output:** Creates the `attack_train_data` dictionary, where each key (e.g., "label_only") holds a list of vectors and a corresponding list of images, ready for training the attack models.

7.  **Step 7: Train Attack Models (Phase 2 - Cell 13):**
    * **Objective:** Train the models that perform the actual image reconstruction.
    * **Process:** For each method ("label_only", "vector_based", etc.):
        * It retrieves the training vectors and corresponding target images generated in Cell 12 (`attack_train_data[method]`).
        * It calls `phase2_train_attack_model.train_attack_model`. This function initializes the `AttackModel` (a deconvolutional/generator network defined in `phase2_train_attack_model.py`) and trains it using the provided vectors (as input) and images (as target output). The model learns to map the specific type of confidence vector back to a plausible image.
    * **Output:** Stores the four trained attack models in the `attack_models` dictionary.

8.  **Step 8: Reconstruct Test Images and Compare Results (Phase 3 & 4 - Cells 14-17):**
    * **Objective:** Use the trained attack models to reconstruct the 10 *test images* (from Cell 10) and visualize the results.
    * **Reload Modules (Cells 14-15):** Ensures the latest versions of the reconstruction and utility scripts are used (useful during development).
    * **Reconstruction (Cell 17):**
        * Retrieves the *test vectors* generated way back in Cell 11 (`vectors_by_method`).
        * Retrieves the *trained attack models* from Cell 13 (`attack_models`).
        * For each method, it feeds the 10 test vectors into the corresponding trained attack model using `phase3_reconstruct.reconstruct_images`.
        * Stores the 10 reconstructed images for each method in the `results` dictionary.
    * **Evaluation/Plotting (Cell 17):**
        * Calls the `utils.plot_comparison` function.
        * This function takes the original 10 *test images* (ground truth) and the `results` dictionary (containing reconstructed images for each method).
        * It generates and displays a plot comparing the ground truth images side-by-side with the reconstructions from the label-only, vector-based, score-based, and one-hot attacks. This visually demonstrates the effectiveness of each attack method.

## References

* **Primary Paper:** Ye, D., Liu, B., Zhu, T., Zhou, S., & Zhou, W. (2022). Label-only Model Inversion Attack: The Attack that Requires the Least Information. *arXiv preprint arXiv:2203.06555*. (Included as `Label-only Model Inversion Attack The Attack that Requires the Least.pdf`)
* **MNIST Dataset:** LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). The MNIST database of handwritten digits. *http://yann.lecun.com/exdb/mnist/*
* **PyTorch:** Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32*.
* **Torchvision:** Part of the PyTorch project. *https://pytorch.org/vision/stable/index.html*
* **Matplotlib:** Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.

---

# Simple model inversion - gradient (A9CQZ0)

The notebooks in `/simple_inversion` demonstrate the original method proposed by Fredrikson et al in 2015 (see Paper section below).

## How to run

Open the `simplemodel.ipynb` Jupyter notebook and run the cells in order. The first code cell will install packages, but the requirements.txt file could be used instead (`pip install -r requirements.txt`).

There is also a torch version (`simplemodel_torch.ipynb`), less documented but similarly functional. Produces slightly different results.

The notebooks were tested on Python 3.11.9. A virtual environment is recommended.

## Dataset

The training and testing data is available in the faces directory. It consists of greyscale images in `pgm` format. It is small enough to run on CPU, adequate for demonstration purposes.

You can check out `simplemodel_torch_celeba.ipynb` which uses the bigger and colorful CelebA dataset (downloaded automatically), but due to technical constraints, I couldn't manage to make it work.

## References

The following GitHub repositories were used to help the implementation:

* [sarahsimionescu/simple-model-inversion](https://github.com/sarahsimionescu/simple-model-inversion)
* [Djiffit/face-decoding-with-model-inversion](https://github.com/Djiffit/face-decoding-with-model-inversion/blob/054bc93fbe405381564dc0fac50d94783c6b385e/inversion.ipynb)

## Paper

Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. 2015. Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS '15). Association for Computing Machinery, New York, NY, USA, 1322–1333. https://doi.org/10.1145/2810103.2813677

---

# EMNIST White-Box Gradient-Based Attack with SimpleCNN (RFTP59)

The `/white_box/emnist final` directory contains a Jupyter notebook (`emnist_final.ipynb`) that implements a white-box gradient-based attack to reconstruct letter images from the EMNIST dataset using a Simple Convolutional Neural Network (CNN). The attack assumes full access to the model’s architecture, weights, and gradients, optimizing random noise to match target letters (A-Z) by minimizing classification, perceptual, and total variation losses. Metrics such as SSIM, MSE, and model confidence are computed to evaluate the reconstructions.

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
- Zhu et al. (2019). Deep Leakage from Gradients. [arXiv:1906.08935](https://arxiv.org/abs/1906.08935)
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
