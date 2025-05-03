# Model Inversion on MNIST using Label-only Attack

This directory contains code demonstrating a model inversion attack on the MNIST dataset, specifically focusing on the "Label-only" attack method. The goal of this type of attack is to reconstruct or infer information about the original training data using only the predicted labels output by a target machine learning model.

This implementation is based on the concepts presented in the paper: **"Label-only Model Inversion Attack: The Attack that Requires the Least"**.

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
