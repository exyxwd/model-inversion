"""
### Black Box Label-only Model Inversion Attack on the EMNIST Dataset

This script demonstrates a **model inversion attack** on a trained neural network model. The goal of the attack is to reconstruct input images based on the model's output probabilities (i.e., the confidence vector). The dataset used is the EMNIST dataset (Extended MNIST), specifically the balanced version containing capital letters A-Z.

Steps of the attack:
1. **Loading and Preprocessing the Dataset**: The EMNIST dataset is loaded, filtered to include only capital letters (A-Z), and split into training and test datasets. It then gets split again into two parts: one for the target model and one for the attack.
2. **Target Model Training**: A target model (a Convolutional Neural Network) is trained on the first part of the training data to classify the letters A-Z.
3. **Attacker Setup**: The second part of the training data (the auxiliary set) is used to estimate the target model's behavior and create a shadow model to help guide the attack.
4. **Model Inversion**: Based on the confidence vector from the target model, an attack model is trained to reconstruct the original images corresponding to each label (A-Z).
5. **Reconstruction and Visualization**: The reconstructed images are compared to the ground truth images to visualize the effectiveness of the attack.

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import random
import tensorflow_datasets as tfds

# --- Step 1: Load and Process EMNIST Dataset ---
ds_train = tfds.load('emnist/balanced', split='train', as_supervised=True)
ds_test = tfds.load('emnist/balanced', split='test', as_supervised=True)

def convert_to_numpy(ds):
    x, y = [], []
    for img, label in tfds.as_numpy(ds):
        x.append(np.array(img))
        y.append(label)
    x = np.array(x).astype('float32') / 255.
    x = np.expand_dims(x, -1)
    y = np.array(y)
    return x, y

x_train_all, y_train_all = convert_to_numpy(ds_train)
x_test_all, y_test_all = convert_to_numpy(ds_test)

# Keep only capital letters A-Z: 10-35
def filter_letters(x, y):
    mask = (y >= 10) & (y <= 35)
    return x[mask], y[mask] - 10

x_train_all, y_train_all = filter_letters(x_train_all, y_train_all)
x_test_all, y_test_all = filter_letters(x_test_all, y_test_all)
num_classes = 26

np.random.seed(42)
perm = np.random.permutation(len(x_train_all))
split = len(x_train_all) // 2

x_target_train, y_target_train = x_train_all[perm[:split]], y_train_all[perm[:split]]
x_aux, y_aux = x_train_all[perm[split:]], y_train_all[perm[split:]]

# --- Step 2: Define and Train Target Model ---
def create_target_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model

T = create_target_model()
T.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])
T.fit(x_target_train, y_target_train, epochs=3, batch_size=64, verbose=1)

# --- Step 3: Attacker Setup ---
def estimate_error_rate(model, D_aux, label, sigma=0.3):
    noisy = D_aux + sigma * np.random.randn(*D_aux.shape)
    noisy = np.clip(noisy, 0, 1)
    logits = model.predict(noisy, verbose=0)
    preds = np.argmax(logits, axis=1)
    return 1 - (np.sum(preds == label) / len(preds))

def compute_distance_from_error(mu, sigma=0.3):
    return -sigma * norm.ppf(mu)

def create_attack_model():
    model = models.Sequential([
        layers.Input(shape=(num_classes,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(7*7*32, activation='relu'),
        layers.Reshape((7, 7, 32)),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')
    ])
    return model

# --- Step 4: Attack Model Train ---
def train_attack_model(attack_model, confidence_vector, D_aux, epochs=200):
    for _ in range(epochs):
        i = np.random.randint(0, len(D_aux))
        img = np.expand_dims(D_aux[i], axis=0)
        vec = np.expand_dims(confidence_vector, axis=0)
        attack_model.train_on_batch(vec, img)

# --- Step 5: Reconstruction and Visualization ---
label_to_letter = {i: chr(ord('A') + i) for i in range(num_classes)}

num_letters_per_row = 6
num_pairs = (num_classes + num_letters_per_row - 1) // num_letters_per_row
fig, axs = plt.subplots(num_pairs, num_letters_per_row * 2, figsize=(2 * num_letters_per_row * 2, 2.5 * num_pairs))
fig.suptitle("Ground Truth (Left) and Reconstructed (Right) Letters", fontsize=16)

def fix_emnist_image(img):
    return np.transpose(img, (1, 0))

for label in range(num_classes):
    print(f"\n🔍 Reconstructing letter: {label_to_letter[label]}")
    aux_indices = np.where(y_aux == label)[0][:100]
    if len(aux_indices) == 0:
        print(f"⚠️  No auxiliary data for label {label_to_letter[label]}")
        continue
    D_aux = x_aux[aux_indices]

    ground_truth_mean = np.mean(x_target_train[y_target_train == label], axis=0).squeeze()
    error_rate = estimate_error_rate(T, D_aux, label)
    d = compute_distance_from_error(error_rate)

    def get_shadow_data(D_aux, n_neg=100):
        X_pos = D_aux.reshape(len(D_aux), -1)
        y_pos = np.ones(len(D_aux))
        neg_samples = []
        while len(neg_samples) < n_neg:
            i = random.randint(0, len(x_aux) - 1)
            if y_aux[i] != label:
                neg_samples.append(x_aux[i])
        X_neg = np.array(neg_samples).reshape(n_neg, -1)
        y_neg = np.zeros(n_neg)
        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))
        return X, y

    X_shadow, y_shadow = get_shadow_data(D_aux)
    shadow_model = LogisticRegression(max_iter=1000)
    shadow_model.fit(X_shadow, y_shadow)

    w = shadow_model.coef_[0]
    norm_w = np.linalg.norm(w)
    score = 1 / (1 + np.exp(d * norm_w))

    confidence_vector = np.ones(num_classes) * ((1 - score) / (num_classes - 1))
    confidence_vector[label] = score
    confidence_vector = confidence_vector.astype('float32')

    attack_model = create_attack_model()
    attack_model.compile(optimizer='adam', loss='mse')
    train_attack_model(attack_model, confidence_vector, D_aux)

    recon = attack_model.predict(np.expand_dims(confidence_vector, axis=0), verbose=0)[0, :, :, 0]

    row = label // num_letters_per_row
    col_base = (label % num_letters_per_row) * 2

    axs[row, col_base].imshow(fix_emnist_image(ground_truth_mean), cmap='gray')
    axs[row, col_base].set_title(f"{label_to_letter[label]}\nGT")
    axs[row, col_base].axis('off')

    axs[row, col_base + 1].imshow(fix_emnist_image(recon), cmap='gray')
    axs[row, col_base + 1].set_title("Recon")
    axs[row, col_base + 1].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
