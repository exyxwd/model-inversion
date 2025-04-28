import torch
from data_loader import load_mnist_data
from phase1_vector_recovery import generate_confidence_vectors
from phase2_train_attack_model import train_attack_model
from phase3_reconstruct import reconstruct_images
from phase4_evaluation import evaluate_reconstructions
from attacks.label_only_attack import train_shadow_model
from utils import add_gaussian_noise

methods = ["label_only", "vector_based", "score_based", "one_hot"]

train_set, test_set = load_mnist_data()
samples = [test_set[i] for i in range(10)]
images = torch.stack([img for img, _ in samples])
labels = torch.tensor([label for _, label in samples])

class DummyTargetModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.fc = torch.nn.Linear(28*28, 10)
    def forward(self, x): return self.fc(x.view(x.size(0), -1))

target_model = DummyTargetModel()
aux_images = torch.stack([train_set[i][0] for i in range(100)])
d_neg = torch.stack([train_set[i][0] for i in range(1000, 1100)])
noisy_aux = add_gaussian_noise(aux_images)
shadow_model = train_shadow_model(aux_images, d_neg)

results = {}
for method in methods:
    vecs = generate_confidence_vectors(images, labels, target_model, method, shadow_model, mu=0.1)
    attack_model = train_attack_model(vecs, images)
    recons = reconstruct_images(attack_model, vecs)
    results[method.replace("_", " ")] = recons

evaluate_reconstructions(images, results)