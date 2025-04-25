import torch
from attacks.label_only_attack import recover_confidence_vector
from attacks.vector_based_attack import get_confidence_vector
from attacks.score_based_attack import get_score_based_vector
from attacks.one_hot_attack import get_one_hot_vector

def generate_confidence_vectors(images, labels, target_model,
                                method="label_only",
                                shadow_model=None, mu=0.1):
    vectors = []
    for x, y in zip(images, labels):
        if method == "label_only":
            vec = recover_confidence_vector(shadow_model, mu, y)
        elif method == "vector_based":
            vec = get_confidence_vector(target_model, x)
        elif method == "score_based":
            vec = get_score_based_vector(target_model, x)
        elif method == "one_hot":
            vec = get_one_hot_vector(target_model, x)
        else:
            raise ValueError("Invalid method")
        vectors.append(vec)
    return torch.stack(vectors)
