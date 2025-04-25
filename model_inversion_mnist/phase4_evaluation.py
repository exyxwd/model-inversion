from utils import compute_mse, plot_comparison

def evaluate_reconstructions(ground_truth, reconstructed_dict):
    plot_comparison(ground_truth, reconstructed_dict)