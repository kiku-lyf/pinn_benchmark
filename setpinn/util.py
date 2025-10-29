import os
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from setpinn.analytical_sol import *
import copy


def init_weights(m):
    """Initialize weights of a model using Xavier initialization."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def get_n_params(model):
    """Get the number of parameters in a model."""
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def get_clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def seed_everything(seed):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_data(data, title, save_path):
    """Helper function to plot data."""
    plt.figure(figsize=(4, 3))
    # TODO: Check the extent values
    plt.imshow(data, extent=[0, 1, 1, 0], aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def eval_and_plot(exp_path, exp_name, model, run_time, data, test_points, loss_track):
    """
    Evaluate the model, calculate errors, and generate plots.

    Args:
        exp_path (str): Path to save experiment results.
        exp_name (str): Experiment name.
        model (nn.Module): Trained model.
        optim (torch.optim.Optimizer): Optimizer used.
        run_time (float): Total training time.
        data (dict): Dataset including test data.
        test_points (int): Number of test points.
        loss_track (list): List of tracked loss values.
    """
    total_params = get_n_params(model)
    print(f"Train Loss: {np.sum(loss_track[-1]):.4f}")

    model.eval()
    with torch.no_grad():
        pred_tensor = model(data["x_test"], data["t_test"])
        # Handle different output formats
        if pred_tensor.dim() == 3:  # [B, S, 1] format for set-based models
            pred = pred_tensor.squeeze(-1).reshape(-1).detach().cpu().numpy()
        elif pred_tensor.dim() == 2:  # [N, 1] format
            pred = pred_tensor.squeeze().detach().cpu().numpy()
        else:  # [N] format
            pred = pred_tensor.reshape(-1).detach().cpu().numpy()
        pred = pred.reshape(test_points, test_points)

    # Analytical solution - ensure inputs are 1D arrays
    x_tensor = data["x_test"].detach().cpu().numpy()
    t_tensor = data["t_test"].detach().cpu().numpy()
    
    # Handle different input formats
    if x_tensor.ndim == 2:  # [N, 1] format
        x = x_tensor.squeeze()
        t = t_tensor.squeeze()
    elif x_tensor.ndim == 3:  # [B, S, 1] format
        x = x_tensor.squeeze(-1).reshape(-1)
        t = t_tensor.squeeze(-1).reshape(-1)
    else:  # [N] format
        x = x_tensor.reshape(-1)
        t = t_tensor.reshape(-1)
    
    u = analytical_sol(exp_name.split("-")[1], x, t).reshape(test_points, test_points)

    # Errors
    rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
    rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u**2))
    print(f"Relative L1 error: {rl1:.4f}")
    print(f"Relative L2 error: {rl2:.4f}")

    # Generate and save plots
    plot_data(pred, "Predicted u(x,t)", os.path.join(exp_path, f"{exp_name}_pred.pdf"))
    plot_data(u, "Exact u(x,t)", os.path.join(exp_path, f"{exp_name}_exact.pdf"))
    plot_data(np.abs(pred - u), "Absolute Error", os.path.join(exp_path, f"{exp_name}_error.pdf"))

    # Save logs
    logging = {
        "model_weights": model.state_dict(),
        "loss": loss_track,
        "runtime": run_time,
        "test_pred": pred,
        "test_exact": u,
        "total_model_parameters": total_params,
    }
    with open(os.path.join(exp_path, "logging.pkl"), "wb") as f:
        pickle.dump(logging, f)

    with open(os.path.join(exp_path, "errors.txt"), "w") as f:
        f.write(f"rl1_error:{rl1}\nrl2_error:{rl2}")


def create_experiment_dir(exp_path, exp_name, seed):
    """
    Create a directory for the experiment.

    Args:
        exp_path (str): Base path for experiments1.
        exp_name (str): Name of the experiment.
        seed (int): Seed for reproducibility.

    Returns:
        str: Path to the created experiment directory.
    """
    model_name, eq_name, technique = exp_name.split("-")
    path = os.path.join(exp_path, eq_name, model_name, technique, f"seed_{seed}")
    os.makedirs(path)
    return path