import os
import time
import argparse
import warnings
from tqdm import tqdm
from setpinn.util import *
from setpinn.models import *
from setpinn.losses import *
from setpinn.data import *
from setpinn.analytical_sol import *
import numpy as np
import torch
from torch.optim import LBFGS, Adam
from torch.optim.lr_scheduler import LambdaLR

torch.set_float32_matmul_precision("high")
best_loss = float("inf")

def _maybe_save_checkpoint(model, exp_path, total_loss):
    global best_loss
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        torch.save(model.state_dict(), os.path.join(exp_path, "best_model.pth"))


def _make_adam_and_scheduler(
    model, base_lr, betas, warmup_steps, total_steps, sched_kind
):
    """Build Adam + (optional) linear-warmup and cosine decay scheduler."""
    adam = Adam(model.parameters(), lr=base_lr, betas=betas)

    if total_steps <= 0:
        return adam, None

    warmup_steps = max(0, int(warmup_steps))
    total_steps = int(total_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if sched_kind == "cosine":
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))  # decay 1 -> 0
        return 1.0

    scheduler = LambdaLR(adam, lr_lambda=lr_lambda)
    return adam, scheduler


def _maybe_apply_rops_sampling(exp_name, data, rops_ctx):
    """Apply region-of-parameter-space (rops) sampling if requested by exp_name."""
    if "rops" not in exp_name:
        return

    x_res = rops_ctx["x_res"]
    t_res = rops_ctx["t_res"]
    sample_num = rops_ctx["sample_num"]
    initial_region = rops_ctx["initial_region"]
    gradient_variance = rops_ctx["gradient_variance"]

    x_res_region_sample_list = []
    t_res_region_sample_list = []
    # clip to a small trust-region side-length
    side = float(np.clip(initial_region / gradient_variance, a_min=0, a_max=0.01))
    for _ in range(sample_num):
        x_region_sample = torch.rand_like(x_res) * side
        t_region_sample = torch.rand_like(t_res) * side
        x_res_region_sample_list.append(x_res + x_region_sample)
        t_res_region_sample_list.append(t_res + t_region_sample)
    data["x_res"] = torch.cat(x_res_region_sample_list, dim=0)
    data["t_res"] = torch.cat(t_res_region_sample_list, dim=0)


def _update_rops_trust_region(exp_name, model, rops_ctx):
    """Update gradient-based trust-region calibration after a step."""
    if "rops" not in exp_name:
        return

    gradient_list_temp = rops_ctx["gradient_list_temp"]
    past_iterations = rops_ctx["past_iterations"]

    # Hook gradients
    flat_grads = []
    for p in model.parameters():
        if p.grad is not None:
            flat_grads.append(p.grad.view(-1))
    if len(flat_grads) == 0:
        flat = torch.zeros(1, device=next(model.parameters()).device)
    else:
        flat = torch.cat(flat_grads)
    gradient_list_temp.append(flat.detach().cpu().numpy())

    # Aggregate over iteration(s)
    rops_ctx["gradient_list_overall"].append(
        np.mean(np.array(gradient_list_temp), axis=0)
    )
    rops_ctx["gradient_list_overall"] = rops_ctx["gradient_list_overall"][
        -past_iterations:
    ]
    gradient_list = np.array(rops_ctx["gradient_list_overall"])

    gradient_variance = (
        np.std(gradient_list, axis=0) / (np.mean(np.abs(gradient_list), axis=0) + 1e-6)
    ).mean()
    if gradient_variance == 0:
        gradient_variance = 1.0

    # Reset temp, store variance
    rops_ctx["gradient_list_temp"] = []
    rops_ctx["gradient_variance"] = float(gradient_variance)


def train_model(
    model,
    data,
    loss_fn,
    training_iterations,
    exp_name,
    exp_path,
    use_adam_warmup=False,
    adam_steps=0,
    warmup_steps=0,
    adam_lr=1e-3,
    adam_betas=(0.9, 0.999),
    adam_scheduler="none",
):
    """
    Train with optional Adam(+warmup) phase followed by LBFGS.

    Adam phase: standard .backward() and .step() per iteration with optional linear warmup and cosine decay.
    LBFGS phase: uses closure as before.
    """
    global best_loss
    best_loss = float("inf")

    device = next(model.parameters()).device
    loss_track = []
    start_time = time.time()

    # Prepare rops context if requested
    rops_ctx = None
    if "rops" in exp_name:
        rops_ctx = {
            "initial_region": 1e-4,
            "sample_num": 1,
            "past_iterations": 10,
            "gradient_list_overall": [],
            "gradient_list_temp": [],
            "gradient_variance": 1.0,
            "x_res": data["x_res"],
            "t_res": data["t_res"],
        }
    # -------- Adam (warmup) phase --------
    adam_iters = int(adam_steps) if use_adam_warmup else 0
    if adam_iters > 0:
        optimizer_adam, scheduler = _make_adam_and_scheduler(
            model, adam_lr, adam_betas, warmup_steps, adam_iters, adam_scheduler
        )
        model.train()
        for step in tqdm(range(adam_iters), desc="Adam warmup/phase"):
            if rops_ctx is not None:
                _maybe_apply_rops_sampling(exp_name, data, rops_ctx)

            optimizer_adam.zero_grad(set_to_none=True)
            losses = loss_fn(model, data)
            total_loss = sum(losses)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_adam.step()
            _maybe_save_checkpoint(model, exp_path, total_loss)
            # log
            loss_track.append([l.item() for l in losses])

            # update trust-region stats
            if rops_ctx is not None:
                _update_rops_trust_region(exp_name, model, rops_ctx)
                
            if scheduler is not None and step % 1000 == 0:
                print("step", step, "lr", scheduler.get_last_lr())

            # step scheduler last
            if scheduler is not None:
                scheduler.step()

    # -------- LBFGS phase --------
    lbfgs_iters = int(training_iterations)
    if lbfgs_iters > 0:
        optimizer_lbfgs = LBFGS(
            model.parameters(),
            line_search_fn="strong_wolfe",
            history_size=100,
            max_iter=20,  # per .step() call; keep modest to avoid stalls
            tolerance_grad=1e-09,
            tolerance_change=1e-12,
        )

        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            losses = loss_fn(model, data)
            total_loss = sum(losses)
            total_loss.backward()
            _maybe_save_checkpoint(model, exp_path, total_loss)
            loss_track.append([l.item() for l in losses])
            return total_loss

        model.train()
        for _ in tqdm(range(lbfgs_iters), desc="LBFGS"):
            if rops_ctx is not None:
                _maybe_apply_rops_sampling(exp_name, data, rops_ctx)

            optimizer_lbfgs.step(closure)

            if rops_ctx is not None:
                _update_rops_trust_region(exp_name, model, rops_ctx)

    run_time = time.time() - start_time
    return loss_track, run_time


def main(args):
    """Main function to handle data preparation, training, and evaluation."""
    # Create experiment directory
    args.exp_path_n = create_experiment_dir(args.exp_path, args.exp_name, args.seed)

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Prepare dataset
    data = get_dataset(args.res_points, args.test_points, args.exp_name, args.device, set_size=4)

    # Initialize and prepare the model
    model = get_model(args.exp_name)
    model.to(args.device)
    print(model)

    # Define loss function
    loss_fn = return_loss_fn(args.exp_name.split("-")[1])

    # Train (Adam warmup -> LBFGS)
    loss_track, run_time = train_model(
        model=model,
        data=data,
        loss_fn=loss_fn,
        training_iterations=args.training_iterations,  # LBFGS iterations
        exp_name=args.exp_name,
        exp_path=args.exp_path_n,
        use_adam_warmup=args.use_adam_warmup,
        adam_steps=args.adam_steps,
        warmup_steps=args.warmup_steps,
        adam_lr=args.adam_lr,
        adam_betas=(args.adam_b1, args.adam_b2),
        adam_scheduler=args.adam_scheduler,
    )
    # load best model if a checkpoint was produced
    best_model_path = os.path.join(args.exp_path_n, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    else:
        warnings.warn(
            "No checkpoint was saved during training; using the final weights instead.",
            RuntimeWarning,
        )

    # Evaluate and plot results
    eval_and_plot(
        args.exp_path_n,
        args.exp_name,
        model,
        run_time,
        data,
        args.test_points,
        loss_track,
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate PINNs")
    parser.add_argument(
        "--exp_path", type=str, required=True, help="Path to save experiment results"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
        help="Device to use (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--res_points", type=int, default=50, help="Number of resolution points"
    )
    parser.add_argument(
        "--test_points", type=int, default=100, help="Number of test points"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (e.g., 'setpinns-wave')",
    )
    parser.add_argument(
        "--training_iterations", type=int, default=500, help="LBFGS iterations"
    )

    # --- Adam warmup flags ---
    parser.add_argument(
        "--use_adam_warmup",
        action="store_true",
        help="Run Adam (with warmup) before LBFGS",
    )
    parser.add_argument(
        "--adam_steps",
        type=int,
        default=50000,
        help="Number of Adam iterations before LBFGS",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="Linear warmup steps inside Adam"
    )
    parser.add_argument(
        "--adam_lr", type=float, default=1e-3, help="Adam base learning rate"
    )
    parser.add_argument("--adam_b1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument(
        "--adam_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="LR schedule after warmup",
    )

    args = parser.parse_args()

    # manual sweeps
    exp_names = [
        "setpinns-wave-norm",
        "setpinns-reaction-norm",
        "setpinns-convection-norm",
        "setpinns-plate-norm",
        #"setpinns-harmonic-norm",
    ]
    seeds = [0]
    for seed in seeds:
        for exp_name in exp_names:
            args.exp_name = exp_name
            args.seed = seed
            print(f"Running experiment: {args.exp_name} with seed {args.seed}")
            main(args)
