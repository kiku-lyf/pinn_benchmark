import torch
import numpy as np


def return_loss_fn(eq_name):
    """Return the appropriate loss function based on the equation name."""
    loss_functions = {
        "wave": loss_fn_1dwave,
        "reaction": loss_fn_1dreaction,
        "convection": loss_fn_convection,
        "plate": loss_fn_rect_patch,
        "harmonic": loss_fn_plate_harmonic_scaled,
    }
    if eq_name not in loss_functions:
        raise ValueError(f"Invalid equation name: {eq_name}")
    return loss_functions[eq_name]

# --- Safe grads for set models (handles both [N,1] and [B,S,1]) ---

def diag_grad(y: torch.Tensor, x: torch.Tensor):
    """
    Returns per-token dy_i/dx_i with 'other tokens' treated as constants.
    Works for both vanilla ([N,1]) and set tensors ([B,S,1]).
    """
    assert y.shape == x.shape, "y and x must have identical shapes"
    create_graph = True
    return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            retain_graph=True, create_graph=create_graph
        )[0]

    # Vanilla case: [N,1]
    if y.dim() == 2:
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            retain_graph=True, create_graph=create_graph
        )[0]

    # Set case: [B,S,1] -> compute diagonal of the Jacobian
    B, S, _ = y.shape
    grads = []
    for s in range(S):
        v = torch.zeros_like(y)
        v[:, s, :] = 1.0                    # select y_i for token s
        g = torch.autograd.grad(
            y, x, grad_outputs=v,
            retain_graph=True, create_graph=create_graph
        )[0]
        grads.append(g[:, s:s+1, :])        # pick dx for the same token s
    return torch.cat(grads, dim=1)


def diag_hessian(y: torch.Tensor, x: torch.Tensor):
    """
    Returns per-token d^2 y_i / dx_i^2.
    """
    dy_dx = diag_grad(y, x)
    return diag_grad(dy_dx, x)

# Define loss functions
def loss_fn_1dwave(model, data):
    """
    Loss function for the 1D wave equation.
    Args:
        model: Neural network model.
        data: Dictionary containing input tensors.
    Returns:
        Tuple of loss components (loss_res, loss_ic, loss_bc).
    """
    device = data["x_res"].device
    pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).to(device)
    
    # Predictions
    pred_res = model(data["x_res"], data["t_res"])
    pred_left = model(data["x_left"], data["t_left"])
    pred_upper = model(data["x_upper"], data["t_upper"])
    pred_lower = model(data["x_lower"], data["t_lower"])

    # Gradients
    u_xx = diag_hessian(pred_res, data["x_res"])
    u_tt = diag_hessian(pred_res, data["t_res"])

    # Residual Loss
    loss_res = torch.mean((u_tt - 4 * u_xx) ** 2)

    # Boundary Condition Loss
    loss_bc = torch.mean(pred_upper**2) + torch.mean(pred_lower**2)

    # Initial Condition Loss
    ui_t = diag_grad(pred_left, data["t_left"])
    loss_ic_1 = torch.mean(
        (
            pred_left[:, 0]
            - torch.sin(pi * data["x_left"][:, 0])
            - 0.5 * torch.sin(3 * pi * data["x_left"][:, 0])
        )
        ** 2
    )
    loss_ic_2 = torch.mean(ui_t**2)
    loss_ic = loss_ic_1 + loss_ic_2
    S = data["x_res"].shape[1] if data["x_res"].dim() == 3 else 1

    return loss_res, loss_ic, loss_bc


def loss_fn_1dreaction(model, data):
    """
    Loss function for the 1D reaction equation.
    Args:
        model: Neural network model.
        data: Dictionary containing input tensors.
    Returns:
        Tuple of loss components (loss_res, loss_ic, loss_bc).
    """
    pred_res = model(data["x_res"], data["t_res"])
    pred_left = model(data["x_left"], data["t_left"])
    pred_upper = model(data["x_upper"], data["t_upper"])
    pred_lower = model(data["x_lower"], data["t_lower"])

    # Gradients
    u_t = diag_grad(pred_res, data["t_res"])

    # Residual Loss
    loss_res = torch.mean((u_t - 5 * pred_res * (1 - pred_res)) ** 2)

    # Boundary Condition Loss
    loss_bc = torch.mean((pred_upper - pred_lower) ** 2)

    # Initial Condition Loss
    loss_ic = torch.mean(
        (
            pred_left[:, 0]
            - torch.exp(
                -((data["x_left"][:, 0] - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2)
            )
        )
        ** 2
    )
    # scale residual by tokens-per-set
    S = data["x_res"].shape[1] if data["x_res"].dim() == 3 else 1

    return loss_res, loss_ic, loss_bc


def loss_fn_convection(model, data):
    """
    Loss function for the 1D convection equation.
    Args:
        model: Neural network model.
        data: Dictionary containing input tensors.
    Returns:
        Tuple of loss components (loss_res, loss_ic, loss_bc).
    """
    pred_res = model(data["x_res"], data["t_res"])
    pred_left = model(data["x_left"], data["t_left"])
    pred_upper = model(data["x_upper"], data["t_upper"])
    pred_lower = model(data["x_lower"], data["t_lower"])

    # Gradients
    u_x = diag_grad(pred_res, data["x_res"])
    u_t = diag_grad(pred_res, data["t_res"])

    # Residual Loss
    loss_res = torch.mean((u_t + 50 * u_x) ** 2)

    # Boundary Condition Loss
    loss_bc = torch.mean((pred_upper - pred_lower) ** 2)

    # Initial Condition Loss
    loss_ic = torch.mean((pred_left[:, 0] - torch.sin(data["x_left"][:, 0])) ** 2)
    S = data["x_res"].shape[1] if data["x_res"].dim() == 3 else 1
    return loss_res, loss_ic, loss_bc


def loss_fn_rect_patch(model, data, x0=0.25, x1=0.3, y0=0.7, y1=0.75, Q=20):
    """
    Loss for the PDE:
      -Delta(u)= f(x,y),
      f(x,y)=Q if x0<=x<=x1 & y0<=y<=y1 else 0,
      and boundary u=0 on [0,1]^2.

    data must have:
      'x_res', 't_res' -> interior collocation points => (x,y)
      'x_left','t_left','x_right','t_right', etc. for boundary
    """

    # 1) PDE Residual
    x_res = data["x_res"]
    y_res = data["t_res"]
    pred_res = model(x_res, y_res)  # shape [N_res,1]

    u_xx = diag_hessian(pred_res, x_res)
    u_yy = diag_hessian(pred_res, y_res)

    # Evaluate f(x_res, y_res)
    # piecewise constant: Q inside rectangle, 0 outside
    mask_x = (x_res >= x0) & (x_res <= x1)
    mask_y = (y_res >= y0) & (y_res <= y1)
    mask_xy = mask_x & mask_y

    f_res = torch.zeros_like(pred_res)
    f_res[mask_xy] = Q

    # PDE => (u_xx + u_yy) - f_res = 0
    residual = (u_xx + u_yy) - f_res
    loss_res = torch.mean(residual**2)

    # 2) Boundary conditions: pinned => u=0
    pred_left = model(data["x_left"], data["t_left"])
    pred_right = model(data["x_right"], data["t_right"])
    pred_upper = model(data["x_upper"], data["t_upper"])
    pred_lower = model(data["x_lower"], data["t_lower"])

    loss_bc = (
        torch.mean(pred_left**2)
        + torch.mean(pred_right**2)
        + torch.mean(pred_upper**2)
        + torch.mean(pred_lower**2)
    )
    S = data["x_res"].shape[1] if data["x_res"].dim() == 3 else 1
    return loss_res, loss_bc


def loss_fn_plate_harmonic_scaled(model, data, kx=5, ky=3, A=500.0):
    """
    Loss for the PDE:
      -Delta(u) = A*sin(kx*pi*x)*sin(ky*pi*y),
      => (u_xx + u_yy) + A*sin(kx*pi*x)*sin(ky*pi*y)=0,
      with boundary condition u=0.

    data: standard dictionary with:
      x_res, t_res -> interior collocation points
      x_left, t_left, x_right, t_right, etc. -> boundary
    """
    x_res = data["x_res"]
    y_res = data["t_res"]
    pred_res = model(x_res, y_res)  # shape [N_res,1]

    # 2nd derivatives
    u_xx = diag_hessian(pred_res, x_res)
    u_yy = diag_hessian(pred_res, y_res)

    device = x_res.device
    pi_t = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).to(device)

    # Forcing = A*sin(kx*pi*x)*sin(ky*pi*y)
    f_res = A * torch.sin(kx * pi_t * x_res) * torch.sin(ky * pi_t * y_res)

    # PDE residual => (u_xx + u_yy)+ f_res = 0
    residual = (u_xx + u_yy) + f_res
    loss_res = torch.mean(residual**2)

    # Boundary condition: pinned => u=0
    pred_left = model(data["x_left"], data["t_left"])
    pred_right = model(data["x_right"], data["t_right"])
    pred_upper = model(data["x_upper"], data["t_upper"])
    pred_lower = model(data["x_lower"], data["t_lower"])
    loss_bc = (
        torch.mean(pred_left**2)
        + torch.mean(pred_right**2)
        + torch.mean(pred_upper**2)
        + torch.mean(pred_lower**2)
    )
    S = data["x_res"].shape[1] if data["x_res"].dim() == 3 else 1
    return loss_res, loss_bc
