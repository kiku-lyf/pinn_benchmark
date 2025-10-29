import torch
import numpy as np

# Define the x and y ranges for each equation
X_Y_RANGES = {
    "wave": [[0, 1], [0, 1]],
    "reaction": [[0, 2 * np.pi], [0, 1]],
    "convection": [[0, 2 * np.pi], [0, 1]],
    "plate": [[0, 1], [0, 1]],
    "harmonic": [[0, 1], [0, 1]],
}

def get_data(x_range, y_range, x_num, y_num):
    """
    Generate mesh data for the specified x and y ranges.

    Args:
        x_range (list): [x_min, x_max]
        y_range (list): [y_min, y_max]
        x_num (int): Number of points along x-axis.
        y_num (int): Number of points along y-axis.

    Returns:
        tuple: Contains reshaped data, boundary conditions, and x, y arrays.
    """
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)
    x_mesh, t_mesh = np.meshgrid(x, t)
    data = np.stack((x_mesh, t_mesh), axis=-1)

    b_left, b_right = data[0, :, :], data[-1, :, :]
    b_upper, b_lower = data[:, -1, :], data[:, 0, :]
    res = data.reshape(-1, 2)

    return res, b_left, b_right, b_upper, b_lower, x, t

def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src

def eas_sampling(x, t, set_size=4):
    N, M = x.shape[0], t.shape[0]
    assert (N - 1) * (M - 1) >= set_size, "set_size is too large"
    # iterate over each square in the grid
    samples = []
    for i in range(N - 1):
        for j in range(M - 1):
            x0, x1 = x[i], x[i + 1]
            t0, t1 = t[j], t[j + 1]
            # sample set_size points in the square
            xs = np.random.uniform(x0, x1, set_size)
            ts = np.random.uniform(t0, t1, set_size)
            samples.append(np.stack((xs, ts), axis=-1))
    samples = np.array(samples)
    return samples

def prepare_data_as_tensors(data_dict, device):
    """Convert data arrays to PyTorch tensors and move to the specified device."""
    return {key: torch.tensor(value, dtype=torch.float32, requires_grad=True).unsqueeze(-1).to(device)
            for key, value in data_dict.items()}

def get_dataset(res_points, test_points, exp_name, device, set_size=None):
    """
    Generate training and testing datasets for different models.
    
    Args:
        res_points (int): Number of collocation points for training (NxN).
        test_points (int): Number of test points (NxN).
        exp_name (str): Experiment name (format: model_name-eq_name).
        device (str): Device to use for PyTorch tensors.
    
    Returns:
        dict: Dictionary of formatted datasets.
    """
    model_name, eq_name, technique = exp_name.split("-")
    x_range, y_range = X_Y_RANGES[eq_name]
    # Get data for training and testing
    train_data = get_data(x_range, y_range, res_points, res_points)
    test_data = get_data(x_range, y_range, test_points, test_points)
    res, b_left, b_right, b_upper, b_lower, xx_train, tt_train = train_data
    res_test, _, _, _, _, xx_test, tt_test = test_data
    # Process data based on model type
    if model_name == "pinnsformer":
        num_step, step = 5, 1e-4
        boundary_conditions = [b_left, b_right, b_upper, b_lower]
        res = make_time_sequence(res, num_step=num_step, step=step)
        boundaries = [make_time_sequence(b, num_step=num_step, step=step) for b in boundary_conditions]
    elif model_name == "setpinns":
        div = int(np.sqrt(set_size))
        res_points = int(res_points // div) + 1
        _,_,_,_,_, xx_train, tt_train = get_data(x_range, y_range, res_points, res_points)
        res = eas_sampling(xx_train, tt_train, set_size)
        boundaries = [b_left, b_right, b_upper, b_lower]
    else:
        boundaries = [b_left, b_right, b_upper, b_lower]

    # Extract x and t for training
    x_res, t_res = res[..., 0], res[..., 1]
    x_boundary, t_boundary = zip(*[(b[..., 0], b[..., 1]) for b in boundaries])

    # Format data as tensors
    formatted_data = {
        "x_res": x_res,
        "t_res": t_res,
        "x_test": res_test[..., 0],
        "t_test": res_test[..., 1],
    }

    boundary_keys = ["x_left", "x_right", "x_upper", "x_lower"]
    for key, x_b, t_b in zip(boundary_keys, x_boundary, t_boundary):
        formatted_data[key] = x_b
        formatted_data[f"t_{key.split('_')[1]}"] = t_b

    # Convert data to tensors and move to the device
    data_tensors = prepare_data_as_tensors(formatted_data, device)

    # Additional formatting for specific models
    if model_name in ["setpinns", "pinnsformer"]:
        data_tensors["x_test"] = data_tensors["x_test"].unsqueeze(-1)
        data_tensors["t_test"] = data_tensors["t_test"].unsqueeze(-1)
    
    if model_name in ["setpinns"]:
        for key in boundary_keys:
            data_tensors[key] = data_tensors[key].unsqueeze(-1)
            data_tensors[f"t_{key.split('_')[1]}"] = data_tensors[f"t_{key.split('_')[1]}"].unsqueeze(-1)

    return data_tensors


if __name__ == "__main__":
    data = get_dataset(50, 50, "setpinns-wave-norm", "cpu", 4)
    x_res = data["x_res"].detach().numpy()
    t_res = data["t_res"].detach().numpy()
    x_test = data["x_test"].detach().numpy()
    t_test = data["t_test"].detach().numpy()
    x_left = data["x_left"].detach().numpy()
    t_left = data["t_left"].detach().numpy()
    x_right = data["x_right"].detach().numpy()
    t_right = data["t_right"].detach().numpy()
    x_upper = data["x_upper"].detach().numpy()
    t_upper = data["t_upper"].detach().numpy()
    x_lower = data["x_lower"].detach().numpy()
    t_lower = data["t_lower"].detach().numpy()

    # print shapes of all the data
    print(x_res.shape, t_res.shape)
    print(x_test.shape, t_test.shape)
    print(x_left.shape, t_left.shape)
    print(x_right.shape, t_right.shape)
    print(x_upper.shape, t_upper.shape)
    print(x_lower.shape, t_lower.shape)