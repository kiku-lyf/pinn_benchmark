import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib
import argparse
from setpinn.models import get_model
from setpinn.data import get_dataset, X_Y_RANGES
from setpinn.analytical_sol import analytical_sol

# --- START OF MODIFICATIONS: Font and Size Configuration ---
# Use a backend that doesn't require a GUI
matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure Matplotlib for publication-quality plots
# This block sets all fonts to Times New Roman and controls the sizes.
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman'],

    # 默认文字大小（如图例、注释）
    'font.size':          10,

    # 子图标题和坐标轴标签
    'axes.titlesize':     14,   # 标题
    'axes.labelsize':     18,   # x,y 轴标签

    # 刻度标签
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,

    # 色条标签和刻度
    'legend.fontsize':    10,
    'figure.titlesize':   16,
})
# --- END OF MODIFICATIONS ---


def load_model_and_predict(exp_name, model_path, device, test_points):
    """
    加载模型并进行预测
    
    Args:
        exp_name: 实验名称，格式为 "模型名-方程名-技术名"
        model_path: 模型权重文件路径
        device: 设备 ('cuda:0' 或 'cpu')
        test_points: 测试点数量 (每个方向)
    
    Returns:
        pred: 预测结果，形状为 (test_points, test_points)
        exact: 精确解，形状为 (test_points, test_points)
        x_range: x轴范围
        t_range: t轴范围
    """
    # 解析实验名称
    model_name, eq_name, technique = exp_name.split("-")
    
    # 获取数据范围
    x_range, t_range = X_Y_RANGES[eq_name]
    
    # 生成测试数据
    data = get_dataset(test_points, test_points, exp_name, device, set_size=4)
    
    # 初始化模型
    model = get_model(exp_name)
    model.to(device)
    
    # 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    # 模型评估模式
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        pred = model(data["x_test"], data["t_test"])
        # 处理不同模型的输出格式
        if pred.dim() == 3:  # [B, S, 1] 格式
            pred = pred.squeeze(-1).reshape(-1, 1)
        elif pred.dim() == 2:  # [N, 1] 格式
            pred = pred
        else:
            pred = pred.reshape(-1, 1)
        
        pred = pred.cpu().numpy().reshape(test_points, test_points)
    
    # 计算精确解
    x_test = data["x_test"].detach().cpu().numpy()
    t_test = data["t_test"].detach().cpu().numpy()
    
    # 处理不同模型的输入格式（x_test 和 t_test 可能是 [N, 1] 或 [N] 格式）
    if x_test.ndim == 2:  # [N, 1] 格式
        x_test = x_test.squeeze()
        t_test = t_test.squeeze()
    elif x_test.ndim == 3:  # [B, S, 1] 格式（理论上不应该出现，但为了安全）
        x_test = x_test.squeeze(-1).reshape(-1)
        t_test = t_test.squeeze(-1).reshape(-1)
    else:  # [N] 格式
        x_test = x_test.reshape(-1)
        t_test = t_test.reshape(-1)
    
    exact = analytical_sol(eq_name, x_test, t_test).reshape(test_points, test_points)
    
    return pred, exact, x_range, t_range


def plot_results(pred, exact, x_range, t_range, exp_name, save_path):
    """
    绘制结果图：真实解、预测解、绝对误差
    
    Args:
        pred: 预测结果，形状为 (test_points, test_points)
        exact: 精确解，形状为 (test_points, test_points)
        x_range: x轴范围 [x_min, x_max]
        t_range: t轴范围 [t_min, t_max]
        exp_name: 实验名称
        save_path: 保存路径
    """
    # 创建网格
    test_points = pred.shape[0]
    x_star = np.linspace(x_range[0], x_range[1], test_points)
    t_star = np.linspace(t_range[0], t_range[1], test_points)
    TT, XX = np.meshgrid(t_star, x_star)
    
    # 计算误差
    error = np.abs(exact - pred)
    rl1 = np.sum(np.abs(exact - pred)) / np.sum(np.abs(exact))
    rl2 = np.sqrt(np.sum((exact - pred) ** 2) / np.sum(exact**2))
    
    print(f'Relative L1 error: {rl1:.4e}')
    print(f'Relative L2 error: {rl2:.4e}')
    
    # 创建画布
    fig = plt.figure(figsize=(18, 5))
    
    # 子图1：真实解
    plt.subplot(1, 3, 1)
    plt.pcolormesh(TT, XX, exact, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Exact Solution')
    
    # 子图2：预测解
    plt.subplot(1, 3, 2)
    plt.pcolormesh(TT, XX, pred, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predicted Solution')
    
    # 子图3：绝对误差
    plt.subplot(1, 3, 3)
    plt.pcolormesh(TT, XX, error, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute Error')
    
    # 统一保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制模型预测结果')
    parser.add_argument(
        '--exp_name',
        type=str,
        required=True,
        help='实验名称，格式: 模型名-方程名-技术名 (例如: setpinns-wave-norm)'
    )
    parser.add_argument(
        '--exp_path',
        type=str,
        default='./runs',
        help='实验路径 (默认: ./runs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='随机种子 (默认: 0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='设备 (默认: cuda:0)'
    )
    parser.add_argument(
        '--test_points',
        type=int,
        default=101,
        help='测试点数量，每个方向 (默认: 101)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='模型权重文件路径 (默认: 自动从exp_path查找best_model.pth)'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='保存图像路径 (默认: exp_path/exp_name_results.pdf)'
    )
    
    args = parser.parse_args()
    
    # 自动查找模型路径
    if args.model_path is None:
        model_name, eq_name, technique = args.exp_name.split("-")
        model_path = os.path.join(
            args.exp_path, eq_name, model_name, technique, 
            f"seed_{args.seed}", "best_model.pth"
        )
    else:
        model_path = args.model_path
    
    # 自动设置保存路径
    if args.save_path is None:
        model_name, eq_name, technique = args.exp_name.split("-")
        save_dir = os.path.join(
            args.exp_path, eq_name, model_name, technique, f"seed_{args.seed}"
        )
        os.makedirs(save_dir, exist_ok=True)
        args.save_path = os.path.join(save_dir, f"{args.exp_name}_results.pdf")
    
    # 加载模型并预测
    print(f"加载模型: {args.exp_name}")
    print(f"模型路径: {model_path}")
    pred, exact, x_range, t_range = load_model_and_predict(
        args.exp_name, model_path, args.device, args.test_points
    )
    
    # 绘制结果
    print(f"绘制结果图...")
    plot_results(pred, exact, x_range, t_range, args.exp_name, args.save_path)
    
    print("完成！")


if __name__ == "__main__":
    main()

