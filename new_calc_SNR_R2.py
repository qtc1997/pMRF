import numpy as np
import math
from new_DGP import new_DGP_1, new_DGP_2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_snr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算信噪比（SNR）
    :param y_true: 真实信号（不含噪声）
    :param y_pred: 观测值（包含噪声）
    :return: 信噪比
    """
    error = y_pred - y_true
    signal_variance = np.var(y_true)
    noise_variance = np.var(error)
    snr = signal_variance / noise_variance
    return snr

# 计算DGP1、DGP2和DGP3的信噪比
n, p = 1600, 100  # 假设样本数为1000，特征数为5
method = "DGP2"
if method == "DGP1":
    # DGP 1
    X, y1, y_true1, error1 = new_DGP_1(n, p)
elif method == "DGP2":
    X, y1, y_true1, error1 = new_DGP_2(n, p)
else:
    raise ValueError("Invalid method")

snr_1 = calculate_snr(y_true1, y1)
print(f"SNR for {method}: {snr_1}")

print(f"theoretical R2 by SNR:{snr_1/(snr_1+1)}")



# 验证SNR和R2的关系
# 用X和X**2做回归
# 只取第一个特征
X_squared = X**2
# 合并特征
X_combined = np.hstack([X, X_squared])
# 拟合线性回归
reg = LinearRegression().fit(X_combined, y1)
y_pred = reg.predict(X_combined)
r2 = r2_score(y1, y_pred)
print(f"R2 score: {r2}")


# scale_list = range(1,11)

# for scale in scale_list:
#     X, y1, y_true1, error1 = DGP_1(n, p, scale)
#     snr_1 = calculate_snr(y_true1, y1)
#     print(f"SNR for DGP 1: {snr_1}")

# n_list = [100, 200, 400, 800, 1600]

# for n in n_list:
#     X, y1, y_true1, error1 = DGP_1(n, p)
#     snr_1 = calculate_snr(y_true1, y1)
#     print(f"SNR for DGP 1: {snr_1}")

#     X, y2, y_true2, error2 = DGP_2(n, p)
#     snr_2 = calculate_snr(y_true2, y2)
#     print(f"SNR for DGP 2: {snr_2}")


