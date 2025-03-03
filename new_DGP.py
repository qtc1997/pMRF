#  Terry 2024-12-15
#  pMRF 那篇论文的所有代码

# 1. 模拟数据生成过程DGP
# 2. 标准RF 的求解 
# 3. MRF 即 带mallows准则的RF  用自带的求解器求解
# 4. pMRF 即 带惩罚的MRF 要用ADMM算法求解  单独写在ADMM_pMRF.py中


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import math


def new_DGP_1(n:int, p:int , scale:int = 10) -> list[np.ndarray, np.ndarray]:
    """
    模拟数据生成过程DGP
    :Biau, G. (2012). Analysis of a random forests model. The Journal of Machine Learning Research, 13(1):1063-1095
    :X ~ U[0, 1]
    :y = 10 * sin(pi * X1) + epsilon
    """
    X = np.random.rand(n, p)
    X1 = X[:,0].reshape(-1,1)
    # error 的shape 要和 X1 一致
    error = np.random.randn(n, 1)
    y_true = scale * np.sin(math.pi * X1 ) 
    y = y_true + error

    return [X, y, y_true,error]

def new_DGP_2(n:int, p:int , scale:int = 10) -> list[np.ndarray, np.ndarray]:
    """
    模拟数据生成过程DGP 2
    :y = 10 sin(pi * X1 * X2) + 20*(X3 - 0.05)**2 + 10*X4 + 5*X5 + epsilon
    """

    X = np.random.rand(n,p)
    # 把X 转成 (n,1) 的shape
    X1 = X[:,0].reshape(-1,1)
    X2 = X[:,1].reshape(-1,1)
    X3 = X[:,2].reshape(-1,1)
    X4 = X[:,3].reshape(-1,1)
    X5 = X[:,4].reshape(-1,1)
    epsilon = np.random.randn(n,1)

    y_true = scale*(np.sin(math.pi * X1 * X2) + 2*(X3 - 0.05)**2 + X4 + 0.5*X5) 
    y = y_true + epsilon
    return [X, y, y_true, epsilon]



def new_DGP_3(n:int, p:int,scale) -> list[np.ndarray, np.ndarray]:
    """
    模拟数据生成过程DGP 3
    """
    X = np.random.rand(n,p)
    # 把X 转成 (n,1) 的shape
    X1 = X[:,0].reshape(-1,1)
    X2 = X[:,1].reshape(-1,1)
    X3 = X[:,2].reshape(-1,1)
    X4 = X[:,3].reshape(-1,1)
    X5 = X[:,4].reshape(-1,1)
    epsilon = np.random.randn(n,1)

    y = np.zeros((n,1))
    for i in range(n):
        ri = decision_tree(X1[i],X2[i],X3[i],X4[i],X5[i],scale)
        y[i] = ri + epsilon[i]

    return [X,y]


def decision_tree(X1, X2, X3, X4, X5,scale):
    if X4 < 0.383:
        if X2 < 0.2342:
            return 0.8177*scale
        else:
            if X1 < 0.2463:
                return 0.8837*scale
            else:
                return 1.315*scale
    else:
        if X1 < 0.47:
            if X5 < 0.2452:
                return 1.099*scale
            else:
                if X3 >= 0.2234:
                    return 1.803*scale
                else:
                    return 1.387*scale
        else:
            if X2 < 0.2701:
                return 1.502*scale
            else:
                if X5 < 0.5985:
                    return 1.861*scale
                else:
                    return 2.174*scale
