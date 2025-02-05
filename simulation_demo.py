from sklearn.model_selection import train_test_split
from pMRF_demo import *
import json
import numpy as np
# 参数设置 ；
# fixed
p = 100 
M = 1000
N = 50
method = 'TLP'

# vary
max_features = p + 0
n_list = [100,200,300,400, 500,750,1000]
# max_depth_list = [10,20,30,50,100,None]
scenario_list = ['DGP1','DGP2','DGP3']
max_features_list = [50,70,80,90,100]
max_depth = None 

# scenario = 'DGP_1'
real_depth_arr = []

for scenario in scenario_list:
    for n in n_list:
        for max_features in max_features_list:
            print("--------------------------------")
            print(f"n={n}, max_depth={max_depth}, max_features={max_features}")


            rmse_rf_list = []
            rmse_mrf_list = []
            # rmse_pmrf_list = []

            tree_depth_list = []
            # tree_leaf_num_list = []


            for i in range(N):
                if scenario == 'DGP1':
                    X, y = DGP_1(n=n,p=p)
                elif scenario == 'DGP2':
                    X, y = DGP_2(n=n,p=p)
                elif scenario == 'DGP3':
                    X, y = DGP_3(n=n,p=p)

                # 分割训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

                # print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")

                # RF
                [trees, trees_pred, km, sigma2, tree_depth, tree_leaf_num] = RF_train(X=X_train, y=y_train, M=M, max_depth=max_depth, max_features=max_features)
                tree_depth_list.append(tree_depth)
                # tree_leaf_num_list.append(tree_leaf_num)
                # 预测
                y_pred = RF_predict(X=X_test, trees=trees)
                # 计算RMSE

                print(f"y_test.shape: {y_test.shape}, y_pred.shape: {y_pred.shape}")
                rmse_rf = np.sqrt(np.mean((y_test - y_pred)**2))
                print(f"RF RMSE: {rmse_rf}")


                # MRF
                sigma2_mean = np.mean(sigma2)
                # print(type(y_train), type(trees_pred), type(km), type(sigma2_mean))
                w = cp_solver_MRF(y=y_train, y_preds=trees_pred, km=km, sigma2_mean=sigma2_mean)
                # 预测
                # print(w)
                print(w.shape)
                y_pred = RF_predict_weighted(X=X_test, trees=trees, w=w)
                # 计算RMSE
                print(f"y_test.shape: {y_test.shape}, y_pred.shape: {y_pred.shape}")
                rmse_mrf = np.sqrt(np.mean((y_test - y_pred)**2))
                print(f"MRF RMSE: {rmse_mrf}")

            


                # # pMRF
                # lamb = 0.003
                # beta_opt, w_opt,cost_list,w_list = grid_search(y=y_train, y_preds=trees_pred, km=km, sigma2_mean=sigma2_mean, w_init=w, lamb=lamb, method=method)

                # print(f"beta_opt: {beta_opt}, w_opt: {w_opt}")

                # # print(w_opt.shape)

                # y_pred = RF_predict_weighted(X=X_test, trees=trees, w=w_opt)
                # rmse_pmrf = np.sqrt(np.mean((y_test - y_pred)**2))
                # print(f"pMRF RMSE: {rmse_pmrf}")


                rmse_rf_list.append(rmse_rf)
                rmse_mrf_list.append(rmse_mrf)
                # rmse_pmrf_list.append(rmse_pmrf)
            dict = {
                "rmse_rf_list": rmse_rf_list,
                "rmse_mrf_list": rmse_mrf_list,
                # "rmse_pmrf_list": rmse_pmrf_list
                "params": {
                    "n": n,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "method": method,
                    "scenario": scenario,
                    "p": p,
                    "M": M,
                    "N": N
                }
            }
            # 变成json文件存储
            file_name = f"./simulation_results/rmse_{scenario}_{n}_{max_depth}_{max_features}.json"
            with open(file_name, 'w') as f:
                json.dump(dict, f)
            print("tree_depth_list:",np.mean(tree_depth_list))
            # print(np.mean(tree_depth_list))   
            # real_depth_arr.append(np.mean(tree_depth_list))
            # print("tree_leaf_num_list:",np.mean(tree_leaf_num_list))


