import json 
import matplotlib.pyplot as plt
# 目标文件夹的名称
path_name2 = 'simulation_results_0126'
path_name = 'simulation_results_0127'
path_name = 'simulation_results'
path_name = 'simulation_results_0128_jsy'

scenarios = 'DGP3'

n_list = [100,200,300,400,500,750,1000]
max_features_list = [50,70,80,90,100]
# max_features_list = [80,90,100]
max_depth = None

n_list = [100,200,300,500,1000]
max_features_list = [60,70,80,90]




visualization_data = {}
for n in n_list:
    for max_features in max_features_list:
        file_name = f'{path_name}/rmse_{scenarios}_{n}_{max_depth}_{max_features}.json'
        # print(file_name)
        file_name = f'{path_name}/rmse_{scenarios}_n{n}_max_features_{max_features}.json'



        with open(file_name, "r") as f:
            data = json.load(f)
        
        # 提取参数和结果
        params = data["params"]
        rmse_rf_list = data["rmse_rf_list"]
        rmse_mrf_list = data["rmse_mrf_list"]
        
        # 计算平均 RMSE
        avg_rmse_rf = sum(rmse_rf_list) / len(rmse_rf_list)
        avg_rmse_mrf = sum(rmse_mrf_list) / len(rmse_mrf_list)
        
        # 按 max_features 分类存储数据
        max_features = params["max_features"]
        n = params["n"]
        
        if max_features not in visualization_data:
            visualization_data[max_features] = {"n": [], "rmse_rf": [], "rmse_mrf": []}
        
        visualization_data[max_features]["n"].append(n)
        visualization_data[max_features]["rmse_rf"].append(avg_rmse_rf)
        visualization_data[max_features]["rmse_mrf"].append(avg_rmse_mrf)



color_dict = {50:'red',60:'red',70:'blue',80:'green',90:'yellow',100:'purple'}

plt.figure(figsize=(10, 6))
for max_features, data in visualization_data.items():
    sorted_indices = sorted(range(len(data["n"])), key=lambda i: data["n"][i])
    n_sorted = [data["n"][i] for i in sorted_indices]
    rmse_rf_sorted = [data["rmse_rf"][i] for i in sorted_indices]
    rmse_mrf_sorted = [data["rmse_mrf"][i] for i in sorted_indices]

    print(max_features)
    print(color_dict[max_features])
    plt.plot(n_sorted, rmse_rf_sorted, label=f"RF (max_features={max_features})", linestyle="--", marker="o",color=color_dict[max_features])
    plt.plot(n_sorted, rmse_mrf_sorted, label=f"MRF (max_features={max_features})", linestyle="-", marker="s",color=color_dict[max_features])

plt.xlabel("Number of Samples (n)", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title(f"RMSE Comparison between RF and MRF in {scenarios}", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

