import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

from model import GCN, ChebNet, GAT,LSTMNet
from metrics import MAE, MAPE, RMSE
from data_loader import get_loader
from visualize_dataset import show_pred

# 设置随机种子以保证实验的可复现性
seed = 2024
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# 配置matplotlib参数以正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据集
train_loader, test_loader = get_loader('PEMS04')

# 初始化模型
gcn = GCN(6, 6, 1)
chebnet = ChebNet(6, 6, 1, 1)
gat = GAT(6, 6, 1)
# 假设每个时间步的输入特征维度为 10，隐藏层维度为 50，网络有 2 层，最终输出维度为 1
lstm = LSTMNet(input_size=10, hidden_size=50, num_layers=2, output_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型放置在适当的设备上（GPU或CPU）
models = [
    chebnet.to(device),
    gcn.to(device),
    gat.to(device),
    lstm.to(device)
]

# 用于保存所有模型的预测结果
all_predict_values = []
# 设置训练的总轮数
epochs = 30

# 训练模型
for i in range(len(models)):
    model = models[i]
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=3e-2)
    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_mae, epoch_rmse, epoch_mape = 0.0, 0.0, 0.0, 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # 训练数据迭代
            data['graph'], data['flow_x'], data['flow_y'] = data['graph'].to(device), data['flow_x'].to(device), data['flow_y'].to(device)
            predict_value = model(data)  # 模型预测
            loss = criterion(predict_value, data["flow_y"])  # 计算损失
            epoch_mae += MAE(data["flow_y"].cpu(), predict_value.cpu())
            epoch_rmse += RMSE(data["flow_y"].cpu(), predict_value.cpu())
            epoch_mape += MAPE(data["flow_y"].cpu(), predict_value.cpu())

            epoch_loss += loss.item()
            num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_mae = epoch_mae / num
        epoch_rmse = epoch_rmse / num
        epoch_mape = epoch_mape / num
        print(
            "Epoch: {:04d}, Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, Time: {:02.2f} mins".format(
                epoch + 1, 10 * epoch_loss / (len(train_loader.dataset) / 64),
                epoch_mae, epoch_rmse, epoch_mape, (end_time - start_time) / 60))

    ## ---------------------------- 开始模型评估 ---------------------------
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num = 0
        all_predict_value = 0
        all_y_true = 0
        for data in test_loader:  # 测试数据迭代
            data['graph'], data['flow_x'], data['flow_y'] = data['graph'].to(device), data['flow_x'].to(device), data['flow_y'].to(device)
            predict_value = model(data)
            if num == 0:
                all_predict_value = predict_value
                all_y_true = data["flow_y"]
            else:
                all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
                all_y_true = torch.cat([all_y_true, data["flow_y"]], dim=0)
            loss = criterion(predict_value, data["flow_y"])
            total_loss += loss.item()
            num += 1
