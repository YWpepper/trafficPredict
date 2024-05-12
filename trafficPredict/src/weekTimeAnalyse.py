# -*- ecoding: utf-8 -*-
# @ModuleName: weekAnalyse
# @Function: 
# @Author: wenYan(pepper)
# @Time: 2024/4/7 11:13

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_flow(filename):
    flow_data = np.load(filename)
    return flow_data['data']

## 对于一周的每一天交通流量进行可视化
def rolingShow(traffic_data,nodeNum):
    # 创建一个表示一天中时间的列表（每5分钟一个数据点）
    full_times = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]

    # 只选择每隔30分钟的时间点作为 x 轴标签
    times = full_times[::12]  # 每30分钟的时间点

    # 设置绘图布局，7 行 1 列
    fig, axs = plt.subplots(7, 1, figsize=(15, 20), sharex=True)

    # 设置滑动平均的窗口大小
    window_size = 12  # 一个小时的数据点

    # 对于一周中的每一天
    for i in range(7):
        # 提取一天的数据 (288 时间点/天) 对于第一个传感器
        day_data = traffic_data[i * 288:(i + 1) * 288, nodeNum, :]

        # 存储该传感器的流量数据，这里假设我们关注的是第一个特征
        daily_traffic = day_data[:, 0]

        # 计算滑动平均
        smoothed_traffic = pd.Series(daily_traffic).rolling(window=window_size, min_periods=1, center=True).mean()

        # 在对应的子图上绘制每一天的流量曲线
        axs[i].plot(full_times, smoothed_traffic, label=f'Day {i + 1}')  # 使用平滑后的数据点
        axs[i].set_title(f'Day {i + 1}')
        axs[i].set_ylabel('Traffic Flow')
        axs[i].legend()

    # 设置共享 x 轴标签
    plt.xlabel('Time of Day')
    axs[-1].set_xticks(times)  # 只在最后一个子图上设置 x 轴刻度
    plt.xticks(rotation=45)  # 设置 x 轴刻度旋转

    plt.tight_layout()
    plt.savefig("../assets/timeViewAnalyse/node" + str(nodeNum) + "oneWeekAnalyse.jpg")
    plt.show()

## 对于一周的每一天交通流量进行可视化
def normalShow(traffic_data,nodeNum):

    # 创建一个表示一天中时间的列表（每5分钟一个数据点）
    full_times = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]

    # 只选择每隔30分钟的时间点作为 x 轴标签
    times = full_times[::12]  # 每30分钟的时间点

    # 设置绘图布局，7 行 1 列
    fig, axs = plt.subplots(7, 1, figsize=(15, 20), sharex=True)

    # 对于一周中的每一天
    for i in range(7):
        # 提取一天的数据 (288 时间点/天) 对于第一个传感器
        day_data = traffic_data[i * 288:(i + 1) * 288, nodeNum, :]

        # 存储该传感器的流量数据，这里假设我们关注的是第一个特征
        daily_traffic = day_data[:, 0]

        # 在对应的子图上绘制每一天的流量曲线
        axs[i].plot(full_times, daily_traffic, label=f'Day {i + 1}')  # 使用全部数据点
        axs[i].set_title(f'Day {i + 1}')
        axs[i].set_ylabel('Traffic Flow')
        axs[i].legend()

    # 设置共享 x 轴标签
    plt.xlabel('Time of Day')
    axs[-1].set_xticks(times)  # 只在最后一个子图上设置 x 轴刻度
    plt.xticks(rotation=45)  # 设置 x 轴刻度旋转

    plt.tight_layout()

    # plt.savefig("../assets/timeViewAnalyse/node" +str(nodeNum) +  "oneWeekAnalyse.jpg")
    plt.show()


## # 收集连续7周的周一数据。
def pearSunShow(traffic_data,nodeNum):


    # 这里需要定义nodeNum，它代表数据集中节点/传感器的索引。
    # 创建一个列表，代表一天中的每5分钟。
    full_times = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]

    # 仅选择每30分钟的时间点作为x轴标签。
    times = full_times[::6]

    # 收集连续7周的周一数据。
    mondays_data = [traffic_data[i * 288:(i + 1) * 288, nodeNum, :] for i in range(0, 7 * 7, 7)]

    # 计算7个周一的交通流量数据的相关系数矩阵。
    mondays_traffic = np.array([day[:, 0] for day in mondays_data])
    correlation_matrix = np.corrcoef(mondays_traffic)

    # 绘制相关系数矩阵的热力图。
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    # 在热力图的每个格子中添加相关系数的值。
    for (i, j), val in np.ndenumerate(correlation_matrix):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val < 0.5 else 'black')

    # 为每个轴设置标签。
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels([f"week{1 + i}" for i in range(7)], rotation=45)
    ax.set_yticklabels([f"week{1 + i}" for i in range(7)])

    # 设置标题和轴标签。
    # plt.title('连续7周每周一的交通量相关性矩阵')
    plt.xlabel('Monday#')
    plt.ylabel('Monday#')
    plt.savefig("../assets/pearsun/node" +str(nodeNum) +  "pear.jpg")

    # 显示图像。
    plt.show()


## 收集第一周的7天数据可视化
def SevendayOneWeek(traffic_data,nodeNum):
    # Assuming traffic_data is a numpy array with shape (intervals, nodes, features)
    # and contains the traffic flow data.

    # Assuming each interval represents 5 minutes, calculate the total number of intervals for one week
    intervals_per_day = 24 * 12  # 24 hours/day * 12 intervals/hour
    intervals_per_week = 7 * intervals_per_day

    # Reshape the first week of data for the selected node into a (7, intervals_per_day) array
    first_week_traffic = traffic_data[:intervals_per_week, nodeNum, 0].reshape(7, intervals_per_day)

    # Calculate the Pearson correlation coefficient matrix for the 7 days
    correlation_matrix = np.corrcoef(first_week_traffic)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    # Add the correlation values to the heatmap
    for (i, j), val in np.ndenumerate(correlation_matrix):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val < 0.5 else 'black')

    # Set labels for each axis
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(days_of_week)
    ax.set_yticklabels(days_of_week)

    # Set the title and axis labels
    # plt.title('Pearson Correlation Coefficient Matrix for Node' + str(nodeNum) + '- First Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Day of the Week')

    # Save the figure
    plt.savefig('/Volumes/pepper/01Code/02biYeSheJi/papper&Code/realWork/assets/pearsun'+'/node5_pearson_heatmap_first_week.jpg')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    nodeNum = 5
    traffic_data = get_flow('../dataset/PEMS/PEMS08/pems08.npz')
    # rolingShow(traffic_data,nodeNum)
    # normalShow(traffic_data,nodeNum)
    # pearSunShow(traffic_data,nodeNum)
    SevendayOneWeek(traffic_data,nodeNum)