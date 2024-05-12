import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



# -----------------------------------路网结构的嵌入模块-----------------------------------

def get_adjacent_matrix_origion(distance_file: str, num_nodes: int, id_file: str = None, graph_type="connect") -> np.array:
    """
    通过CSV文件构建邻接矩阵
    :param distance_file: 保存节点间距离的CSV文件路径
    :param num_nodes: 图中的节点数量
    :param id_file: 保存节点顺序的txt文件路径
    :param graph_type: ["connect", "distance"]，如果使用加权，应设为"distance"
    :return:
    """

    A = np.zeros([int(num_nodes), int(num_nodes)])

    if id_file:
        with open(id_file, "r") as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3:
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")

    return A

def get_adjacent_matrix(distance_file: str, num_nodes: int, threshold_distance=1.0) -> np.array:
    """
    ## 此处创新-采用高斯核滤波器
    构建邻接矩阵，综合考虑地理关系和逻辑关系。
    :param distance_file: CSV文件的路径，保存了节点之间的距离。
    :param correlation_file: CSV文件的路径，保存了节点之间的相关系数。
    :param num_nodes: 图中的节点数量。
    :param threshold_distance: 距离阈值，用于定义何种程度的地理接近性被视为有效连接。
    :param threshold_correlation: 相关性阈值，只有当相关系数的绝对值大于此阈值时，节点间才建立连接。
    :return: 构建的邻接矩阵。
    """
    A = np.zeros([num_nodes, num_nodes])  # 初始化邻接矩阵为零矩阵

    # 处理地理距离数据
    with open(distance_file, "r") as file:
        file.readline()  # 跳过标题行
        reader = csv.reader(file)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])
            # 使用距离的倒数作为权重，只有当距离小于等于阈值时才考虑为有效连接
            if distance <= threshold_distance:
                A[i, j] = A[j, i] = 1 / distance


    return A

def get_flow_data(flow_file: str) -> np.array:
    """
    从npz文件解析流数据。
    :param flow_file: npz文件路径，数据结构应为 (N, T, D)
    :return: 返回一个numpy数组或其他数据结构，包含解析后的流数据。
    """
    data = np.load(flow_file)
    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, D]  D = 1
    return flow_data



# -----------------------------------加载处理数据类PEMSDataset---------------------------------------
class PEMSDataset(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
        """
        # 导入数据类
        data_path: 包含两个文件名的列表，一个是图文件名，另一个是流数据文件名。
        num_nodes: 图中的节点数。
        divide_days: 一个包含训练天数和测试天数的列表。
        time_interval: 两条交通数据记录之间的时间间隔（分钟）。
        history_length: 使用的历史数据长度。
        train_mode: 训练模式，可以是 "train" 或 "test"。
        """

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]
        self.test_days = divide_days[1]
        self.history_length = history_length  # 6
        self.time_interval = time_interval  # 5 min
        self.one_day_length = int(24 * 60 / self.time_interval)
        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes)
        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1)

    def __len__(self):
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = PEMSDataset.slice_data(self.flow_data, self.history_length, index, self.train_mode)
        data_x = PEMSDataset.to_tensor(data_x)  # [N, H, D]
        data_y = PEMSDataset.to_tensor(data_y).unsqueeze(1)  # [N, 1, D]
        return {"graph": PEMSDataset.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}



## ---------------------------- 划分数据集和测试集---------------------------------------
    @staticmethod
    def slice_data(data, history_length, index, train_mode):
        """
            data: 标准化后的交通数据，一个多维数组。
            history_length: 用于预测的历史数据长度。
            index: 指定提取数据的时间轴上的索引点。
            train_mode: 指示函数操作模式是训练（'train'）还是测试（'test'）。
        :return:
            data_x: 形状为 [N, H, D] 的数组，其中 N 是节点数，H 是历史长度，D 是数据的特征维度。
            data_y: 形状为 [N, D] 的数组，用于模型预测的目标数据。
        """

        ## 从给定的索引点开始，向后取固定长度的历史数据作为输入（data_x），紧接着的数据点作为输出（data_y）。
        if train_mode == "train":
            start_index = index
            end_index = index + history_length

        ## 从给定索引点向前取固定长度的历史数据作为输入，当前索引点的数据作为输出。
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]
        data_y = data[:, end_index]

        return data_x, data_y




## ------------------------------------ 数据预处理模块---------------------------------------------------------------

    @staticmethod
    def pre_process_data(data, norm_dim):
        """
          对原始交通数据进行预处理，包括标准化。
          :param data: np.array, 原始交通数据，未经标准化。
          :param norm_dim: int, 标准化处理的维度。
          :return:
              norm_base: list, 包含数据标准化的基准（最大值和最小值）。
              norm_data: np.array, 经过标准化处理的交通数据。
          """
        norm_base = PEMSDataset.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = PEMSDataset.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

## 目的：计算数据的最大值和最小值，用作后续标准化处理的基础。
    @staticmethod
    def normalize_base(data, norm_dim):
        """
        计算数据的标准化基准（最大值和最小值）。
        :param data: np.array, 原始交通数据，未经标准化。
        :param norm_dim: int, 需要在此维度上计算最大值和最小值。
        :return:
            max_data: np.array, 每个维度上的最大值。
            min_data: np.array, 每个维度上的最小值。
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
        min_data = np.min(data, norm_dim, keepdims=True)
        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data


    ## 将经过标准化处理的数据恢复到其原始的数值范围
    @staticmethod
    def recover_data(max_data, min_data, data):

        mid = min_data
        base = max_data - min_data
        recovered_data = data * base + mid

        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)



## --------------------------- 数据导入主函数---------------------------

def get_loader(ds_name="PEMS04"):
    num_nodes = 307 if ds_name == 'PEMS04' else 170
    train_data = PEMSDataset(data_path=["../dataset/PEMS/{}/distance.csv".format(ds_name), "../dataset/PEMS/{}/data.npz".format(ds_name)], num_nodes=num_nodes,
                             divide_days=[45, 14],
                             time_interval=5, history_length=6,
                             train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = PEMSDataset(data_path=["../dataset/PEMS/{}/distance.csv".format(ds_name), "../dataset/PEMS/{}/data.npz".format(ds_name)], num_nodes=num_nodes,
                            divide_days=[45, 14],
                            time_interval=5, history_length=6,
                            train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader

