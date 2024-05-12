import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



# ------------------------------传统图卷积GCN网络---------------------------------------
class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        """
        初始化图卷积网络 (GCN)。
        :param in_c: 输入特征的通道数。
        :param hid_c: 隐藏层的节点数。
        :param out_c: 输出特征的通道数。
        """
        super(GCN, self).__init__()
        # 第一层线性变换，将输入特征从 in_c 转换到 hid_c
        self.linear_1 = nn.Linear(in_c, hid_c)
        # 第二层线性变换，从隐藏层特征转换到输出特征
        self.linear_2 = nn.Linear(hid_c, out_c)
        # 激活函数
        self.act = nn.ReLU()

    def forward(self, data):
        """
        前向传播函数。
        :param data: 包含图结构和流特征的字典。
        :return: 经过两层图卷积处理的节点特征。
        """
        # 从输入数据中提取图的邻接矩阵
        graph_data = data["graph"][0]  # 邻接矩阵 [N, N]
        # 处理邻接矩阵，增加自环并归一化
        graph_data = self.process_graph(graph_data)

        # 从输入数据中提取流特征
        flow_x = data["flow_x"]  # 流特征 [B, N, H, D]
        B, N = flow_x.size(0), flow_x.size(1)
        # 将所有历史时间步的特征合并
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]

        # 第一次线性变换后应用图卷积
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # 应用图卷积

        # 第二次线性变换后再次应用图卷积
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)  # 调整输出维度以适配后续处理

    @staticmethod
    def process_graph(graph_data):
        """
        处理图的邻接矩阵，增加自环并进行归一化。
        :param graph_data: 邻接矩阵，形状为 [N, N]。
        :return: 归一化后的图的邻接矩阵。
        """
        N = graph_data.size(0)
        # 增加自环
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        # 计算归一化的度矩阵
        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)  # 度的逆
        degree_matrix[degree_matrix == float("inf")] = 0.  # 处理度为0的节点

        # 构造度矩阵
        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        # 返回归一化的邻接矩阵
        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)


# ------------------------------传统lstm预测---------------------------------------
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        """
        初始化 LSTM 预测模型。
        :param input_size: int, 每个时间步输入特征的维度。
        :param hidden_size: int, LSTM 隐藏层的维度。
        :param num_layers: int, LSTM 堆叠的层数。
        :param output_size: int, 输出层的维度。
        :param dropout_rate: float, 在输出层前应用的 dropout 比率。
        """
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 使用 Dropout 层减少过拟合
        self.dropout = nn.Dropout(dropout_rate)

        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播定义。
        :param x: 输入数据，形状为 [batch_size, sequence_length, input_size]
        :return: 输出预测结果。
        """
        # 初始化隐状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 从 LSTM 层获取所有时间步的输出以及最终的隐状态和细胞状态
        out, _ = self.lstm(x, (h0, c0))

        # 应用 dropout
        out = self.dropout(out)

        # 只取序列的最后一步
        out = out[:, -1, :]

        # 通过全连接层得到最终的输出
        out = self.fc(out)

        return out


# ------------------------------ 切比雪夫多项式近似图卷积----------------------------------------
class ChebConv(nn.Module):
    # ChebConv 类实现了一个图卷积层，其特别之处在于使用了切比雪夫多项式来扩展标准的图卷积

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        """
        初始化切比雪夫卷积层。
        :param in_c: 输入通道数。
        :param out_c: 输出通道数。
        :param K: 切比雪夫多项式的阶数。
        :param bias: 是否添加偏置项。
        :param normalize: 是否对拉普拉斯矩阵进行归一化。
        """
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # 初始化权重参数
        nn.init.xavier_normal_(self.weight)  # 使用Xavier初始化权重

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))  # 初始化偏置项
            init.zeros_(self.bias)  # 使用零初始化偏置
        else:
            self.register_parameter("bias", None)

        self.K = K + 1  # 存储多项式的阶数

    def forward(self, inputs, graph):
        """
        前向传播函数。
        :param inputs: 输入数据，形状为 [B, N, C]，其中 B 是批大小，N 是节点数，C 是通道数。
        :param graph: 图的邻接矩阵，形状为 [N, N]。
        :return: 卷积后的结果，形状为 [B, N, D]，其中 D 是输出通道数。
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # 获取拉普拉斯矩阵
        mul_L = self.cheb_polynomial(L).unsqueeze(1)  # 计算切比雪夫多项式
        result = torch.matmul(mul_L, inputs)  # 应用多项式于输入
        result = torch.matmul(result, self.weight)  # 应用权重
        result = torch.sum(result, dim=0) + self.bias  # 加上偏置项并求和

        return result

    def cheb_polynomial(self, laplacian):
        """
        计算切比雪夫多项式。
        :param laplacian: 拉普拉斯矩阵，形状为 [N, N]。
        :return: 多阶的拉普拉斯矩阵，形状为 [K, N, N]。
        """
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)  # T_0(x) = 1

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian  # T_1(x) = x
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    # T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        计算图的拉普拉斯矩阵。
        :param graph: 邻接矩阵，形状为 [N, N]。
        :param normalize: 是否进行归一化处理。
        :return: 拉普拉斯矩阵。
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

class ChebNet(nn.Module):

    ## ChebNet 类是一个完整的神经网络模型，它利用 ChebConv 图卷积层来构建多层图神经网络。 【两层conv+relu】

    def __init__(self, in_c, hid_c, out_c, K):
        """
        初始化 ChebNet 网络。
        :param in_c: int, 输入特征的维数，即每个节点的特征数。
        :param hid_c: int, 中间隐藏层的特征维数。
        :param out_c: int, 输出特征的维数，通常对应于预测的交通流量维数。
        :param K: 切比雪夫多项式的最高阶数，决定了图卷积的局部感知范围。
        """
        super(ChebNet, self).__init__()
        # 第一层切比雪夫卷积，将输入特征从 in_c 转换到 hid_c
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        # 第二层切比雪夫卷积，进一步处理特征，从 hid_c 转换到 out_c
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)
        # 使用 ReLU 激活函数增加非线性处理能力，提高模型的表达能力
        self.act = nn.ReLU()

    def forward(self, data):
        """
        网络的前向传播过程。
        :param data: 包含图信息和流特征的字典。
        :return: 输出预测结果，具体为交通流量预测。
        """
        # 从输入数据中提取图的邻接矩阵和流量特征
        graph_data = data["graph"][0]  # 邻接矩阵 [N, N]
        flow_x = data["flow_x"]  # 流量特征 [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)  # 批大小和节点数

        # 将时间维度和特征维度合并处理
        flow_x = flow_x.view(B, N, -1)  # 调整流量特征维度为 [B, N, H*D]

        # 通过第一层图卷积处理特征，并应用激活函数
        output_1 = self.act(self.conv1(flow_x, graph_data))
        # 通过第二层图卷积进一步处理特征，并再次应用激活函数
        output_2 = self.act(self.conv2(output_1, graph_data))

        # 将输出结果的维度调整为预期格式以匹配后续处理或输出要求
        return output_2.unsqueeze(2)  # 调整输出维度为 [B, N, 1, D]





# ------------------------------ 图注意力网络（GAT）--------------------------------------------
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_c, out_c, alpha=0.2):
        """
        初始化图注意力层。
        :param in_c: int, 输入特征的维度。
        :param out_c: int, 输出特征的维度。
        :param alpha: float, LeakyReLU激活函数的负斜率。
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.alpha = alpha

        # 权重矩阵W，用于特征转换
        self.W = nn.Parameter(torch.empty(size=(in_c, out_c)))
        nn.init.xavier_normal_(self.W.data)  # 使用Xavier初始化方法
        # 注意力机制的参数向量a
        self.a = nn.Parameter(torch.empty(size=(2 * out_c, 1)))
        nn.init.xavier_normal_(self.a.data)  # 使用Xavier初始化方法
        # 带有alpha参数的LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, features, adj):
        """
        前向传播计算。
        :param features: tensor, 输入的特征矩阵，形状为 [B, N, in_features]。
        :param adj: tensor, 邻接矩阵，形状为 [B, N, N]。
        :return: tensor, 输出的特征矩阵，形状为 [B, N, out_features]。
        """
        B, N = features.size(0), features.size(1)

        # 检查是否可用GPU并进行相应配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 确保邻接矩阵和单位矩阵在同一设备上
        adj = adj.to(device) + torch.eye(N, dtype=adj.dtype).to(device)

        # 将输入特征矩阵通过权重矩阵W进行线性变换
        h = torch.matmul(features, self.W)  # [B, N, out_features]

        # 为计算注意力系数准备张量
        a_input = torch.cat([
            h.repeat(1, 1, N).view(B, N * N, -1),  # 将每个节点的特征重复N次
            h.repeat(1, N, 1)  # 对每个节点重复特征N次
        ], dim=2).view(B, N, -1, 2 * self.out_c)

        # 计算注意力系数的原始分数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B, N, N]

        # 用非常小的负数替换不相连的节点的分数，保持数值稳定性
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 只有相连的节点才考虑

        # 对注意力系数应用softmax，标准化成概率分布
        attention = F.softmax(attention, dim=2)

        # 使用注意力系数加权输入特征
        h_prime = torch.matmul(attention, h)  # [B, N, out_features]

        return h_prime

    def __repr__(self):
        """
        返回模型的字符串描述。
        """
        return self.__class__.__name__ + ' (' + str(self.in_c) + ' -> ' + str(self.out_c) + ')'



# ------------------------------ 多头图注意力网络（GAT）-------------------------------------------
class GAT(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads=6):
        """
        初始化图注意力网络（GAT）。
        :param in_c: int, 输入特征的维数。
        :param hid_c: int, 隐藏层特征的维数。
        :param out_c: int, 输出特征的维数。
        :param n_heads: int, 多头注意力机制中的头数。
        """
        super(GAT, self).__init__()
        # 初始化多头注意力层，每个头都对输入特征进行转换
        self.attentions = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        # 第二层注意力，将多头注意力的结果进一步处理到输出维度
        self.conv2 = GraphAttentionLayer(hid_c*n_heads, out_c)
        # 激活函数
        self.act = nn.ELU()

    def forward(self, data):
        """
        前向传播过程。
        :param data: dict, 包含图结构和流特征的字典。
        :return: tensor, 经过两层图注意力处理后的节点特征。
        """
        # 从输入数据中提取邻接矩阵和特征矩阵
        adj = data["graph"][0]  # 邻接矩阵，形状为 [N, N]
        x = data["flow_x"]  # 特征矩阵，原始形状为 [B, N, H, D]

        # 获取批次大小和节点数
        B, N = x.size(0), x.size(1)
        # 将特征矩阵重新整形为 [B, N, H*D] 以合并所有时间步的特征
        x = x.view(B, N, -1)

        # 应用多头注意力机制，对每个头的输出沿最后一维拼接，然后通过激活函数
        outputs = self.act(torch.cat([attention(x, adj) for attention in self.attentions], dim=-1))
        # 将多头注意力的结果通过第二层注意力处理，并再次应用激活函数
        output_2 = self.act(self.conv2(outputs, adj))

        # 将输出调整为 [B, 1, N, 1] 形状，以符合预期输出格式
        return output_2.unsqueeze(2)
