import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as scio
from pulp import *
from collections import Counter
from tensorboardX import SummaryWriter

N_STATES=24+50+3+20
n_agents=3
N_obs=12

qmix_hidden_dim=256
rnn_hidden_dim=256
n_actions=8*3+1
BATCH_SIZE=256
timelength=40
class QMixNet(nn.Module):
    def __init__(self,):
        super(QMixNet, self).__init__()
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        # if args.two_hyper_layers:
        # 这一行定义了一个序列模块 hyper_w1，它包含两个线性层和一个ReLU激活函数，用于生成QMix网络的权重。
        # 第一个线性层将输入特征空间从 N_STATES 映射到 rnn_hidden_dim，然后通过ReLU激活函数激活，
        # 最后一个线性层将其映射到 n_agents * qmix_hidden_dim 的维度。
        # 这个序列模块的输出是一个向量，表示QMix网络的权重。
        self.hyper_w1 = nn.Sequential(nn.Linear(N_STATES, rnn_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(rnn_hidden_dim, n_agents * qmix_hidden_dim))
        # self.hyper_w1.weight.data.normal_(0, 0.1)  # initialization
        #     # 经过hyper_w2得到(经验条数, 1)的矩阵
        # 这一行定义了另一个序列模块 hyper_w2，结构类似于 hyper_w1，用于生成另一组QMix网络的权重。
        self.hyper_w2 = nn.Sequential(nn.Linear(N_STATES, rnn_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(rnn_hidden_dim, qmix_hidden_dim))
        # self.hyper_w1.weight.data.normal_(0, 0.1)  # initialization
        # else:
        # self.hyper_w1 = nn.Linear(N_STATES, n_agents * qmix_hidden_dim)
        #     # 经过hyper_w2得到(经验条数, 1)的矩阵
        # self.hyper_w2 = nn.Linear(N_STATES, qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        # 定义了一个线性层 hyper_b1，用于生成QMix网络的偏置。
        self.hyper_b1 = nn.Linear(N_STATES, qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        # 定义了另一个序列模块 hyper_b2，结构类似于 hyper_w2，用于生成另一组QMix网络的偏置。
        self.hyper_b2 =nn.Sequential(nn.Linear(N_STATES, qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        # 这两行代码将输入的 q_values 和 states 调整为合适的形状，以便与权重进行矩阵相乘
        q_values = q_values.view(-1, 1, n_agents)
        states = states.view(-1, N_STATES)
        # 这两行代码分别将状态 states 输入到 hyper_w1 和 hyper_b1 中，生成QMix网络的权重和偏置。
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)  # (1920, 32)

        w1 = w1.view(-1, n_agents, qmix_hidden_dim)
        b1 = b1.view(-1, 1, qmix_hidden_dim)
        # 这一行代码将 q_values 和 w1 进行矩阵乘法，并加上偏置 b1，然后经过 ELU 激活函数，得到隐藏层的输出 hidden。
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        # 这两行代码分别将状态 states 输入到 hyper_w2 和 hyper_b2 中，生成另一组QMix网络的权重和偏置。
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)
        # 这一行代码将隐藏层的输出 hidden 与 w2 进行矩阵乘法，并加上偏置 b2，得到最终的Q值 q_total。
        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(BATCH_SIZE*timelength, -1)
        return q_total


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self,):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(N_obs+1, rnn_hidden_dim)
        # 这一行对 fc1 的权重进行了初始化，使用了正态分布初始化方法。
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # 这一行创建了一个 GRUCell 层，输入和输出大小均为 rnn_hidden_dim，
        # GRU 是一种门控循环单元，用于处理序列数据。
        # 输入形状的(batch, input_size)：包含输入特征的张量
        # 隐形状的(batch, hidden_size):tensor 包含批次中每个元素的初始隐藏状态。如果未提供，则默认为零。
        # 输出：h' H'形状的(batch, hidden_size):tensor 包含批次中每个元素的下一个隐藏状态
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        # 这一行创建了一个全连接层 fc2，输入大小为 rnn_hidden_dim，输出大小为 n_actions。
        # 这一层的作用是将 RNN 层的输出映射到动作空间。
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)
        # 这一行对 fc2 的权重进行了初始化，使用了正态分布初始化方法。
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x, h):
        # 这一行对输入数据 x 先经过 fc1 层进行线性变换，然后通过 ReLU 激活函数进行非线性变换。
        x = F.relu(self.fc1(x))
        # 这一行将输入的隐藏状态 h 调整为 (batch_size, rnn_hidden_dim) 的形状。
        h = h.reshape(-1, rnn_hidden_dim)
        # 这一行将经过激活函数的数据 x 和隐藏状态 h 输入到 GRUCell 层中，进行 RNN 计算，得到新的隐藏状态 h。
        h = self.rnn(x, h)
        # 这一行将 RNN 层的输出 h 经过 fc2 层进行线性变换，得到输出 q，即动作的预测值。
        q = self.fc2(h)
        return q, h
    # 这是一个用于初始化隐藏状态的方法，根据传入的参数 training 是否为 True 来初始化不同形状的隐藏状态。
    def init_hidden_state(self, training=None):
        # 在形状为[1, BATCH_SIZE, rnn_hidden_dim] 的张量中，第一个维度的大小为1，
        # 通常用于表示序列长度或时间步数。在循环神经网络（RNN）中，
        # 通常使用批次大小（batchsize）和序列长度作为输入数据的维度。
        # 因此，在这里的[1, BATCH_SIZE, rnn_hidden_dim]中，1表示序列长度为1，
        # 即每次输入的数据都是一个独立的序列，而BATCH_SIZE表示批次大小，
        # 即同时处理的序列的数量，rnn_hidden_dim表示隐藏状态的维度。
        if training is True:
            return torch.zeros([1, BATCH_SIZE, rnn_hidden_dim]), torch.zeros([1, BATCH_SIZE, rnn_hidden_dim])
        else:
            return torch.zeros([1, 1, rnn_hidden_dim]), torch.zeros([1, 1, rnn_hidden_dim])

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization二次分布生成数据
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization二次分布生成数据
        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization二次分布生成数据
        self.out = nn.Linear(128,n_actions)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # 激励函数
        x = F.relu(x)
        x = self.fc3(x)  # 激励函数
        x = F.relu(x)
        actions_value = self.out(x)  # Q值
        return actions_value