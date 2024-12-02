import random
import time
import os
from collections import namedtuple, deque
import numpy as np
import models
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)


class DQAgent:
    ''' 深度Q学习代理，适用于基于图的强化学习任务。'''
    def __init__(self, args):
        '''
        lr: learning rate
        n_step: (s_t-n,a_t-n,r,s_t)
        '''
        self.model_name = args.model # 模型名称（如S2V_DUEL, S2V_DQN等）
        self.gamma = 0.99 # 折扣因子（用于计算未来奖励的折扣）
        self.n_step = args.n_step # num of steps to accumulate rewards n步后计算累积奖励

        self.training = not(args.test) # 是否处于训练模式
        self.T = args.T # S2V

        self.memory = ReplayMemory(args.memory_size) # 经验回放池，用于存储历史经验
        self.batch_size = args.bs # batch size for experience replay

        self.double_dqn = args.double_dqn # 是否使用双重DQN
        self.device = args.device # 使用的设备（CPU或GPU）

        self.node_dim = 2 # 节点特征的维度
        self.edge_dim = 4 # 边特征的维度
        self.reg_hidden = args.reg_hidden # 正则化的隐藏层维度
        self.embed_dim = args.embed_dim # 嵌入维度
        # store node embeddings of each graph, avoid multiprocess copy
        self.graph_node_embed = {}
        # model and graph input
        if self.model_name == 'S2V_DUEL':
            self.model = models.S2V_DUEL(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, edge_dim=4,
                T=self.T, w_scale=0.01, avg=False).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.S2V_DUEL(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, 
                    edge_dim=4, T=self.T, w_scale=0.01, avg=False).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            # graph input
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'S2V_DQN':
            self.model = models.S2V_DQN(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, edge_dim=4,
                T=self.T, w_scale=0.01, avg=False).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.S2V_DQN(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, 
                    edge_dim=4, T=self.T, w_scale=0.01, avg=False).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            # graph input
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'Tripling':
            self.model = models.Tripling(embed_dim=self.embed_dim, sgate_l1_dim=128, tgate_l1_dim=128, T=3, 
                hidden_dims=[50, 50, 50], w_scale=0.01).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.Tripling(embed_dim=self.embed_dim, sgate_l1_dim=128, tgate_l1_dim=128, T=3,
                    hidden_dims=[50, 50, 50], w_scale=0.01).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            # graph input
            self.setup_graph_input = self.setup_graph_input_tripling

        else:
            raise NotImplementedError(f'RL Model {self.model_name}')

        # 定义损失函数和优化器
        self.criterion = torch.nn.MSELoss(reduction='mean') # 均方误差损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr) # Adam优化器

        if not self.training:
            # load pretrained model for testing
            cwd = os.getcwd() # 获取当前工作目录
            self.model.load_state_dict(torch.load(os.path.join(cwd, args.model_file)))
            self.model.eval()

    def reset(self):
        ''' restart '''
        pass

    @torch.no_grad()
    def setup_graph_input_s2v(self, graphs, states, actions=None):
        ''' create a batch data loader from a batch of
                states, # pred all
                states, actions, # pred
                node features (states), edge features
            return a batch from the data loader
        '''
        sample_size = len(graphs)
        data = []
        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim) # 初始化节点特征
            x[:, 1] = 1 - states[i] # selected node feature set 0
            edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous() # 边的索引
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim) # 初始化边的特征
            edge_attr[:, 1] = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float) # 边的权重
            edge_attr[:, 0] = states[i][edge_index[0]] # 当前状态
            edge_attr[:, 2] = torch.abs(states[i][edge_index[0]] - states[i][edge_index[1]]) # 边的状态差异

            y = actions[i].clone() if actions is not None else None # 如果有动作，复制动作

            data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        loader = DataLoader(data, batch_size=sample_size, shuffle=False) # 创建数据加载器（迭代器）
        for batch in loader:
            # adjust y if applicable
            if actions is not None:
                total_num = 0
                for i in range(1, sample_size):
                    total_num += batch[i - 1].num_nodes
                    batch[i].y += total_num # 将前面所有图的节点数加到当前图的目标 y 上 保证偏移
            return batch.to(self.device)


    def setup_graph_input_tripling(self, graphs, states, actions=None):
        ''' create a batch data loader from a batch of
                states, # pred all
                states, actions, # pred
                node features (states), edge features
            return a batch from the data loader
        '''
        sample_size = len(graphs)
        data = []
        for i in range(sample_size):
            # initialize node embedding if not
            if id(graphs[i]) not in self.graph_node_embed:
                self.graph_node_embed[id(graphs[i])] = models.get_init_node_embed(graphs[i], 30, self.device) # epochs for initial embedding
            with torch.no_grad():
                # copy node embedding as node feature
                x = self.graph_node_embed[id(graphs[i])].detach().clone()
                x = torch.cat((x, states[i].detach().clone().unsqueeze(dim=1)), dim=-1) # 沿着最后一个维度拼接 x:(node_size, 101(node_dim)) states:(sample_size, node_size)
                edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
                # use edge weight 
                edge_weight = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)

                y = actions[i].detach().clone() if actions is not None else None
                data.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y))

        with torch.no_grad():
            loader = DataLoader(data, pin_memory=True, num_workers=8, batch_size=sample_size, shuffle=False)
            for batch in loader:
                # adjust y if applicable
                if actions is not None:
                    total_num = 0
                    for i in range(1, sample_size):
                        total_num += batch[i - 1].num_nodes
                        batch[i].y += total_num # 将前面所有图的节点数加到当前图的目标 y 上 保证偏移
                return batch.to(self.device)


    @torch.no_grad()
    def setup_graph_pred(self, graphs, states, actions):
        ''' create a batch data loader from a batch of
                states, actions, 
                node features (states), edge features
            return a batch from the data loader
        '''
        sample_size = len(states)
        data = []
        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim)
            x[:, 1] = 1 - states[i] # selected node feature set 0
            edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim)
            edge_attr[:, 1] = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)
            edge_attr[:, 0] = states[i][edge_index[0]]
            edge_attr[:, 2] = torch.abs(states[i][edge_index[0]] - states[i][edge_index[1]])

            y = actions[i].clone()

            data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            # adjust y
            total_num = torch.tensor([0], dtype=torch.long)
            for i in range(1, sample_size):
                total_num.add_(batch[i - 1].num_nodes)
                batch[i].y = torch.add(batch[i].y, total_num)
            return batch.to(self.device)

    @torch.no_grad()
    def setup_graph_pred_all(self, graph, state):
        ''' create a batch data loader from 
                state,
                node features (state), edge features
            return a batch from the data loader
        '''
        # node features 初始化节点特征张量
        x = torch.ones(graph.num_nodes, self.node_dim)
        # 将每个节点的第二个特征设置为 1 减去对应的状态值
        x[:, 1] = 1 - state # selected node feature set 0
        # from to edges
        edge_index = torch.tensor(graph.from_to_edges(), dtype=torch.long).t().contiguous()
        # edge features
        edge_attr = torch.ones(graph.num_edges, self.edge_dim)
        edge_attr[:, 1] = torch.tensor([p[-1] for p in graph.from_to_edges_weight()], dtype=torch.float)
        edge_attr[:, 0] = state[edge_index[0]]
        edge_attr[:, 2] = torch.abs(state[edge_index[0]] - state[edge_index[1]])
        # creat mini-batch loader
        data = [Data(x=x, edge_index=edge_index, edge_attr=edge_attr)]
        loader = DataLoader(data, batch_size=1, shuffle=False)
        for batch in loader:
            return batch.to(self.device)

    def select_action(self, graph, state, epsilon, training=True, budget=None):
        ''' act upon state '''
        # testing
        if not(training):
            # 构造输入图数据，将state扩展为batch
            graph_input = self.setup_graph_input([graph], state.unsqueeze(dim=0))
            with torch.no_grad():
                q_a = self.model(graph_input)
            q_a[state.nonzero()] = -1e5 # 非零元素设置一个很小的数

            # 将当前状态下已经选择的节点的Q值设置为一个非常小的数（排除已选择的节点）
            if budget is None:
                return torch.argmax(q_a).detach().clone()
            else: # return all seed nodes within budget at one time
                return torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone()
        # training
        available = (state == 0).nonzero() # 获取当前状态中值为0的节点（即尚未选择的节点）
        if epsilon > random.random(): # epsilon-greedy策略：以epsilon的概率选择一个随机动作（探索），否则选择Q值最大的动作（利用）
            return random.choice(available)
        else:
            # 如果是利用策略，首先构造图数据输入
            graph_input = self.setup_graph_input([graph], state.unsqueeze(dim=0))
            with torch.no_grad():
                q_a = self.model(graph_input)
            # 找到Q值最大的位置
            max_position = (q_a == q_a[available].max().item()).nonzero()
            # 在Q值最大的节点中随机选择一个
            return torch.tensor(
                [random.choice(
                    np.intersect1d(available.cpu().contiguous().view(-1).numpy(), 
                        max_position.cpu().contiguous().view(-1).numpy()))], 
                dtype=torch.long)

    def memorize(self, env):
        '''n step for stability'''
        # 从环境(env)中获取状态列表、奖励列表和动作列表，
        # 将其加入n步回放内存中。
        sum_rewards = [0.0] # 初始化一个空的累计奖励列表
        for reward in reversed(env.rewards):
            # normalize reward by number of nodes
            reward /= env.graph.num_nodes # 将奖励除以节点数量进行归一化
            sum_rewards.append(reward + self.gamma * sum_rewards[-1]) # 累计奖励计算公式：当前奖励 + 折扣因子乘以之前的累计奖励
        sum_rewards = sum_rewards[::-1] # 反转累计奖励列表

        # 遍历环境中的每个状态
        for i in range(len(env.states)):
            # 如果当前状态后面还有n步，则将状态、动作、下一个状态和奖励存入内存
            if i + self.n_step < len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long),  # 当前状态
                    torch.tensor([env.actions[i]], dtype=torch.long),  # 当前动作
                    torch.tensor(env.states[i + self.n_step], dtype=torch.long), # n步后的状态
                    torch.tensor([sum_rewards[i] - (self.gamma ** self.n_step) * sum_rewards[i + self.n_step]], dtype=torch.float), # n步累计奖励
                    env.graph)
            # 如果当前状态刚好是最后一个状态，且没有后续n步，则只存当前状态的奖励
            elif i + self.n_step == len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long), 
                    torch.tensor([env.actions[i]], dtype=torch.long), 
                    None,
                    torch.tensor([sum_rewards[i]], dtype=torch.float),  
                    env.graph)


    def fit(self):
        '''fit on a batch sampled from replay memory 通过从回放内存中采样批次数据来训练模型'''
        # optimize model 确定批次大小：如果内存中样本数大于批次大小，使用批次大小；否则使用内存中所有样本
        sample_size = self.batch_size if len(self.memory) >= self.batch_size else len(self.memory)
        # need to fix dimension and restrict action space 从回放内存中采样一个批次的数据
        transitions = self.memory.sample(sample_size)
        # 转换成Transition对象，方便后续操作  Transition如下定义了
        batch = Transition(*zip(*transitions)) # batch.state = states
        # 生成一个布尔掩码，标记哪些状态不是最终状态（即有后续状态）
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool, device=self.device)

        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states_graphs = [batch.graph[i] for i, s in enumerate(batch.next_state) if s is not None]

        # 获取当前批次的状态、动作和奖励
        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # graph batch for setting up training batch
        graph_batch = batch.graph

        # 计算当前状态的Q值：模型输出的Q值对于每个状态-动作对
        state_action_values = self.model(self.setup_graph_input(graph_batch, state_batch, action_batch)).squeeze(dim=1)
        # 初始化下一个状态的Q值
        next_state_values = torch.zeros(sample_size, device=self.device)

        # 如果存在非最终状态
        if len(non_final_next_states) > 0:
            if self.double_dqn: # 如果使用Double DQN策略
                # 获取非最终状态的图结构和状态输入
                batch_non_final = self.setup_graph_input(non_final_next_states_graphs, non_final_next_states)
                # 获取目标网络的Q值，并计算下一个状态的Q值
                next_state_values[non_final_mask] = scatter_max(
                    self.target(batch_non_final).squeeze(dim=1).add_(torch.cat(non_final_next_states).to(self.device) * (-1e5)), 
                    batch_non_final.batch)[0].clamp_(min=0).detach()
            else:
                # 如果不使用Double DQN，直接通过当前模型计算下一个状态的Q值
                batch_non_final = self.setup_graph_input(non_final_next_states_graphs, non_final_next_states)
                next_state_values[non_final_mask] = scatter_max(
                    self.model(batch_non_final).squeeze(dim=1).add_(torch.cat(non_final_next_states).to(self.device) * (-1e5)), 
                    batch_non_final.batch)[0].clamp_(min=0).detach()

        expected_state_action_values = next_state_values * self.gamma ** self.n_step + reward_batch.to(self.device)

        # ***这是强化学习中的关键loss***
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        # if double dqn, update target network if needed
        if self.double_dqn:
            # 将当前模型的权重复制到目标网络
            self.target.load_state_dict(self.model.state_dict())
            return True # 返回True表示目标网络已更新
        return False # 如果不使用Double DQN策略，不更新目标网络

    def save_model(self, file_name):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), os.path.join(cwd, file_name))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'graph'))

class ReplayMemory(object):
    '''random replay memory'''
    def __init__(self, capacity):
        # temporily save 1-step snapshot
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


Agent = DQAgent
