import numpy as np
import statistics
from multiprocessing import Pool
import random
import time
import utils.graph_utils as graph_utils

random.seed(123)
np.random.seed(123)


class Environment:
    ''' environment that the agents run in '''
    def __init__(self, name, graphs, budget, method='RR', use_cache=False, training=True):
        '''
            method: 'RR' or 'MC'
            use_cache: use cache to speed up
        '''
        # sampled set of graphs
        self.name = name # 环境的名称
        self.graphs = graphs # 输入的图数据集
        # IM
        self.budget = budget # 预算，控制最大可选择的节点数
        self.method = method # 影响力计算方法，'RR' 表示随机传播，'MC' 表示蒙特卡洛方法
        # useful only if run on the same graph multiple times
        self.use_cache = use_cache # 是否使用缓存来加速计算
        if self.use_cache:
            if self.method == 'MC':
                # this may be not needed by cached RR
                # not used for RR
                self.influences = {} # cache source set to influence value mapping
            elif self.method == 'RR':
                self.RRs_dict = {}
        self.training = training # 是否处于训练模式

    def reset_graphs(self, num_graphs=10):
        ''' 重置图集，生成新的图 '''
        raise NotImplementedError()

    def reset(self, idx=None, training=True):
        ''' 重置环境，重新初始化状态 '''
        if idx is None:
            self.graph = random.choice(self.graphs) # 随机选择一个图
        else:
            self.graph = self.graphs[idx] # 使用指定索引的图
        self.state = [0 for _ in range(self.graph.num_nodes)] # 初始化所有节点的状态为0
        # IM
        self.prev_inf = 0 # 初始化前一个影响力值
        # store RR sets in case there are more than one graph
        if self.use_cache and self.method == 'RR':
            self.RRs = self.RRs_dict.setdefault(id(self.graph), []) # id有点类似于地址
        self.states = []
        self.actions = []
        self.rewards = []
        self.training = training

    def compute_reward(self, S):
        ''' 计算选择节点集合S，S是actions，的奖励 '''
        num_process = 5 # 使用的并行进程数
        num_trial = 10000 # 每次计算的试验次数
        # fetch influence value
        need_compute = True # 是否需要计算新的影响力
        if self.use_cache and self.method == 'MC':
            S_str = f"{id(self.graph)}.{','.join(map(str, sorted(S)))}"
            need_compute = S_str not in self.influences  # 如果缓存中没有该节点集合的影响力，则需要计算

        if need_compute:
            if self.method == 'MC':
                with Pool(num_process) as p:
                    # 使用蒙特卡洛方法计算影响力
                    es_inf = statistics.mean(p.map(graph_utils.workerMC, 
                        [[self.graph, S, int(num_trial / num_process)] for _ in range(num_process)]))
            elif self.method == 'RR':
                if self.use_cache:
                    # cached without incremental
                    es_inf = graph_utils.computeRR(self.graph, S, num_trial, cache=self.RRs)
                else:
                    es_inf = graph_utils.computeRR(self.graph, S, num_trial)
            else:
                raise NotImplementedError(f'{self.method}') # 如果方法未知，抛出异常

            if self.use_cache and self.method == 'MC':
                self.influences[S_str] = es_inf
        else:
            es_inf = self.influences[S_str]

        reward = es_inf - self.prev_inf # 奖励是当前影响力与上次影响力的差值
        self.prev_inf = es_inf # 更新前一个影响力值
        # store reward
        self.rewards.append(reward) # 将奖励记录下来

        return reward

    def step(self, node, time_reward=None):
        ''' 执行一步操作，选择一个节点并计算奖励 '''
        # node has already been selected
        if self.state[node] == 1:
            return
        # store state and action
        self.states.append(self.state.copy())
        self.actions.append(node)
        # update state
        self.state[node] = 1
        # calculate reward
        if self.name != 'IM':
            raise NotImplementedError(f'Environment {self.name}')

        S = self.actions # 当前选择的节点集合
        # whether game is over, budget is reached
        done = len(S) >= self.budget # 检查是否已达到预算限制

        if self.training:
            # 如果是训练模式，计算奖励
            reward = self.compute_reward(S)
        else:
            # 如果是测试模式，计算奖励并计算时间（如果提供了time_reward）
            if done:
                if time_reward is not None:
                    start_time = time.time()
                reward = self.compute_reward(S)
                if time_reward is not None:
                    time_reward[0] = time.time() - start_time
            else:
                reward = None

        return (reward, done)
