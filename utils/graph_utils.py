import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from numpy.linalg import inv
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool

random.seed(123)
np.random.seed(123)


class Graph:
    ''' 图类，表示图结构，包含节点、边、子节点、父节点等信息 '''
    def __init__(self, nodes, edges, children, parents):
        self.c = 0.15

        self.nodes = nodes # set()节点集合
        self.edges = edges # dict{(src,dst): weight, }，存储每条边的权重
        self.children = children # dict{node: set(), }，每个节点的子节点集合
        self.parents = parents # dict{node: set(), }，每个节点的父节点集合

        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        # 图的节点数和边数
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        # 用于缓存邻接矩阵和其他图的信息
        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])

    def get_prob(self, edge):
        ''' 返回某条边的概率（权重） '''
        return self.edges[edge]

    def get_adj(self):
        ''' 返回图的邻接矩阵，使用稀疏矩阵存储 '''
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes)) # 创建矩阵
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge] # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

    # Structure
    def get_S(self):
        idx_map = {j: i for i, j in enumerate(self.nodes)}
        edges = np.array(list(map(idx_map.get, np.array(self.from_to_edges()).flatten())),
                         dtype=np.int64).reshape(np.array(self.from_to_edges()).shape)
        adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                         shape=(self.num_nodes, self.num_nodes),
                         dtype=np.float64)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
        return eigen_adj


    def from_to_edges(self):
        ''' 返回所有的边列表 [(src, dst), ...] '''
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        ''' 返回带权重的边列表 [(src, dst, weight), ...] '''
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx


def read_graph(path, ind=0, directed=False):
    ''' 从文件读取图数据，并返回Graph对象 '''

    # 初始化父节点、子节点、边和节点的集合
    parents = {} # 用于存储每个节点的父节点集合
    children = {} # 用于存储每个节点的子节点集合
    edges = {} # 用于存储每条边及其权重
    nodes = set() # 用于存储图中的所有节点

    with open(path, 'r') as f:
        for line in f:
            line = line.strip() # 去除行两端的空白字符
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue # 如果行为空或以注释符号（# 或 %）开头，则跳过该行
            row = line.split() # 按空格分割行，获取节点信息
            src = int(row[0]) - ind # 将节点编号调整为从0开始
            dst = int(row[1]) - ind # 目标节点编号
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst) # 将目标节点加入源节点的子节点集合中
            parents.setdefault(dst, set()).add(src) # 将源节点加入目标节点的父节点集合中 将源节点添加到父节点中去
            edges[(src, dst)] = 0.0 # 初始化边的权重为0.0

            if not(directed): # 如果是无向图
                # regard as undirectional
                children.setdefault(dst, set()).add(src) # 添加反向子节点
                parents.setdefault(src, set()).add(dst) # 添加反向父节点
                edges[(dst, src)] = 0.0 # 初始化反向边的权重为0.0

    # 设置边的权重为1/入度（即节点的父节点数）
    for src, dst in edges:
        edges[(src, dst)] = 1.0 / len(parents[dst])
            
    return Graph(nodes, edges, children, parents)

def computeMC(graph, S, R):
    ''' 在IC传播模型的条件下使用MC方法计算影响力
        graph: 图对象，包含节点、边、传播概率等信息
        S: 种子节点集合（初始激活的节点）
        R: 试验次数 每次计算的试验次数/使用的并行进程数
    '''
    sources = set(S) # 将种子节点集合 S 转换为集合类型（去重）
    inf = 0 # 初始化影响力为 0
    for _ in range(R): # 进行 R 次蒙特卡罗试验
        source_set = sources.copy() # 初始化源节点集合，复制种子节点集合
        queue = deque(source_set) # 使用队列进行广度优先传播，初始化队列为源节点集合
        while True:
            curr_source_set = set() # 当前一轮传播后被激活的节点集合
            while len(queue) != 0: # 遍历当前队列中的每一个节点，进行传播
                curr_node = queue.popleft() # 从队列中弹出一个节点
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)]) # 遍历该节点的所有子节点，判断是否传播
            if len(curr_source_set) == 0: # 如果当前一轮没有新的激活节点，结束传播
                break
            queue.extend(curr_source_set) # 更新队列
            source_set |= curr_source_set # 更新源节点集合
        inf += len(source_set) # 累加影响力
        
    return inf / R # 返回平均影响力

def workerMC(x):
    ''' 用于并行处理MC计算 '''
    return computeMC(x[0], x[1], x[2])

def computeRR(graph, S, R, cache=None):
    ''' 计算在IC模型下使用RR（Randomized Reachability）算法的预期影响力
        R: 试验次数
        The generated RR sets are not saved; 
        We can save those RR sets, then we can use those RR sets
            for any seed set
        cache: maybe already generated list of RR sets for the graph
        l_c: a list of RR set covered, to compute the incremental score
            for environment step
    '''
    # generate RR set
    covered = 0 # 统计覆盖的节点数
    generate_RR = False # 是否需要生成新的RR集合
    if cache is not None: # 如果cache不为空，使用缓存中的RR集合
        if len(cache) > 0:
            # 如果cache已经存在数据，直接计算并返回结果
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True # 如果缓存为空，则生成新的随机游走集合

    for i in range(R): # R次反向集覆盖
        # generate one set
        source_set = {random.randint(0, graph.num_nodes - 1)} # 随机选择一个节点作为初始源节点
        queue = deque(source_set) # 使用队列来存储当前的源节点
        while True: # 执行随机游走
            curr_source_set = set() # 当前随机游走步骤的新覆盖的节点集合
            while len(queue) != 0: # 遍历队列中的节点，扩展节点
                curr_node = queue.popleft() # 从队列中取出第一个节点
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)]) # 扩展当前节点的父节点集，如果满足传播概率条件
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set) # 将当前扩展的节点集合加入队列，继续传播
            source_set |= curr_source_set # 更新源节点集合 并集
        # compute covered(RR) / number(RR)
        for s in S: # S 表示的是action
            if s in source_set:
                covered += 1 # 如果源节点集合中包含S中的元素，计算覆盖
                break # 一旦S中的一个元素被覆盖，跳出循环
        if generate_RR: # 如果需要生成新的RR集合，将当前生成的集合添加到缓存
            cache.append(source_set) # 如果需要生成新的RR集合，保存生成的集合
    return covered * 1.0 / R * graph.num_nodes # 返回期望影响力


def workerRR(x):
    ''' for multiprocessing '''
    return computeRR(x[0], x[1], x[2])

def computeRR_inc(graph, S, R, cache=None, l_c=None):
    ''' compute expected influence using RR under IC
        R: number of trials
        The generated RR sets are not saved; 
        We can save those RR sets, then we can use those RR sets
            for any seed set
        cache: maybe already generated list of RR sets for the graph
        l_c: a list of RR set covered, to compute the incremental score
            for environment step用于计算增量分数
    '''
    # generate RR set
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            # might use break for efficiency for large seed set size or number of RR sets
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True

    for i in range(R):
        # generate one set
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        # compute covered(RR) / number(RR)
        for s in S:
            if s in source_set:
                covered += 1
                break
        if generate_RR:
            cache.append(source_set)
    return covered * 1.0 / R * graph.num_nodes


if __name__ == '__main__':
    # path of the graph file
    path = "../soc-dolphins.txt"
    # number of parallel processes
    num_process = 5
    # number of trials
    num_trial = 10000
    # load the graph
    graph = read_graph(path, ind=1, directed=False)
    print('Generating seed sets:')
    list_S = []
    for _ in range(10):
      list_S.append(random.sample(range(graph.num_nodes), k=random.randint(3, 10)))
      print(f'({str(list_S[-1])[1:-1]})')

    # cached single-process RR
    print('Cached single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    RR_cache = []
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial, cache=RR_cache))
      times.append(time.time() - time_start)
    time_2 = time.time()

    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    # no-cache single-process RR
    print('No-cache single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    # multi-process MC
    print('Multi-process MC:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      with Pool(num_process) as p:
        es_infs.append(statistics.mean(p.map(workerMC, [[graph, S, num_trial // num_process] for _ in range(num_process)])))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')
