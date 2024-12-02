import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool

random.seed(123)
np.random.seed(123)


class Graph:
    ''' 图类，表示图结构，包含节点、边、子节点、父节点等信息 '''
    def __init__(self, nodes, edges, children, parents): 
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
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge] # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

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


def read_graph(path, ind=0, directed=False):
    ''' 从文件读取图数据，并返回Graph对象 '''
    parents = {}
    children = {}
    edges = {}
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split()
            src = int(row[0]) - ind # 将节点编号调整为从0开始
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst) # 添加子节点
            parents.setdefault(dst, set()).add(src) # 添加父节点
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
        R: 试验次数 每次计算的试验次数/使用的并行进程数
    '''
    sources = set(S)
    inf = 0 # 期望影响力
    for _ in range(R):
        source_set = sources.copy() # 初始种子集合
        queue = deque(source_set) # 使用队列进行广度优先传播
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft() # 弹出当前节点
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set # 更新源节点集合
        inf += len(source_set) # 累加影响力
        
    return inf / R # 返回平均影响力

def workerMC(x):
    ''' 用于并行处理MC计算 '''
    return computeMC(x[0], x[1], x[2])

def computeRR(graph, S, R, cache=None):
    ''' compute expected influence using RR under IC
        R: number of trials
        The generated RR sets are not saved; 
        We can save those RR sets, then we can use those RR sets
            for any seed set
        cache: maybe already generated list of RR sets for the graph
        l_c: a list of RR set covered, to compute the incremental score
            for environment step
    '''
    # generate RR set
    covered = 0
    generate_RR = False # 是否生成新的RR集合
    if cache is not None:
        if len(cache) > 0:
            # might use break for efficiency for large seed set size or number of RR sets
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True # 如果缓存为空，则生成新的随机游走集合

    for i in range(R):
        # generate one set
        source_set = {random.randint(0, graph.num_nodes - 1)} # 随机选择一个节点作为初始源节点
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
            source_set |= curr_source_set # 更新源节点集合
        # compute covered(RR) / number(RR)
        for s in S:
            if s in source_set:
                covered += 1 # 如果源节点集合中包含S中的元素，计算覆盖
                break
        if generate_RR:
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
