import numpy as np
from itertools import count
import torch
import rl_agents
import models
import statistics
from tqdm import tqdm
import os
import time
from statistics import mean

# 设置随机种子，确保实验可重复
torch.manual_seed(123)
np.random.seed(123)

class Runner:
    ''' 运行智能体在环境中的训练和测试 '''
    def __init__(self, train_env, test_env, agent, training):
        self.train_env = train_env
        self.test_env = test_env # 测试环境
        self.agent = agent # 智能体
        self.training = training # 是否训练模式

    def play_game(self, num_iterations, epsilon, training=True, time_usage=False, one_time=False):
        ''' play the game num_iterations times 
        Arguments:
            time_usage: off if False; True: print average time usage for seed set generation
            one_time: generate the seed set at once without regenerating embeddings
        '''
        if training:
            self.env = self.train_env # 训练模式，使用训练环境
        else:
            self.env = self.test_env # 测试模式，使用测试环境

        c_rewards = [] # 存储每次游戏的奖励（用于测试过程中的结果评估）
        im_seeds = [] # 存储每次游戏生成的种子（用于记录智能体的行动决策）

        # 如果需要计算时间
        if time_usage:
            total_time = 0.0 # 总时间，用于计算每次生成种子集的平均时间

        for iteration in range(num_iterations):
            # handle multiple graphs for evaluation during training
            if training:
                self.env.reset()

                for i in count():
                    state = torch.tensor(self.env.state, dtype=torch.long)
                    action = self.agent.select_action(self.env.graph, state, epsilon, training=training).item()
                    reward, done = self.env.step(action)
                    # this game is over
                    if done:
                        # memorize the trajectory
                        self.agent.memorize(self.env)
                        break
            else:
                # 测试模式下，依次处理每个图
                for g_idx in range(len(self.env.graphs)):
                    # measure time of generating initial embedding if need to print time
                    # this may prevent the initial embedding generation of rl_agent side:
                    #   the number of deep walk training iterations
                    if time_usage and (id(self.env.graphs[g_idx]) not in self.agent.graph_node_embed):
                        start_time = time.time()
                        self.agent.graph_node_embed[id(self.env.graphs[g_idx])] = models.get_init_node_embed(self.env.graphs[g_idx], 0, self.agent.device) # epochs for initial embedding
                        print(f'Time of generating initial embedding for {self.env.graphs[g_idx].path_graph}: {time.time()-start_time:.2f} seconds')
                        
                    if time_usage:
                        start_time = time.time()
                        time_reward = [0.0] # time of calculating reward, needs to be subtracted
                    else:
                        time_reward = None

                    # 重置环境状态
                    self.env.reset(g_idx, training=training)
                    if one_time: # 如果是一次性
                        state = torch.tensor(self.env.state, dtype=torch.long)
                        actions = self.agent.select_action(self.env.graph, state, epsilon, training=training, budget=self.env.budget).tolist()

                        # no sort of actions selected
                        im_seeds.append(actions)

                        if time_usage:
                            total_time += time.time() - start_time

                        final_reward = self.env.compute_reward(actions)
                        c_rewards.append(final_reward)

                    else:
                        for i in count():
                            state = torch.tensor(self.env.state, dtype=torch.long)
                            action = self.agent.select_action(self.env.graph, state, epsilon, training=training).item()

                            final_reward, done = self.env.step(action, time_reward)
                            # this game is over
                            if done:
                                # no sort of action selected
                                im_seeds.append(self.env.actions)
                                c_rewards.append(final_reward)
                                break
                        if time_usage:
                            total_time += time.time() - start_time - time_reward[0]
        if time_usage:
            print(f'Seed set generation per iteration time usage is: {total_time/num_iterations:.2f} seconds')
        return c_rewards, im_seeds #


    def train(self, num_epoch, model_file, result_file):
        ''' let agent act and learn from the environment '''
        # 预训练阶段
        tqdm.write('Pretraining:') # 输出预训练的提示信息
        self.play_game(1000, 1.0) # 智能体进行1000次游戏，epsilon=1.0（全随机选择动作）

        eps_start = 1.0 # 起始 epsilon 值（完全随机选择动作）
        eps_end = 0.05 # 最终 epsilon 值（逐渐减少随机性，更多地利用学到的策略）
        eps_step = 10000.0 # epsilon 衰减的步长

        # 开始训练过程
        tqdm.write('Starting fitting:') # 输出训练开始的提示信息
        progress_fitting = tqdm(total=num_epoch) # 创建一个进度条，表示训练的进度，最多 num_epoch 个周期
        for epoch in range(num_epoch):
            # 根据当前 epoch 计算 epsilon 的值，逐渐减少 epsilon，减少随机性
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - epoch) / eps_step)

            # 每隔10个epoch进行一次游戏，来收集训练数据
            if epoch % 10 == 0:
                self.play_game(10, eps) # 进行10局游戏，epsilon为当前值

            # 每隔10个epoch，进行一次测试
            if epoch % 10 == 0:
                # test
                # 测试阶段：epsilon=0.0表示完全使用智能体当前的策略，不进行随机探索
                rewards, seeds = self.play_game(1, 0.0, training=False) # 进行1局测试，记录奖励和随机种子
                tqdm.write(f'{epoch}/{num_epoch}: ({str(seeds[0])[1:-1]}) | {rewards[0]}') # 输出测试结果

            # 每隔10个epoch，保存模型
            if epoch % 10 == 0:
                # save model
                self.agent.save_model(model_file + str(epoch)) # 保存模型，文件名包含当前的epoch值

            # 每隔100个epoch，更新目标网络
            if epoch % 100 == 0:
                self.agent.update_target_net() # 更新目标网络，通常是复制当前网络的权重到目标网络

            # train the model
            self.agent.fit()

            # 更新进度条
            progress_fitting.update(1)
        progress_fitting.close() # 关闭进度条

        # 训练完成后，展示测试结果
        rewards, seeds = self.play_game(1, 0.0, training=False) # 进行一次测试，不再探索，使用最终的策略
        tqdm.write(f'{num_epoch}/{num_epoch}: ({str(seeds[0])[1:-1]}) | {rewards[0]}') # 输出最终的测试结果

        self.agent.save_model(model_file)


    def test(self, num_trials=1):
        ''' let agent act in the environment
            num_trials: may need multiple trials to get average
        '''
        print('Generate seeds at one time:', flush=True)
        all_rewards, all_seeds = self.play_game(num_trials, 0.0, False, time_usage = True, one_time = True)
        print(f'Number of trials: {num_trials}')
        print(f'Graph path: {", ".join(g.path_graph for g in self.env.graphs)}')
        cnt = 0
        for a_r, a_s in zip(all_rewards, all_seeds):
            print(f'Seeds: {a_s} | Reward: {a_r}')

            # 如果环境中有多个图，打印一个空行以分隔不同图的结果
            if len(self.env.graphs) > 1:
                cnt += 1
                if cnt == len(self.env.graphs):
                    print('') # 打印换行符
                    cnt = 0 # 重置计数器

        print('Generate seed one by one:', flush=True)
        all_rewards, all_seeds = self.play_game(num_trials, 0.0, False, time_usage = True, one_time = False)
        print(f'Number of trials: {num_trials}')
        print(f'Graph path: {", ".join(g.path_graph for g in self.env.graphs)}')
        cnt = 0
        for a_r, a_s in zip(all_rewards, all_seeds):
            print(f'Seeds: {a_s} | Reward: {a_r}')

            # 如果环境中有多个图，打印一个空行以分隔不同图的结果
            if len(self.env.graphs) > 1:
                cnt += 1
                if cnt == len(self.env.graphs):
                    print('')
                    cnt = 0
