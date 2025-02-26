import numpy as np
import matplotlib.pyplot as plt
from generate_episode import gen_multi_grid_episodes, gen_grid_episode
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import deque


def get_num(item):
    return int(item.split("_")[-1])

def trans_state2ij(item, grid_edge_length=5):
    """
    把状态一维索引转换为二维索引, i,j = (0~24)
    """
    if isinstance(item, str):
        state = get_num(item)
    else:
        state = item
    mod = state % grid_edge_length
    j = mod if mod != 0 else 5
    i = int(((state - j) / grid_edge_length) + 1)
    return (i-1, j-1)

def batch_trans_state2ij(item):
    """
    输入为一维数组
    """
    item = np.vectorize(trans_state2ij)(item)
    return  np.array(list(zip(item[0], item[1])))



def visual_3d(data, z_min=-5, z_max=-2):
    # 假设 pred 已经有数据
    # pred = np.random.rand(5, 5)  # 示例数据替代你的 pred 数组

    # 生成坐标网格（假设是 5x5 的二维数组）
    
    
    x = np.arange(0, 5)
    y = np.arange(0, 5)
    X, Y = np.meshgrid(x, y)  # 生成网格坐标
    Z = data  # 你的二维数组

    # 创建三维画布
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面图
    surf = ax.plot_surface(Y, X, Z, cmap='viridis', edgecolor='k', linewidth=0.5)

    # 固定z轴取值范围
    ax.set_zlim(z_min, z_max)

    # 添加颜色条和标签
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel('row')
    ax.set_ylabel('column')
    ax.set_zlabel('Value')
    ax.set_title('3D Surface Plot')

    # 启用交互模式，允许图表旋转
    plt.ion()

    # 显示图形
    plt.show()

    # 保持图表直到关闭窗口
    plt.ioff()
    
def visual_error(x, y, alpha=None):
    # %matplotlib inline
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    ax.plot(x, y, label=f"TD-Linear:α={alpha}")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("State Value error(RMSE)")
    ax.legend()
    plt.show()

def visual_return(x, y, alpha=None):
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    ax.plot(x, y, label=f"Sarsa:α={alpha}")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Return")
    ax.legend()
    plt.show()

def get_random_sample(experience_samples, batch_szie):

    """
    - 对整个episode采样batch_size个tranjectory  
    
    """
    indices = np.random.randint(0, len(experience_samples), size=(batch_szie))
    sample_experience = experience_samples[indices]

    sars = []
    
    for i, trajectory in enumerate(sample_experience):
        # 状态索引为一维, 值域(1~25)
        s_t = get_num(trajectory[0])
        a_t = get_num(trajectory[1])
        r_next_t = get_num(trajectory[2])
        s_next_t = get_num(trajectory[3])
        sars.append([s_t, a_t, r_next_t, s_next_t])

    return torch.tensor(sars, dtype=torch.float)

    
class Critic(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, s):
        return self.seq(s)
       


class Actor(nn.Module):
    def __init__(self, n_actions, hidden_size=128):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.LogSoftmax(-1)
        )

    def forward(self, s):
        return self.seq(s)
        pass

class ReplayBuffer:
    def __init__(self, capacity=10):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        ids = np.random.choice(len(self.buffer[0]), batch_size)
        return np.asarray(self.buffer[0])[ids]
    

if __name__ =="__main__":

    ground_truth = np.array([[-3.8, -3.8, -3.6, -3.1, -3.2],
                            [-3.8, -3.8, -3.8, -3.1, -2.9],
                            [-3.6, -3.9, -3.4, -3.2, -2.9],
                            [-3.9, -3.6, -3.4, -2.9, -3.2],
                            [-4.5, -4.2, -3.4, -3.4, -3.5]])
    n_actions = 5
    grid_edge_length = 5
    n_states = ground_truth.size    
    forbidden_states = [7 ,8, 13, 17, 19, 22] 
    tgt_state = 18    
    # pi's shape: (25, 5)
    pi = np.zeros(shape=(n_states, n_actions)) + 0.2
    r_forbid = -1
    r_bound = -1
    r_normal = 0
    r_tgt = 1
    gamma = 0.9
    alpha = 1e-6
    beta = 1e-6
    epoch = 200
    batch_szie = 256
    device = "cuda"
    error_ls = []
    G_ls = []
    capacity = 10000

    critic = Critic().to(device)
    actor = Actor(n_actions).to(device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=alpha)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=beta)

    # 初始化Replay Buffer
    rb = ReplayBuffer(capacity)
    # 初始化采样
    experience_samples = gen_grid_episode(pi=pi, episode_length=batch_szie, grid_edge_length=grid_edge_length, forbidden_state=forbidden_states, 
                tgt_state=tgt_state, r_normal=r_normal, r_bound=r_bound, r_forbid=r_forbid, r_tgt=r_tgt, 
                mode="sars", init_pos=(0,0))
    
    rb.add(experience_samples)
    for e in range(epoch):
        for _ in range((len(experience_samples) // batch_szie) + 1):
            data = rb.sample(batch_szie)

            # 把状态索引转换为二维
            s_t = batch_trans_state2ij(data[:, 0])
            a_t = np.vectorize(get_num)(data[:, 1:2])
            r_next_t = np.vectorize(get_num)(data[:, 2:3])
            s_next_t = batch_trans_state2ij(data[:, 3])

            s_t = torch.tensor(s_t, dtype=torch.float, device=device)
            a_t = torch.tensor(a_t, dtype=torch.float, device=device)
            r_next_t = torch.tensor(r_next_t, dtype=torch.float, device=device)
            s_next_t = torch.tensor(s_next_t, dtype=torch.float, device=device)

            # TD-error(Advantege): targets - values
            values = critic(s_t)
            next_values = critic(s_next_t)
            targets = r_next_t + gamma * next_values

            # value update
            # 让values接近targets
            critic_loss = F.mse_loss(values, targets.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # policy update
            out = actor(s_t)
            log_probs = out.gather(-1, a_t.long())
            advantege = targets.detach() - values.detach()
            actor_loss = -(advantege * log_probs).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
                
            print(f"Epoch: {e+1}/{epoch}, TD Error: {critic_loss.mean().item():.15f}, Value: {values.mean().item():.15f}")

            # 更新pi online
            states = torch.tensor(batch_trans_state2ij(np.arange(1, 26)), dtype=torch.float, device=device)
            pi = torch.exp(actor(states)).detach().cpu().numpy()
            # 用更新后的 pi 生成新的数据
            experience_samples = gen_grid_episode(pi=pi, episode_length=batch_szie, grid_edge_length=grid_edge_length, forbidden_state=forbidden_states, 
                    tgt_state=tgt_state, r_normal=r_normal, r_bound=r_bound, r_forbid=r_forbid, r_tgt=r_tgt, 
                    mode="sars", init_pos=(0,0))
            rb.add(experience_samples)



    s2a = pi.argmax(-1)
    a_dic = {0: "上", 1: "右", 2: "下", 3: "左", 4: "停", }
    policy = [(s+1, a_dic[a]) for s, a in enumerate(s2a)]
    print(policy)
    print(pi)
    episode = gen_grid_episode(pi=pi, init_pos=(0,0), end_pos=(3,2))
    print(len(episode))
