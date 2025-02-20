import numpy as np

def trans_state2index(i, j, grid_edge_length):
    """
    一维坐标转换为二维索引, index = (1~25)
    """
    # 状态索引是从1开始到25. 如果从0到24，则不加1
    return i * grid_edge_length + j + 1

def get_fbd_tgt_nor_state_reward(next_i, next_j, grid_edge_length, forbidden_state, tgt_state=18, r_normal=0, r_forbid=-1, r_tgt=1):

    state = trans_state2index(next_i, next_j, grid_edge_length)

    if state in forbidden_state:
        return r_forbid
    
    elif state == tgt_state:
        return r_tgt

    else:
        return r_normal


def get_state_reward(i, j, a, grid_edge_length, forbidden_state, tgt_state=18, r_normal=0, r_bound=-1, r_forbid=-1, r_tgt=1):
    """
    网格世界中的坐标以左上角为(0, 0)
    """
    # key: a, value: (next_i, next_j)
    i_j_dic = {0: (i-1, j), 1: (i, j+1), 2: (i+1, j), 3: (i, j-1), 4: (i, j)}

    next_i, next_j = i_j_dic[a]
    fbd_tgt_nor_state_reward = get_fbd_tgt_nor_state_reward(next_i, next_j, grid_edge_length, forbidden_state, tgt_state=tgt_state, 
                                                r_normal=r_normal, r_forbid=r_forbid, r_tgt=r_tgt)
    # up, row - 1
    if a == 0:
        if next_i < 0:
            return max(next_i, 0), next_j, r_bound
        
        else:
            return next_i, next_j, fbd_tgt_nor_state_reward
    # right, col + 1
    if a == 1:
        if next_j >= grid_edge_length:
            return next_i, min(next_j, grid_edge_length - 1), r_bound
        
        else:
            return next_i, next_j, fbd_tgt_nor_state_reward
    # down, i + 1
    if a == 2:
        if next_i >= grid_edge_length:
            return min(next_i, grid_edge_length - 1), next_j, r_bound
        
        else:
            return next_i, next_j, fbd_tgt_nor_state_reward
    # left, j - 1
    if a == 3:
        if next_j < 0:
            return next_i, max(next_j, 0), r_bound
        
        else:
            return next_i, next_j, fbd_tgt_nor_state_reward
    # stay
    if a == 4:
        return next_i, next_j, fbd_tgt_nor_state_reward


# 生成一个episode
def gen_episode(pi, episode_length, grid_edge_length, forbidden_state, tgt_state=18, r_normal=0, r_bound=-1, r_forbid=-1, r_tgt=1, mode="sarsa", init_pos=None, end_pos=None, max_len=2000):
    """
    mode: sarsa, sars, srs 
    
    当end_pos存在, 则episode_length不起作用
    """
    episode = []
    # init
    if init_pos is None:
        i, j  = np.random.randint(0, grid_edge_length), np.random.randint(0, grid_edge_length)
    else:
        i, j = init_pos

    t = 0
    cnt = 0
    while cnt < episode_length:
        
        # 二维坐标转换为一维索引
        s = trans_state2index(i, j, grid_edge_length)
        if t == 0:
            a = np.random.choice([0, 1, 2, 3, 4], p=pi[s - 1])

        i, j, r = get_state_reward(i, j, a, grid_edge_length, forbidden_state, tgt_state, r_normal, r_bound, r_forbid, r_tgt)
        next_s = trans_state2index(i, j, grid_edge_length)

        next_a = np.random.choice([0, 1, 2, 3, 4], p=pi[next_s - 1])
        
        if mode == "sarsa":
            s_t = f"s{t}_{s}"
            a_t = f"a{t}_{a}"
            r_next_t = f"r{t+1}_{r}"
            s_next_t = f"s{t+1}_{next_s}"
            a_next_t = f"a{t+1}_{next_a}"
            episode.append((s_t, a_t, r_next_t, s_next_t, a_next_t))

        elif mode == "sars":
            s_t = f"s{t}_{s}"
            a_t = f"a{t}_{a}"
            r_next_t = f"r{t+1}_{r}"
            s_next_t = f"s{t+1}_{next_s}"
            episode.append((s_t, a_t, r_next_t, s_next_t))

        elif mode == "srs":
            s_t = f"s{t}_{s}"
            r_next_t = f"r{t+1}_{r}"
            s_next_t = f"s{t+1}_{next_s}"
            episode.append((s_t, r_next_t, s_next_t))

        else:
            raise Exception("Mode Error!")
        
        s = next_s
        a = next_a

        if end_pos is not None:
            cnt -= 1
            if s == trans_state2index(end_pos[0], end_pos[1], grid_edge_length) or t > max_len:
                return episode
        cnt += 1    
        t += 1
    return episode


# 生成多个episode
def gen_multi_episodes(n_episodes, pi, episode_length, grid_edge_length, forbidden_state, 
                      tgt_state=18, r_normal=0, r_bound=-1, r_forbid=-1, r_tgt=1, mode="sarsa", init_pos=None, end_pos=None, max_len=2000):

    """
    mode: sarsa, sars, srs
    """
    multi_episodes = []
    for n in range(n_episodes):
        multi_episodes.append(gen_episode(pi, episode_length, grid_edge_length, forbidden_state, 
                                              tgt_state, r_normal, r_bound, r_forbid, r_tgt, mode, init_pos, end_pos, max_len))
    return multi_episodes

if __name__ == "__main__":
    ground_truth = np.array([[-3.8, -3.8, -3.6, -3.1, -3.2],
                             [-3.8, -3.8, -3.8, -3.1, -2.9],
                             [-3.6, -3.9, -3.4, -3.2, -2.9],
                             [-3.9, -3.6, -3.4, -2.9, -3.2],
                             [-4.5, -4.2, -3.4, -3.4, -3.5]])
    n_actions = 5
    state_length = 5
    n_state = ground_truth.size    
    forbidden_state = [7, 8, 13, 17, 19, 22]     
    # pi's shape: (25, 5)
    pi = np.zeros(shape=(n_state, n_actions)) + 0.2
    r_forbidden = r_bound = -1
    r_normal = 0
    r_target = 1
    gamma = 0.9
    # episodes = gen_multi_episodes(500, pi, 500, 5, forbidden_state, mode="srs")
    # print(episodes)

    episodes = gen_episode(pi, 10, 5, forbidden_state, mode="sarsa", init_pos=(0,0), end_pos=(3,2))
    print(episodes)
