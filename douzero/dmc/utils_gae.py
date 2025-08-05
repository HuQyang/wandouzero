import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array
from torch.distributions import Categorical

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective,flags.show_action)

def  get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers

# def create_buffers(flags, device_iterator):
#     """
#     We create buffers for different positions as well as
#     for different devices (i.e., GPU). That is, each device
#     will have three buffers for the three positions.
#     """
#     T = flags.unroll_length
#     positions = ['landlord', 'landlord_up', 'landlord_down']
#     buffers = {}
#     for device in device_iterator:
#         buffers[device] = {}
#         for position in positions:
#             # x_dim = 452 if position == 'landlord' else 589
#             x_dim = 385 if position == 'landlord' else 385+137
#             specs = dict(
#                 done=dict(size=(T,), dtype=torch.bool),
#                 episode_return=dict(size=(T,), dtype=torch.float32),
#                 target=dict(size=(T,), dtype=torch.float32),
#                 obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
#                 obs_action=dict(size=(T, 67), dtype=torch.int8),
#                 obs_z=dict(size=(T, 5, 201), dtype=torch.int8),
#                 obs_x_addition=dict(size=(T, 134), dtype=torch.int8),  # <= NEW
#             )
#             _buffers: Buffers = {key: [] for key in specs}
#             for _ in range(flags.num_buffers):
#                 for key in _buffers:
#                     if not device == "cpu":
#                         _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
#                     else:
#                         _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
#                     _buffers[key].append(_buffer)
#             buffers[device][position] = _buffers
#     return buffers

# def act(i, device, free_queue, full_queue, model, buffers, flags):
#     """
#     This function will run forever until we stop it. It will generate
#     data from the environment and send the data to buffer. It uses
#     a free queue and full queue to syncup with the main process.
#     """
#     positions = ['landlord', 'landlord_up', 'landlord_down']
#     try:
#         T = flags.unroll_length
#         log.info('Device %s Actor %i started.', str(device), i)

#         env = create_env(flags)
#         env = Environment(env, device)

#         done_buf = {p: [] for p in positions}
#         episode_return_buf = {p: [] for p in positions}
#         target_buf = {p: [] for p in positions}
#         obs_x_no_action_buf = {p: [] for p in positions}
#         obs_action_buf = {p: [] for p in positions}
#         obs_z_buf = {p: [] for p in positions}
#         size = {p: 0 for p in positions}
#         obs_x_addition_buf = {p: [] for p in positions}  # <= NEW
#         obs_advantage_buf = {p: [] for p in positions}  # <= NEW

#         position, obs, env_output = env.initial()

#         while True:
#             while True:
#                 obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])

#                 obs_x_addition_buf[position].append(env_output['obs_x_addition'])  # <= NEW

#                 obs_advantage_buf[position].append(env_output['advantage'])  # <= NEW

#                 obs_z_buf[position].append(env_output['obs_z'])
#                 with torch.no_grad():
#                     agent_output = model.forward(position, obs['z_batch'], obs['x_batch'],obs['x_addition_batch'],return_value=False, flags=flags)
#                 _action_idx = int(agent_output['action'].cpu().detach().numpy())
#                 action = obs['legal_actions'][_action_idx]
#                 print("action: ", len(action),action)
#                 # action = select_action(agent,obs['z_batch'], obs['x_batch'],obs['x_addition_batch'], obs['legal_actions'])
                
#                 obs_action_buf[position].append(_cards2tensor(action))
#                 size[position] += 1
#                 position, obs, env_output = env.step(action)
#                 # print("env_output: ", position,env_output['done'])
#                 if env_output['done']:
#                     for p in positions:
#                         diff = size[p] - len(target_buf[p])
#                         if diff > 0:
#                             done_buf[p].extend([False for _ in range(diff-1)])
#                             done_buf[p].append(True)

#                             episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
#                             episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
#                             episode_return_buf[p].append(episode_return)
#                             target_buf[p].extend([episode_return for _ in range(diff)])
#                     break

#             for p in positions:
#                 while size[p] > T: 
#                     index = free_queue[p].get()
#                     if index is None:
#                         break
#                     for t in range(T):
#                         buffers[p]['done'][index][t, ...] = done_buf[p][t]
#                         buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
#                         buffers[p]['target'][index][t, ...] = target_buf[p][t]
#                         buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
#                         buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
#                         buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
#                         buffers[p]['obs_x_addition'][index][t, ...] = obs_x_addition_buf[p][t]  # <= NEW
#                         buffers[p]['advantage'][index][t, ...] = obs_advantage_buf[p][t]  # <= NEW
                        
#                     full_queue[p].put(index)
#                     done_buf[p] = done_buf[p][T:]
#                     episode_return_buf[p] = episode_return_buf[p][T:]
#                     target_buf[p] = target_buf[p][T:]
#                     obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
#                     obs_action_buf[p] = obs_action_buf[p][T:]
#                     print("obs_z_buf[p]: ", len(obs_z_buf[p]), obs_z_buf[p][0].shape)
#                     obs_z_buf[p] = obs_z_buf[p][T:]
#                     obs_x_addition_buf[p] = obs_x_addition_buf[p][T:]  # <= NEW
#                     size[p] -= T

#     except KeyboardInterrupt:
#         pass  
#     except Exception as e:
#         log.error('Exception in worker process %i', i)
#         traceback.print_exc()
#         print()
#         raise e


def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 385 if position == 'landlord' else 385+137
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 67), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 201), dtype=torch.int8),
                obs_x_addition=dict(size=(T, 134), dtype=torch.int8),
                advantage=dict(size=(T,), dtype=torch.float32),  # 添加advantage buffer
                rewards=dict(size=(T,), dtype=torch.float32),    # 添加rewards buffer
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

# def act(i, device, free_queue, full_queue, model, buffers, flags):

#     positions = ['landlord', 'landlord_up', 'landlord_down']
#     try:
#         T = flags.unroll_length
#         log.info('Device %s Actor %i started.', str(device), i)

#         env = create_env(flags)
#         env = Environment(env, device)

#         # 基础缓存区
#         done_buf = {p: [] for p in positions}
#         episode_return_buf = {p: [] for p in positions}
#         target_buf = {p: [] for p in positions}
#         obs_x_no_action_buf = {p: [] for p in positions}
#         obs_action_buf = {p: [] for p in positions}
#         obs_z_buf = {p: [] for p in positions}
#         obs_x_addition_buf = {p: [] for p in positions}
#         advantages_buf = {p: [] for p in positions}
#         rewards_buf = {p: [] for p in positions}
        
#         size = {p: 0 for p in positions}

#         position, obs, env_output = env.initial()

#         while True:
#             # 轻量级episode数据收集 - 只存储必要信息
#             episode_data = {
#                 'positions': [],
#                 'values': {p: [] for p in positions},
#                 'obs_counts': {p: 0 for p in positions},
#                 'step_count': 0
#             }
            
#             while True:
#                 episode_data['positions'].append(position)
#                 episode_data['step_count'] += 1
#                 episode_data['obs_counts'][position] += 1
                
#                 # 存储observation数据到buffer
#                 obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
#                 obs_z_buf[position].append(env_output['obs_z'])
#                 obs_x_addition_buf[position].append(env_output['obs_x_addition'])
                
#                 # 临时解决方案：分别调用action和value
#                 with torch.no_grad():
#                     # 先获取action
#                     action_output = model.forward(position, obs['z_batch'], obs['x_batch'], 
#                                                 obs['x_addition_batch'], return_value=False, flags=flags)
#                     action_idx = action_output['action'].cpu().item()
#                     action_idx = min(max(action_idx, 0), len(obs['legal_actions']) - 1)
#                     action = obs['legal_actions'][action_idx]
                    
#                     # 再获取value - 只用第一个sample来估计state value
#                     value_output = model.forward(position, obs['z_batch'][:1], obs['x_batch'][:1], 
#                                                obs['x_addition_batch'][:1], return_value=True, flags=flags)
#                     value = value_output['critic_value']
#                     current_value = value.item() if value.numel() == 1 else value.view(-1)[0].item()
                
#                 episode_data['values'][position].append(current_value)
#                 obs_action_buf[position].append(_cards2tensor(action))
#                 size[position] += 1
                
#                 # 执行action
#                 position, obs, env_output = env.step(action)
                
#                 if env_output['done']:
#                     break
                    
#                 # 检查obs是否有效
#                 if obs is None:
#                     print("Warning: obs is None after step, breaking episode")
#                     break

#             # Oracle reward计算在episode结束后进行
#             if getattr(flags, 'use_oracle_reward', False):
#                 try:
#                     oracle_rewards = env.env._get_oracle_reward()
#                     # 确保oracle_rewards是列表且长度正确
#                     if not isinstance(oracle_rewards, list):
#                         oracle_rewards = [oracle_rewards] * episode_data['step_count']
#                     elif len(oracle_rewards) != episode_data['step_count']:
#                         if len(oracle_rewards) < episode_data['step_count']:
#                             oracle_rewards.extend([0.0] * (episode_data['step_count'] - len(oracle_rewards)))
#                         else:
#                             oracle_rewards = oracle_rewards[:episode_data['step_count']]
#                 except Exception as e:
#                     print(f"Oracle reward calculation failed: {e}, using sparse reward")
#                     final_reward = env.env._get_reward()
#                     oracle_rewards = [0.0] * (episode_data['step_count'] - 1) + [final_reward]
#             else:
#                 # 使用sparse reward
#                 final_reward = env.env._get_reward()
#                 oracle_rewards = [0.0] * (episode_data['step_count'] - 1) + [final_reward]
            
#             # 为每个position分配数据
#             step_idx_by_pos = {p: 0 for p in positions}
            
#             for global_step, step_position in enumerate(episode_data['positions']):
#                 local_step = step_idx_by_pos[step_position]
                
#                 if local_step < len(episode_data['values'][step_position]):
#                     # 分配reward
#                     if step_position == 'landlord':
#                         reward = oracle_rewards[global_step]
#                     else:
#                         reward = -oracle_rewards[global_step]
                    
#                     rewards_buf[step_position].append(reward)
                    
#                     # 简化的advantage计算 - 使用TD error近似
#                     value = episode_data['values'][step_position][local_step]
#                     if local_step == len(episode_data['values'][step_position]) - 1:
#                         # 最后一步
#                         next_value = 0.0
#                         done = True
#                     else:
#                         next_value = episode_data['values'][step_position][local_step + 1]
#                         done = False
                    
#                     # 简单的TD advantage
#                     gamma = getattr(flags, 'gamma', 0.99)
#                     advantage = reward + gamma * next_value * (1 - done) - value
#                     advantages_buf[step_position].append(advantage)
                    
#                     # TD target
#                     target = reward + gamma * next_value * (1 - done)
#                     target_buf[step_position].append(target)
                    
#                     # Episode信息
#                     done_buf[step_position].append(done)
#                     episode_return = sum(rewards_buf[step_position]) if done else 0.0
#                     episode_return_buf[step_position].append(episode_return)
                    
#                 step_idx_by_pos[step_position] += 1

#             # 快速buffer传输
#             for p in positions:
#                 while size[p] >= T:
#                     index = free_queue[p].get()
#                     if index is None:
#                         break
                    
#                     # 批量传输数据
#                     buffers[p]['done'][index][:T] = torch.tensor(done_buf[p][:T], dtype=torch.bool)
#                     buffers[p]['episode_return'][index][:T] = torch.tensor(episode_return_buf[p][:T], dtype=torch.float32)
#                     buffers[p]['target'][index][:T] = torch.tensor(target_buf[p][:T], dtype=torch.float32)
#                     buffers[p]['advantage'][index][:T] = torch.tensor(advantages_buf[p][:T], dtype=torch.float32)
#                     buffers[p]['rewards'][index][:T] = torch.tensor(rewards_buf[p][:T], dtype=torch.float32)
                    
#                     # observation数据
#                     for t in range(T):
#                         buffers[p]['obs_x_no_action'][index][t] = obs_x_no_action_buf[p][t]
#                         buffers[p]['obs_action'][index][t] = obs_action_buf[p][t]
#                         buffers[p]['obs_z'][index][t] = obs_z_buf[p][t]
#                         buffers[p]['obs_x_addition'][index][t] = obs_x_addition_buf[p][t]
                    
#                     full_queue[p].put(index)
                    
#                     # 批量清理数据
#                     for buf_name in [done_buf, episode_return_buf, target_buf, obs_x_no_action_buf, 
#                                    obs_action_buf, obs_z_buf, obs_x_addition_buf, advantages_buf, rewards_buf]:
#                         buf_name[p] = buf_name[p][T:]
                    
#                     size[p] -= T

#     except KeyboardInterrupt:
#         pass  
#     except Exception as e:
#         log.error('Exception in worker process %i', i)
#         traceback.print_exc()
#         print()
#         raise e
                
def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    高性能版本：合并模型调用，减少GPU传输
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        # 基础缓存区
        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        obs_x_addition_buf = {p: [] for p in positions}
        advantages_buf = {p: [] for p in positions}
        rewards_buf = {p: [] for p in positions}
        
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            # 只收集必要信息，不做复杂计算
            episode_positions = []
            episode_values = {p: [] for p in positions}
            step_count = 0
            
            while True:
                episode_positions.append(position)
                step_count += 1
                
                # 存储observation数据
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                obs_x_addition_buf[position].append(env_output['obs_x_addition'])
                
                # 关键优化：合并模型调用，一次获取action和value
                with torch.no_grad():
                    # 修改模型让它一次返回所有需要的信息
                    agent_output = model.forward(position, obs['z_batch'][:1], obs['x_batch'][:1], 
                                               obs['x_addition_batch'][:1], return_value=True, flags=flags)
                    
                    # 从logits中选择action
                    logits = agent_output['logits']
                    if logits.dim() > 1:
                        logits = logits[0]
                    
                    # 使用更快的action selection
                    if hasattr(flags, 'exp_epsilon') and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                        action_idx = np.random.randint(len(obs['legal_actions']))
                    else:
                        # 只在legal actions范围内选择
                        if logits.size(0) > len(obs['legal_actions']):
                            legal_logits = logits[:len(obs['legal_actions'])]
                        else:
                            legal_logits = logits
                        action_idx = torch.argmax(legal_logits).item()
                    
                    action_idx = min(action_idx, len(obs['legal_actions']) - 1)
                    action = obs['legal_actions'][action_idx]
                    
                    # 获取value
                    value = agent_output['critic_value']
                    current_value = value.item() if value.numel() == 1 else value.view(-1)[0].item()
                
                episode_values[position].append(current_value)
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                
                # 执行action
                position, obs, env_output = env.step(action)
                
                if env_output['done']:
                    break
                    
                if obs is None:
                    break

            # 快速reward和advantage计算
            if getattr(flags, 'use_oracle_reward', False):
                try:
                    # 使用最快的oracle版本
                    oracle_rewards = env.env._get_oracle_reward_fast()
                    if len(oracle_rewards) != step_count:
                        if len(oracle_rewards) < step_count:
                            oracle_rewards.extend([0.0] * (step_count - len(oracle_rewards)))
                        else:
                            oracle_rewards = oracle_rewards[:step_count]
                except:
                    # fallback到sparse reward
                    final_reward = env.env._get_reward()
                    oracle_rewards = [0.0] * (step_count - 1) + [final_reward]
            else:
                final_reward = env.env._get_reward()
                oracle_rewards = [0.0] * (step_count - 1) + [final_reward]
            
            # 快速数据分配 - 使用向量化操作
            step_idx_by_pos = {p: 0 for p in positions}
            
            for global_step, step_position in enumerate(episode_positions):
                local_step = step_idx_by_pos[step_position]
                
                if local_step < len(episode_values[step_position]):
                    # 简化reward分配
                    reward = oracle_rewards[global_step] if step_position == 'landlord' else -oracle_rewards[global_step]
                    rewards_buf[step_position].append(reward)
                    
                    # 使用更简单的advantage计算
                    value = episode_values[step_position][local_step]
                    if local_step == len(episode_values[step_position]) - 1:
                        advantage = reward - value  # 简化版本
                        done = True
                    else:
                        next_value = episode_values[step_position][local_step + 1]
                        advantage = reward + 0.99 * next_value - value
                        done = False
                    
                    advantages_buf[step_position].append(advantage)
                    target_buf[step_position].append(reward + 0.99 * (next_value if not done else 0))
                    done_buf[step_position].append(done)
                    episode_return_buf[step_position].append(sum(rewards_buf[step_position]) if done else 0.0)
                    
                step_idx_by_pos[step_position] += 1

            # 批量buffer传输 - 减少内存操作
            for p in positions:
                while size[p] >= T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    
                    # 使用切片操作，更快的数据传输
                    buffers[p]['done'][index][:T] = torch.tensor(done_buf[p][:T], dtype=torch.bool)
                    buffers[p]['episode_return'][index][:T] = torch.tensor(episode_return_buf[p][:T], dtype=torch.float32)
                    buffers[p]['target'][index][:T] = torch.tensor(target_buf[p][:T], dtype=torch.float32)
                    buffers[p]['advantage'][index][:T] = torch.tensor(advantages_buf[p][:T], dtype=torch.float32)
                    buffers[p]['rewards'][index][:T] = torch.tensor(rewards_buf[p][:T], dtype=torch.float32)
                    
                    # 批量处理observation数据
                    for t in range(T):
                        buffers[p]['obs_x_no_action'][index][t] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t] = obs_z_buf[p][t]
                        buffers[p]['obs_x_addition'][index][t] = obs_x_addition_buf[p][t]
                    
                    full_queue[p].put(index)
                    
                    # 批量清理 - 使用列表切片
                    for buf_name in [done_buf, episode_return_buf, target_buf, obs_x_no_action_buf, 
                                   obs_action_buf, obs_z_buf, obs_x_addition_buf, advantages_buf, rewards_buf]:
                        buf_name[p] = buf_name[p][T:]
                    
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
            

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix


def compute_gae_advantage(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算GAE (Generalized Advantage Estimation)
    
    Input:
      rewards: Tensor of shape [T], 每步即时奖励 r_t
      values:  Tensor of shape [T+1], 包括从 s_0 到 s_T 的 V_phi(s)  
      dones:   Tensor of shape [T], 1 表示该步后游戏终止
      gamma:   折扣因子
      lam:     GAE参数，控制bias-variance tradeoff
      
    Returns:
      advantages: Tensor [T], 对应每步 A_t
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]   # 如果 done，则不把下一步的价值带入
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        
    return advantages


def compute_gae(batch, gamma: float, lam: float):
    """
    Expects batch to contain:
      - rewards:    Tensor of shape [T]
      - values:     Tensor of shape [T+1]  (last one is V(s_T) bootstrap)
      - dones:      Tensor of shape [T]    (1.0 if terminal)
    It will append into batch:
      - advantages: Tensor of shape [T]
      - returns:    Tensor of shape [T]    ( = advantages + values[:-1] )
    """
    T = batch['rewards'].size(0)
    rewards = batch['rewards']
    values  = batch['values']
    dones   = batch['dones']

    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values[:-1]

    batch['advantages'] = advantages
    batch['returns']    = returns
    return batch

def collect_rollout(env, agent, T):
    batch = {
        'obs': [], 'actions': [], 'log_probs': [],
        'values': [], 'rewards': [], 'dones': []
    }
    state = env.reset()
    for t in range(T):
        obs_tensor = agent.preprocess(state)    # however you turn state → tensor
        probs, value, logits = agent(obs_tensor)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        logp   = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())

        batch['obs'].append(obs_tensor)
        batch['actions'].append(action)
        batch['log_probs'].append(logp)
        batch['values'].append(value)
        batch['rewards'].append(torch.tensor(reward, dtype=torch.float32))
        batch['dones'].append(torch.tensor(done,   dtype=torch.float32))

        state = next_state
        if done:
            break

    # bootstrap final value V(s_T)
    obs_tensor = agent.preprocess(state)
    _, v_T, _ = agent(obs_tensor)
    batch['values'].append(v_T)

    # stack lists into tensors
    for k in batch:
        batch[k] = torch.stack(batch[k], dim=0)

    return batch