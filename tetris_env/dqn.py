import gym
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from TetrisEnv import TetrisEnv
from models.ConvBased import ConvBasedNetwork
from models.HeuristicOnly import HeuristicNetwork
from models.HeuristicWithHoles import HeuristicHoleNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def obs_to_tensor(obs, device = device):
    torch_obs = {
            'board': torch.FloatTensor(obs['board']).to(device).unsqueeze(0),
            'heuristic' : torch.FloatTensor(obs['heuristic']).to(device),
            'holes' : torch.FloatTensor(obs['holes']).to(device),
            'other_data': torch.cat([torch.flatten(torch.FloatTensor(obs[key])).to(device) for key in obs if key not in ('board', 'heuristic', 'holes')])
    }

    return torch_obs

def tensor_dict_stack(dicts):
    stacked = {
        key : torch.stack([dict[key] for dict in dicts], dim = 0)
        for key in dicts[0]
    }

    return stacked



env = TetrisEnv()

BATCH_SIZE = 512
GAMMA = 0.95
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 10
SAVE_INTERVAL = 100
EPISODE_LEN = 300

# pick model
DQNetwork = HeuristicHoleNetwork

policy_net = DQNetwork().to(device)
save_file = 'data/heur_hole_network/model_4300'
save_dir = 'heur_hole_new'
policy_net.load_state_dict(torch.load(save_file))
target_net = DQNetwork().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(50000)

total_steps = 0

def select_action(next_states, use_greedy = True):
    global total_steps
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * total_steps / EPS_DECAY)
    total_steps += 1
    best_action = None
    with torch.no_grad():
        obs_stacked = tensor_dict_stack([obs_to_tensor(s[0]) for s in next_states])
        rewards_stacked = torch.FloatTensor([s[2] for s in next_states]).to(device).unsqueeze(1)
        pred_value = policy_net(obs_stacked) + rewards_stacked * GAMMA
        best_action = torch.argmax(torch.squeeze(pred_value))

    if sample > eps_threshold or (not use_greedy):
        return best_action, best_action
    else:
        return torch.tensor(random.randrange(len(next_states)), device=device, dtype=torch.long), best_action

episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = tensor_dict_stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = tensor_dict_stack(batch.state)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).squeeze()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

def eval_model():
    env.reset()
    tot_score = 0
    for t in count():
        next_states = env._get_next_obs_hold()
        action, best_action = select_action(next_states, False)
        next_state, moves, reward, done = next_states[action]
        tot_score += reward

        if done or t >= EPISODE_LEN:
            break
        else:
            next_state = obs_to_tensor(next_state)

        # Move to the next state
        for m in moves:
            env.step(m)

    print("EVAL RESULTS: {0} score".format(tot_score))


def main():
    print("starting...")
    num_episodes = 10000
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = obs_to_tensor(env.reset())
        tot_score = 0
        for t in count():
            # Select and perform an action
            next_states = env._get_next_obs_hold()
            action, best_action = select_action(next_states, True)
            next_state, moves, reward, done = next_states[action]
            tot_score += reward

            best_state, _, best_reward, best_done = next_states[best_action]

            if done:
                next_state = None
            else:
                next_state = obs_to_tensor(next_state)

            if best_done:
                best_state = None
            else:
                best_state = obs_to_tensor(best_state)

            best_reward = torch.FloatTensor([best_reward]).to(device)

            # Store the transition in memory
            memory.push(state, best_state, best_reward)

            # Move to the next state
            state = next_state
            for m in moves:
                env.step(m)

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done or t >= EPISODE_LEN:
                episode_durations.append(t + 1)
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * total_steps / EPS_DECAY)
                print(tot_score, episode_durations[-1], i_episode, total_steps, eps_threshold)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % SAVE_INTERVAL == 0:
            torch.save(policy_net.state_dict(), 'data/{0}/model_{1}'.format(save_dir, i_episode))
            eval_model()


if __name__ == '__main__':
    main()












