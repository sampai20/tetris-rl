from TetrisEnv import TetrisEnv
from models.HeuristicOnly import HeuristicNetwork
from models.ConvBased import ConvBasedNetwork

from dqn import GAMMA, tensor_dict_stack, obs_to_tensor
import torch
import random
import numpy as np
import time

device = torch.device('cpu')

# pick model
DQNetwork = HeuristicNetwork
model = DQNetwork()
model_save = 'data/heur_hold/model_2200'
model.load_state_dict(torch.load(model_save))
model.eval()

def select_action(next_states, use_greedy = True):
    with torch.no_grad():
        obs_stacked = tensor_dict_stack([obs_to_tensor(s[0], device=device) for s in next_states])
        rewards_stacked = torch.FloatTensor([s[2] for s in next_states]).to(device).unsqueeze(1)
        pred_value = model(obs_stacked) + rewards_stacked * GAMMA
        best_action = torch.argmax(torch.squeeze(pred_value))

        return best_action

def select_greedy(next_states):
    obs_stacked = tensor_dict_stack([obs_to_tensor(s[0], device=device) for s in next_states])
    holes = obs_stacked['heuristic'][:, -1]
    diffs = torch.abs(torch.diff(obs_stacked['heuristic'][:, :-1], dim = 1))
    rewards_stacked = torch.FloatTensor([s[2] for s in next_states]).to(device)
    best_action = torch.argmin(torch.sum(torch.abs(obs_stacked['heuristic']), dim = 1) + torch.sum(diffs, dim = 1))
    return best_action


def run_eval():
    env = TetrisEnv()
    env.engine.ATTACK_TABLE = [0, 0, 1, 2, 4]
    score = 0
    attack = 0
    state = env.reset()
    pieces = 0
    done = False
    GARB_PROB = 0.07
    while True:
        future_states = env._get_next_obs_hold()
        a = select_action(future_states)
        moves = future_states[a][1]
        exp_board = future_states[a][0]['board']
        for m in moves:
            state, attack, done, _ = env.step(m)
            env.render()

        score += attack - 1
        pieces += 1
        print(score, pieces, score / pieces)


        if not np.all(exp_board == state['board']):
            print("panic")

        if random.random() < GARB_PROB:
            env.engine.add_garbage(random.randint(1, 4))


        if done:
            state = env.reset()

run_eval()

        
        

