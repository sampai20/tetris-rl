from TetrisEnv import TetrisEnv
from models.HeuristicOnly import HeuristicNetwork
from models.ConvBased import ConvBasedNetwork
from models.HeuristicWithHoles import HeuristicHoleNetwork
import time

from dqn import GAMMA, tensor_dict_stack, obs_to_tensor
import torch
import random
import numpy as np
import time
from tqdm import tqdm

device = torch.device('cpu')

# pick model
DQNetwork = HeuristicNetwork

def select_action(model, next_states, use_greedy = True):
    with torch.no_grad():
        obs_stacked = tensor_dict_stack([obs_to_tensor(s[0], device=device) for s in next_states])
        rewards_stacked = torch.FloatTensor([s[2] for s in next_states]).to(device).unsqueeze(1)
        pred_value = model(obs_stacked) + rewards_stacked * GAMMA
        best_action = torch.argmax(torch.squeeze(pred_value))

        return best_action


def run_n_trajs(model_file, num_trajs, max_steps=150, garb_prob = 0.):
    model = DQNetwork()
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()
    env = TetrisEnv()
    env.engine.ATTACK_TABLE = [0, 0, 1, 2, 4]
    score = 0
    state = env.reset()
    pieces = 0
    done = False
    logs = []
    for ep_num in tqdm(range(num_trajs)):
        while True:
            future_states = env._get_next_obs_hold()
            a = select_action(model, future_states)
            moves = future_states[a][1]
            exp_board = future_states[a][0]['board']
            for m in moves:
                state, attack, done, _ = env.step(m)
                score += max(0, attack - 1)
            pieces += 1

            if not np.all(exp_board == state['board']):
                print("panic")

            if random.random() < garb_prob:
                env.engine.add_garbage(random.randint(1, 4))


            if done or pieces >= max_steps:
                logs.append(score)
                score = 0
                pieces = 0
                done = False
                state = env.reset()
                break

    return logs

print(run_n_trajs('data/heur_hold/model_2200', 2))

        
        

