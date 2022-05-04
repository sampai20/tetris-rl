import gym
import torch.nn as nn
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../tetris_env")

from TetrisEnv import TetrisEnv

from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env

def make_env():
    return TetrisEnv()

class NetworkBody(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.board_conv = nn.Sequential(
                nn.ConstantPad2d(1, 1),
                nn.Conv2d(1, 16, 5, padding='same'),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 1, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
        )
        
        self.current_conv = nn.Sequential(
                nn.ConstantPad2d(1, 1),
                nn.Conv2d(1, 16, 5, padding='same'),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 1, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
        )

        self.fc = nn.Sequential(
                nn.Linear(86, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
        )


    def forward(self, input):

        held_column = torch.unsqueeze(input['used_held'], 1)

        board_out = self.board_conv(torch.unsqueeze(input['board'], 1))
        mask_out = self.current_conv(torch.unsqueeze(input['piece_mask'], 1))

        other_input = torch.cat(
                tuple(torch.flatten(input[key], start_dim = 1) for key in input if key not in ('board', 'piece_mask', 'used_held')),
                dim = -1
        )

        fc_input = torch.cat((board_out, mask_out, other_input, held_column), dim = -1)

        return self.fc(fc_input)


def main2():
    env = make_env()
    body = NetworkBody()
    obs = env.reset()
    print(body([obs, obs]).shape)
    

def main():
    set_config('ppo')
    cfg_from_cmd(cfg.alg)
    if cfg.alg.resume or cfg.alg.test:
        if cfg.alg.test:
            skip_params = [
                'test_num',
                'num_envs',
                'sample_action',
            ]
        else:
            skip_params = []
        cfg.alg.restore_cfg(skip_params=skip_params)

    if cfg.alg.env_name is None:
        cfg.alg.env_name = 'TetrisEnv'

    set_random_seed(cfg.alg.seed)
    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs,
                       seed=cfg.alg.seed,
                       env_func = make_env)
    env.reset()

    actor_body = NetworkBody()

    critic_body = NetworkBody()

    act_size = env.action_space.n
    actor = CategoricalPolicy(actor_body,
                              in_features=64,
                              action_dim=act_size)


    critic = ValueNet(critic_body, in_features=64)
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
                       runner=runner)
    engine.train()
    env.close()


if __name__ == '__main__':
    main()
