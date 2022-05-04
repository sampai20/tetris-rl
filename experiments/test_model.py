import gym
import torch
import torch.nn as nn

from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.utils.torch_util import load_state_dict

env = gym.make('CartPole-v1')
env.reset()

ob_size = env.observation_space.shape[0]

actor_body = nn.Sequential(
        nn.Linear(ob_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU()
)

act_size = env.action_space.n
actor = CategoricalPolicy(actor_body,
                          in_features=64,
                          action_dim=act_size)

model_file = "data/CartPole-v1/default/seed_2/model/model_best.pt"
state_dict = torch.load(model_file)
actor.load_state_dict(state_dict['actor_state_dict'])

observation = env.reset()
for _ in range(1000):
    env.render()
    dist = actor(torch.from_numpy(observation))[0]
    observation, reward, done, info = env.step(dist.sample().item())

    if done:
        observation, info = env.reset(return_info=True)

env.close()
