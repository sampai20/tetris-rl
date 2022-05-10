import gym
from gym import spaces
import numpy as np
from TetrisEngine import TetrisEngine

class TetrisEnv(gym.Env):
    metadata = {"render-modes": ["human"], "render_fps": 30}

    def __init__(self):

        self.engine = TetrisEngine()
        self.observation_space = spaces.Dict(
                {
                    "board": spaces.MultiBinary([self.engine.BOARD_WIDTH, self.engine.BOARD_HEIGHT + 3]),
                    "held": spaces.MultiBinary(7),
                    "used_held": spaces.Discrete(2),
                    "current": spaces.MultiBinary(7),
                    "previews": spaces.MultiBinary([5, 7]),
                    "piece_mask": spaces.MultiBinary([self.engine.BOARD_WIDTH, self.engine.BOARD_HEIGHT + 3]),
                }
        )
        self.action_space = spaces.Discrete(9)

        self.PENALTY_COEFFS = [
                -0.5, # spikiness
                -0.5, # holes
                -0.5 # height
        ]

        self.reset()

    def _get_obs(self, engine = None):

        if engine is None:
            engine = self.engine

        obs = {}
        obs["board"] = np.where(engine.board == -1, 0, 1)
        obs["held"] = np.where(np.arange(0, 7) == engine.held_piece, 1, 0)
        obs["used_held"] = np.array([1]) if engine.used_held else np.array([0])

        current_piece = engine.bags[0][engine.piece_index]
        obs["current"] = np.where(np.arange(0, 7) == current_piece, 1, 0)

        previews = np.expand_dims(engine.get_previews(), 1)
        obs["previews"] = np.where(np.expand_dims(np.arange(0, 7), 0) == previews, 1, 0)

        obs["heuristic"], obs["holes"] = self._board_potential(obs["board"])

        



        return obs

    def _get_next_obs(self):

        next_states = self.engine._get_next_states()
        return [(self._get_obs(e), m, r - (20 if e.game_over else 0), e.game_over) for e, m, r in next_states]

    def _get_next_obs_hold(self):

        next_states = self.engine._get_next_states_hold()
        return [(self._get_obs(e), m, r - (20 if e.game_over else 0), e.game_over) for e, m, r in next_states]

    def _get_info(self):
        return {}

    def _board_potential(self, board):
        # compute max of board interface
        highest = board.shape[1] - np.argmax(np.flip(board, 1), axis=1)
        highest = np.where(highest == board.shape[1], 0, highest)

        # holes penalty
        holes = (np.sum(highest) - np.sum(board))

        hole_pos = np.zeros(board.shape[0])
        for i in range(board.shape[0]):
            if highest[i] != 0:
                hole_pos[i] = highest[i] - np.argmin(np.flip(board[i, :highest[i]]))
                if hole_pos[i] == highest[i]:
                    hole_pos[i] = 0

        return np.concatenate((highest, np.array([holes]))), hole_pos


    def reset(self, seed=None, return_info=False, options=None):
        # seed RNG
        super().reset(seed=seed)

        # Reset TetrisEngine instance
        self.engine.reset()

        observation, info = self._get_obs(), self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        attack = self.engine.handle_action(action)
        done = self.engine.game_over
        observation, info = self._get_obs(), self._get_info()

        if(done):
            attack -= 20

        return observation, attack, done, info

    def render(self, mode="human"):
        self.engine.render()

    def close(self):
        self.engine.close()


if __name__ == '__main__':
    import random

    env = TetrisEnv()
    env.reset()
    for i in range(100):
        next_states = env._get_next_obs()
        action_id = random.randint(0, len(next_states) - 1)
        for m in next_states[action_id][1]:
            env.step(m)

        print(np.all(env._get_obs()['board'] == next_states[action_id][0]['board']))



    
