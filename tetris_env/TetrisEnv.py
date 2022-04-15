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

    def _get_obs(self):
        obs = {}
        obs["board"] = np.where(self.engine.board == -1, 0, 1)
        obs["held"] = np.where(np.arange(0, 7) == self.engine.held_piece, 1, 0)
        obs["used_held"] = 0 if self.engine.used_held else 1

        current_piece = self.engine.bags[0][self.engine.piece_index]
        obs["current"] = np.where(np.arange(0, 7) == current_piece, 1, 0)

        previews = np.expand_dims(self.engine.get_previews(), 1)
        obs["previews"] = np.where(np.expand_dims(np.arange(0, 7), 0) == previews, 1, 0)

        piece_squares = self.engine._rotate_offsets(current_piece, self.engine.piece_rot) + self.engine.piece_pos
        obs["piece_mask"] = np.zeros_like(obs["board"])
        obs["piece_mask"][tuple(piece_squares.T)] = 1

        return obs

    def _get_info(self):
        return {}

    def _board_potential(self, board):
        # compute max of board interface
        highest = board.shape[1] - np.argmax(np.flip(board, 1), axis=1)

        # spikiness penalty (ignore top 2 variances for well)
        diffs = np.abs(np.diff(highest))
        top2 = diffs[np.argpartition(diffs, -2)[-2:]]
        spikiness = self.PENALTY_COEFFS[0] * np.sum(diffs) - np.sum(top2)

        # holes penalty
        holes = self.PENALTY_COEFFS[1] * (np.sum(highest) - np.sum(board))

        # height penalty (comfort zone is around 7 ish)
        max_height = np.max(highest)
        discomfort = self.PENALTY_COEFFS[2] * np.abs(max_height - 7)

        return spikiness + holes + discomfort


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

        return observation, attack, done, info

    def render(self, mode="human"):
        self.engine.render()

    def close(self):
        self.engine.close()


