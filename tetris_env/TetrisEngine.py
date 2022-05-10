import numpy as np
import pygame
from copy import deepcopy
from pygame import gfxdraw

class TetrisEngine():
    '''
    Python implementation of Tetris that is Guideline compliant (i.e. uses SRS rotation and Guideline piece spawning)
    Kick tables / SRS algorithm taken from https://tetris.wiki/Super_Rotation_System
    The action space consists of:
    - 0 (1): DAS left (right), i.e. send piece maximally to the left (right).
    - 2 (3): rotate left (right) using SRS rules.
    - 4: soft drop, i.e. send piece maximally down.
    - 5: hard drop, i.e. send piece maximally down and place.
    - 6: hold piece, i.e. swap out piece with currently held piece.
    '''
    def __init__(self, board=None):
        self._init_constants()
        self.reset(board)

    def _init_constants(self):
        KICK_TABLE_DEFAULT = np.array([
                [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
                [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
                [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
                [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)]
        ])
        KICK_TABLE_I = np.array([
                [(0, 0), (-1, 0), (2, 0), (-1, 0), (2, 0)],
                [(0, 1), (0, 1), (0, 1), (0, -1), (0, 2)],
                [(-1, 1), (1, 1), (-2, 1), (1, 0), (-2, 0)],
                [(-1, 0), (0, 0), (0, 0), (0, 1), (0, -2)]
        ])
        KICK_TABLE_O = np.array([
                [(0, 0)],
                [(-1, 0)],
                [(-1, -1)],
                [(0, -1)]
        ])
        self.PIECE_LETTERS = ['J', 'L', 'S', 'T', 'Z', 'I', 'O', 'G']
        self.KICK_TABLES = [
                KICK_TABLE_DEFAULT,
                KICK_TABLE_DEFAULT,
                KICK_TABLE_DEFAULT,
                KICK_TABLE_DEFAULT,
                KICK_TABLE_DEFAULT,
                KICK_TABLE_I,
                KICK_TABLE_O
        ]
        self.PIECE_OFFSETS = [
                np.array([(0, 0), (-1, 0), (-1, 1), (1, 0)]),
                np.array([(0, 0), (-1, 0), (1, 0), (1, 1)]),
                np.array([(0, 0), (-1, 0), (0, 1), (1, 1)]),
                np.array([(0, 0), (-1, 0), (0, 1), (1, 0)]),
                np.array([(0, 0), (0, 1), (-1, 1), (1, 0)]),
                np.array([(0, 0), (-1, 0), (1, 0), (2, 0)]),
                np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
        ]
        self.PIECE_COLORS = [
                (0, 0, 255),
                (255, 127, 0),
                (0, 255, 0),
                (127, 0, 127),
                (255, 0, 0),
                (0, 255, 255),
                (255, 255, 0),
                (127, 127, 127)
        ]
        self.COMBO_TABLE = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5]
        self.ATTACK_TABLE = [0, 1, 3, 6, 10]
        self.T_SPIN_MULTIPLIER = 2
        self.B2B_BONUS = 1
        self.BOARD_WIDTH = 10
        self.BOARD_HEIGHT = 20


    def reset(self, board=None):
        if board is None:
            self.board = np.full((self.BOARD_WIDTH, self.BOARD_HEIGHT + 3), -1)
        else:
            self.board = board

        # initialize piece
        self.bags = (np.random.permutation(7), np.random.permutation(7)) # type: ignore
        self.piece_index = 0
        self.piece_pos = np.array([self.BOARD_WIDTH // 2 - 1, self.BOARD_HEIGHT - 1])
        self.piece_rot = 0 # starts at original, increases CCW modulo 4

        # held piece
        self.held_piece = None
        self.used_held = False
        self.game_over = False

        # attack
        self.combo = 0
        self.attack = 0
        self.back_to_back = False

        # t-spin tracking
        self.last_rotate = False
        self.last_kick = False

        self.screen = None
        self.clock = None

    def copy(self, screen = False):
        new_engine = TetrisEngine()
        new_engine.board = self.board.copy()
        new_engine.bags = deepcopy(self.bags)
        new_engine.piece_index = self.piece_index
        new_engine.piece_pos = deepcopy(self.piece_pos)
        new_engine.piece_rot = self.piece_rot
        new_engine.held_piece = self.held_piece
        new_engine.used_held = self.used_held
        new_engine.game_over = self.game_over
        new_engine.combo = self.combo
        new_engine.attack = self.attack
        new_engine.back_to_back = self.back_to_back
        new_engine.last_rotate = self.last_rotate
        new_engine.last_kick = self.last_kick

        if screen:
            new_engine.screen = self.screen
            new_engine.clock = self.clock

        return new_engine

    def _rotate(self, pos, rot):
        dx, dy = pos[0], pos[1]
        if rot == 0:
            return pos
        if rot == 1:
            return np.array([-dy, dx])
        if rot == 2:
            return np.array([-dx, -dy])
        if rot == 3:
            return np.array([dy, -dx])
        raise ValueError("rotation amount must be between 0 and 3 inclusive")
    
    def _rotate_offsets(self, piece, rot):
        offsets = self.PIECE_OFFSETS[piece]
        return np.array([self._rotate(offset, rot) for offset in offsets])

    def _check_piece(self, squares):
        if np.any(squares < 0) or np.any(squares >= [self.BOARD_WIDTH, self.BOARD_HEIGHT + 3]):
            return False
        return np.all(self.board[tuple(squares.T)] == -1)

    def _piece_shift(self, delta):
        offsets = self._rotate_offsets(self.bags[0][self.piece_index], self.piece_rot)
        offsets += self.piece_pos + delta
        if self._check_piece(offsets):
            self.piece_pos += delta
            self.last_rotate, self.last_kick = False, False
            return True
        return False
        
    def _max_piece_shift(self, delta):
        offsets = self._rotate_offsets(self.bags[0][self.piece_index], self.piece_rot)
        offsets += self.piece_pos
        num_shifts = 0
        while True:
            offsets += delta
            if self._check_piece(offsets):
                num_shifts += 1
            else:
                break

        self.piece_pos += num_shifts * delta
        if num_shifts:
            self.last_rotate, self.last_kick = False, False
        return num_shifts

    def _rotate_piece(self, rot):
        piece = self.bags[0][self.piece_index]
        cur_kick = self.KICK_TABLES[piece][self.piece_rot]
        next_kick = self.KICK_TABLES[piece][(self.piece_rot + rot) % 4]
        kick_deltas = cur_kick - next_kick
        offsets = self._rotate_offsets(piece, (self.piece_rot + rot) % 4)
        for i in range(kick_deltas.shape[0]):
            if self._check_piece(offsets + kick_deltas[i] + self.piece_pos):
                self.piece_pos += kick_deltas[i]
                self.piece_rot = (self.piece_rot + rot) % 4
                self.last_rotate = True
                if i == 4:
                    self.last_kick = True
                return True
        return False

    def _new_piece(self):
        self.piece_index += 1
        if self.piece_index == 7:
            self.bags = (self.bags[1], np.random.permutation(7)) # type: ignore
            self.piece_index = 0
        self.piece_rot = 0
        self.piece_pos = np.array([self.BOARD_WIDTH // 2 - 1, self.BOARD_HEIGHT - 1])
        self.used_held = False
        self.last_rotate, self.last_kick = False, False

        squares = self.PIECE_OFFSETS[self.bags[0][self.piece_index]] + self.piece_pos # type: ignore
        if not self._check_piece(squares):
            self.game_over = True

    def _clear_lines(self):
        line_mask = np.any(self.board == -1, axis=0)
        num_cleared = self.board.shape[1] - line_mask.sum()
        if num_cleared:
            new_board = np.pad(self.board[:, line_mask], ((0, 0), (0, num_cleared)), 'constant', constant_values=(0, -1)) # type: ignore
            self.board = new_board
        return num_cleared

    def _hard_drop(self):
        self._max_piece_shift(np.array([0, -1]))
        squares = self._rotate_offsets(self.bags[0][self.piece_index], self.piece_rot) + self.piece_pos
        self.board[tuple(squares.T)] = self.bags[0][self.piece_index]

        # check for t-spin
        tspin = False
        if self.PIECE_LETTERS[self.bags[0][self.piece_index]] == 'T':
            corners = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
            proper, total = 0, 0
            for i in range(4):
                square = self.piece_pos + corners[i]
                if np.any(square < [0, 0]) or square[0] >= self.BOARD_WIDTH or square[1] >= self.board.shape[1] or self.board[tuple(square)] != -1:
                    total += 1
                    if i == self.piece_rot or i == (self.piece_rot + 1) % 4:
                        proper += 1
            tspin = self.last_rotate and ((total >= 3 and proper == 2) or (total >= 3 and self.last_kick))

        num_clear = self._clear_lines()
        reward = 0

        if num_clear == 0:
            self.combo = 0
        else:
            if tspin:
                reward += 2 * self.ATTACK_TABLE[num_clear]
            else:
                reward += self.ATTACK_TABLE[num_clear] # type: ignore

            reward += self.COMBO_TABLE[min(self.combo, len(self.COMBO_TABLE))]
            self.combo += 1
            
            if tspin or num_clear == 4:
                if self.back_to_back:
                    self.attack += 1
                self.back_to_back = True
            else:
                self.back_to_back = False

        self._new_piece()

        self.attack += reward
        return reward + 1

    def _hold_piece(self):
        if self.used_held:
            return False
        if self.held_piece is None:
            self.held_piece = self.bags[0][self.piece_index]
            self._new_piece()
        else:
            self.held_piece, self.bags[0][self.piece_index] = self.bags[0][self.piece_index], self.held_piece
            self.piece_index -= 1
            self._new_piece()
        self.used_held = True
        return True

    def _expand_shifts(self, setup = tuple()):
        cur_state = self.copy()
        cur_state.handle_action(0)
        states = [(cur_state.copy(), setup + (0,))]
        while True:
            if cur_state._piece_shift(np.array([1, 0])):
                states.append((cur_state.copy(), states[-1][1] + (3,)))
            else:
                return states

    def _expand_rotations(self, setup = tuple()):
        states = [(self.copy(), setup)]

        rot_left = self.copy()
        if rot_left._rotate_piece(1):
            states.append((rot_left, setup + (4,)))

        rot_right = self.copy()
        if rot_right._rotate_piece(3):
            states.append((rot_right, setup + (5,)))
        
        rot_twice = self.copy()
        if rot_twice._rotate_piece(1) and rot_twice._rotate_piece(1):
            states.append((rot_twice, setup + (4, 4)))

        return states

    def _push_states_down(self, states):
        _ = [state[0]._max_piece_shift(np.array([0, -1])) for state in states]
        return [(eng, move + (6,)) for eng, move in states]

    def _apply_shifts(self, states):
        new_states = []
        for s, m in states:
            ss = s._expand_shifts(setup = m)
            new_states += ss
        return new_states

    def _apply_rotations(self, states):
        new_states = []
        for s, m in states:
            ss = s._expand_rotations(setup = m)
            new_states += ss
        return new_states

    def _prune(self, states):
        
        cache = set()
        pruned = []
        for s, m in states:
            key = tuple(map(tuple, s._rotate_offsets(s.bags[0][s.piece_index], s.piece_rot) + s.piece_pos))
            key = tuple(sorted(key))
            if key in cache:
                continue
            else:
                cache.add(key)
                pruned.append((s, m))

        return pruned

    def _clear(self, states):
        return [(e, m + (7,), e._hard_drop()) for e, m in states]





    def _get_next_states_old(self):

        states = self._prune(self._expand_rotations())

        ns = self._prune(self._push_states_down(self._apply_rotations(self._push_states_down(self._apply_shifts(states)))))

        return self._clear(ns)

        
    def _get_next_states(self):

        state_squares = lambda s : tuple(sorted(tuple(map(tuple, s._rotate_offsets(s.bags[0][s.piece_index], s.piece_rot) + s.piece_pos))))


        states = [(self.copy(), tuple())]
        vis = set(state_squares(states[0][0]))

        cur_pos = 0
        while cur_pos < len(states):
            cur_state, moves = states[cur_pos]

            #left
            l = cur_state.copy()
            res = l._piece_shift(np.array([-1, 0]))
            if res and not (state_squares(l) in vis):
                states.append((l, moves + (2,)))
                vis.add(state_squares(l))

            #right
            r = cur_state.copy()
            res = r._piece_shift(np.array([1, 0]))
            if res and not (state_squares(r) in vis):
                states.append((r, moves + (3,)))
                vis.add(state_squares(r))

            #down
            d = cur_state.copy()
            res = d._max_piece_shift(np.array([0, -1]))
            if res and not (state_squares(d) in vis):
                states.append((d, moves + (6,)))
                vis.add(state_squares(d))

            #rot left
            rl = cur_state.copy()
            res = rl._rotate_piece(1)
            if res and not (state_squares(rl) in vis):
                states.append((rl, moves + (4,)))
                vis.add(state_squares(rl))

            #rot right
            rr = cur_state.copy()
            res = rr._rotate_piece(3)
            if res and not (state_squares(rr) in vis):
                states.append((rr, moves + (5,)))
                vis.add(state_squares(rr))

            cur_pos += 1

        return self._clear(self._prune(self._push_states_down(states)))

    def _get_next_states_hold(self):

        ans = self._get_next_states()

        hold_state = self.copy()
        if hold_state._hold_piece():
            held_ans = hold_state._get_next_states()
            prefix_held = [(s, (8,) + m, r) for s, m, r in held_ans]

            ans += prefix_held

        return ans






    def get_previews(self, num_previews=5):
        if self.piece_index + num_previews > 6:
            return np.concatenate((self.bags[0][self.piece_index + 1:7], self.bags[1][:num_previews + self.piece_index  - 6])) # type: ignore
        else:
            return self.bags[0][self.piece_index + 1: self.piece_index + num_previews + 1]

    def handle_action(self, action_id):
        '''
        - 0 (1): DAS left (right), i.e. send piece maximally to the left (right).
        - 2 (3): tap left (right), i.e. move once left (right) if possible.
        - 4 (5): rotate left (right) using SRS rules.
        - 6: soft drop, i.e. send piece maximally down.
        - 7: hard drop, i.e. send piece maximally down and place
        - 8: hold piece, i.e. swap out piece with currently held piece.
        '''
        reward = 0

        if action_id == 0:
            self._max_piece_shift(np.array([-1, 0]))
        elif action_id == 1:
            self._max_piece_shift(np.array([1, 0]))
        elif action_id == 2:
            self._piece_shift(np.array([-1, 0]))
        elif action_id == 3:
            self._piece_shift(np.array([1, 0]))
        elif action_id == 4:
            self._rotate_piece(1)
        elif action_id == 5:
            self._rotate_piece(3)
        elif action_id == 6:
            self._max_piece_shift(np.array([0, -1]))
        elif action_id == 7:
            reward = self._hard_drop()
        elif action_id == 8:
            self._hold_piece()

        return reward

    def add_garbage(self, amt):

        garbage_pos = np.random.randint(self.BOARD_WIDTH)
        garb_array = np.ones((self.BOARD_WIDTH, amt), dtype=int) * 7
        garb_array[garbage_pos, :] = -1

        self.board = np.concatenate((garb_array, self.board), axis = 1)[:, :self.BOARD_HEIGHT + 3]

        

    def vis_board(self):
        squares = self._rotate_offsets(self.bags[0][self.piece_index], self.piece_rot) + self.piece_pos
        self.board[tuple(squares.T)] = self.bags[0][self.piece_index]
        board_str = "#" * (self.BOARD_WIDTH + 2) + "\n#"
        for i in range(self.BOARD_HEIGHT, -1, -1):
            for j in range(self.BOARD_WIDTH):
                square = self.board[j, i]
                board_str += (self.PIECE_LETTERS[square] if square != -1 else " ")
            board_str += "#\n#"
        board_str += "#" * (self.BOARD_WIDTH + 1)
        board_str += "\nHeld: {0}".format(self.PIECE_LETTERS[self.held_piece] if self.held_piece else None)
        board_str += "\nPreviews: {0}".format(" ".join(map(lambda x : self.PIECE_LETTERS[x], self.get_previews())))
        board_str += "\nAttack: {0}".format(self.attack)
        self.board[tuple(squares.T)] = -1
        return board_str

    def render(self):
        # modified from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

        screen_width, screen_height = 400, 800
        side = 20
        margin = 10

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        # draw box
        game_area = pygame.Rect(margin, margin, self.BOARD_WIDTH * side, self.BOARD_HEIGHT * side)
        gfxdraw.rectangle(self.surf, game_area, (0, 0, 0))

        # draw board
        squares = self._rotate_offsets(self.bags[0][self.piece_index], self.piece_rot) + self.piece_pos
        self.board[tuple(squares.T)] = self.bags[0][self.piece_index]
        for i in range(self.BOARD_WIDTH):
            for j in range(self.BOARD_HEIGHT):
                square = self.board[i, j]
                if square != -1:
                    box = pygame.Rect((margin + i * side + 1, margin + j * side - 1, side-1, side-1))
                    color = pygame.Color(self.PIECE_COLORS[square])
                    gfxdraw.box(self.surf, box, color) 
        self.board[tuple(squares.T)] = -1


        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        pygame.event.pump()
        self.clock.tick(50)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()



if __name__ == '__main__':
    import time

    board = np.array([
        [7, 7, 7, 7, 7, 7, 7, 7, -1, 7],
        [7, 7, 7, 7, 7, 7, 7, -1, -1, -1],
        [7, 7, 7, 7, 7, 7, 7, 7, -1, -1]
    ]).T
    board = np.pad(board, ((0, 0), (0, 23 - board.shape[1])), mode='constant', constant_values=(-1, -1)) # type: ignore
    engine = TetrisEngine()
    engine.bags[0][0] = 3
    engine._hard_drop()
    engine.render()
    input()
    engine.add_garbage(5)
    engine.render()
    input()
    for state, move, reward in engine._get_next_states():
        engine.board = state.board.copy()
        engine.render()
        print(reward, move)
        state.close()
        time.sleep(0.2)




"""
    while not engine.game_over:
        engine.render()
        # print(engine.vis_board())
        a = input()
        try:
            engine.handle_action(int(a))
        except Exception as e:
            print(e)
        print(engine.held_piece)
        time.sleep(0.01)
"""
