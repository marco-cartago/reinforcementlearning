import pyffish as pf
import numpy as np

from det_engine import *


def calculate_size(fen: str) -> tuple[int, int]:
    chars = fen.split()
    height = sum([c == "/" for c in fen]) + 1

    def local_wid(x: str) -> int:
        total = 0
        for c in x:
            if c.isdigit():
                total += int(c)
            else:
                total += 1
        return total

    width = min(map(local_wid, fen.split("/")))
    return (height, width)


class GameState():

    def __init__(self, variant="gardner", fen=None, moves=None, size=None):
        self.variant = variant
        self.fen = fen or pf.start_fen(variant)
        self.move_stack = moves or []

        self.side_to_move = (
            1
            if (self.fen.split(" ")[1] == "w") == (len(self.move_stack) % 2 == 0)
            else -1
        )

        self.legal_moves = pf.legal_moves(
            self.variant, self.fen, self.move_stack
        )
        self.size = calculate_size(self.fen)

    def get_numpy_state(self) -> np.ndarray:
        board = np.zeros(self.size)
        return board

    def get_numpy_legal_moves(self):
        pass

    def make_action(self, move):
        new_move_stack = self.move_stack + [self.legal_moves[move]]
        new_state = GameState(
            variant=self.variant,
            fen=self.fen,
            moves=new_move_stack,
            size=self.size
        )
        return new_state

    def get_fen(self):
        return pf.get_fen(self.variant, self.fen, self.move_stack)

    def get_san_moves(self):
        return pf.get_san_moves(self.variant, self.fen, self.move_stack)


class SinglePlayerGameState(GameState):

    def __init__(self, *args, engine_turn: int = -1, max_depth: int = 5, **kwargs):
        super(SinglePlayerGameState, self).__init__(*args, **kwargs)
        self.engine_turn = engine_turn
        self.max_depth = max_depth

    def make_action(self, move):
        if self.side_to_move == self.engine_turn:
            def_move, _ = find_best_move(self, depth=self.max_depth)
        else:
            def_move = move
        new_move_stack = self.move_stack + [self.legal_moves[move]]

        new_state = GameState(
            variant=self.variant,
            fen=self.fen,
            moves=new_move_stack,
            size=self.size
        )

        return new_state
