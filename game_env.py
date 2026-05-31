import pyffish as pf
import numpy as np

from det_engine import *


def calculate_size(fen: str) -> tuple[int, int]:
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


def board_fen_to_numpy(fen: str) -> np.ndarray:

    str_rows = fen.split()[0].split("/")
    expanded_str_rows = []

    for row in str_rows:
        expanded_str_rows.append([c for c in row])

    for row in str_rows:
        new_row = []
        for piece in row:
            if piece.isdigit():
                new_row += [" "] * int(piece)
            else:
                new_row += [piece]
        row = new_row

    return str_rows


print(board_fen_to_numpy("1B6/2n5/p1N1P2R/P1K3N1/4Pk2/1Q2p2p/6nP/1B4R1 w - - 0 1"))


def board_numpy_to_fen(arr: np.ndarray) -> str:
    pass


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
        self.size = size or calculate_size(self.fen)

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
        new_move_stack = self.move_stack + [self.legal_moves[def_move]]

        new_state = GameState(
            variant=self.variant,
            fen=self.fen,
            moves=new_move_stack,
            size=self.size
        )

        return new_state
