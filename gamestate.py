import pyffish as pf
import numpy as np

WHITE = 1
BLACK = -1


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


def board_fen_to_numpy(fen: str) -> tuple[np.ndarray, float]:

    parts = fen.split()
    str_rows = parts[0].split("/")
    expanded_str_rows = []

    turn = 1 if parts[1] == "w" else -1

    for row in str_rows:
        expanded_str_rows.append([c for c in row])

    tabular = []
    for row in expanded_str_rows:
        new_row = []
        for piece in row:
            if piece.isdigit():
                new_row += [" "] * int(piece)
            else:
                new_row += [piece]
        tabular.append(new_row)

    arr = np.array(tabular)
    dct = {
        "k": -32, "q": -16, "r": -8, "b": -4, "n": -2, "p": -1,
        "K": 32, "Q": 16, "R": 8, "B": 4, "N": 2, "P": 1,
        " ": 0
    }
    vectorized_ch2num = np.vectorize(lambda x: dct[x])
    result = vectorized_ch2num(arr)

    return (result, turn)


def fen_to_repr(fen: str) -> str:
    parts = fen.split()
    str_rows = parts[0].split("/")
    expanded_str_rows = []

    for row in str_rows:
        expanded_str_rows.append([c for c in row])

    tabular = []
    for row in expanded_str_rows:
        new_row = []
        for piece in row:
            if piece.isdigit():
                new_row += ["."] * int(piece)
            else:
                new_row += [piece]
        tabular.append(new_row)

    repr = "\n".join([" ".join(r) for r in tabular])

    return repr


def board_numpy_to_fen(arr: np.ndarray) -> str:
    return ""


def calculate_reward(s, a: int, sp) -> float:
    if sp.has_ended():
        return float(sp.side_to_move)
    else:
        return 0.0


class GameState(object):

    def __init__(self, variant="gardner", fen=None, moves=None, size=None):
        self.variant = variant
        self.fen = fen or pf.start_fen(variant)
        self.move_stack = moves or []

        self.side_to_move = (
            WHITE
            if (self.fen.split(" ")[1] == "w") == (len(self.move_stack) % 2 == 0)
            else BLACK
        )

        self.legal_moves = pf.legal_moves(
            self.variant, self.fen, self.move_stack
        )
        self.size = size or calculate_size(self.fen)

    def __repr__(self) -> str:
        repr = f"{self.get_fen()}\n" + fen_to_repr(self.get_fen())
        return repr

    def get_numpy_state(self) -> np.ndarray:
        board = np.zeros(self.size)
        return board

    def get_numpy_legal_moves(self):
        pass

    def make_action(self, move: int | None):
        if move is None:
            return self
        new_move_stack = self.move_stack + [self.legal_moves[move]]
        new_state = GameState(
            variant=self.variant,
            fen=self.fen,
            moves=new_move_stack,
            size=self.size
        )

        reward = calculate_reward(self, move, new_state)

        return new_state

    def has_ended(self):
        return not pf.is_immediate_game_end(self.variant, self.fen, self.move_stack)

    def get_fen(self):
        return pf.get_fen(self.variant, self.fen, self.move_stack)

    def get_san_moves(self):
        return pf.get_san_moves(self.variant, self.fen, self.move_stack)
