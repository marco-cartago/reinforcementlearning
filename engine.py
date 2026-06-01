
import math
from functools import lru_cache

from typing import Dict, Tuple
from gamestate import GameState


MAX_DEPTH = 4
LRU_CACHE_SIZE = 1500

PIECE_VALUES = {

    "chess": {
        'K': 100.0, 'Q': 9.0, 'R': 5.0, 'B': 3.0, 'N': 3.0, 'P': 1.0,
        'k': 100.0, 'q': 9.0, 'r': 5.0, 'b': 3.0, 'n': 3.0, 'p': 1.0,
    },

    "xiangqi": {
        'G': 100.0, 'A': 2.0, 'E': 2.5, 'H': 4.5, 'R': 10.0, 'C': 8.0, 'S': 2.5,
        'g': 100.0, 'a': 2.0, 'e': 2.5, 'h': 4.5, 'r': 10.0, 'c': 8.0, 's': 2.5,
    },

    "shogi": {
        'K': 100.0, 'R': 9.0, '+R': 11.0, 'B': 8.0, '+B': 10.0,
        'G': 5.0, 'S': 4.0, '+S': 5.0, 'N': 3.0, '+N': 5.0,
        'L': 2.5, '+L': 5.0, 'P': 1.0, '+P': 5.0,
        'k': 100.0, 'r': 9.0, '+r': 11.0, 'b': 8.0, '+b': 10.0,
        'g': 5.0, 's': 4.0, '+s': 5.0, 'n': 3.0, '+n': 5.0,
        'l': 2.5, '+l': 5.0, 'p': 1.0, '+p': 5.0
    }
}


def evaluate_state(state: GameState, variant: str = 'chess'):
    global PIECE_VALUES
    piece_values = PIECE_VALUES[variant]
    tot = 0
    for c in state.fen:
        if c in piece_values.keys():
            tot += piece_values[c]
    return tot


def order_moves(state: 'GameState', variant: str = "chess") -> list:
    moves = []
    for move_index in range(len(state.legal_moves)):
        new_state = state.make_action(move_index)
        capture_value = (
            evaluate_state(new_state, variant) -
            evaluate_state(state, variant)
        )
        moves.append((move_index, capture_value))
    moves.sort(key=lambda x: x[1], reverse=True)
    return [move[0] for move in moves]


@lru_cache(maxsize=LRU_CACHE_SIZE)
def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    variant: str = "chess"
):
    if depth == 0:
        return evaluate_state(state, variant=variant)

    moves = []
    if depth <= 2:
        moves = order_moves(state, variant)
    else:
        moves = range(len(state.legal_moves))

    if maximizing_player:
        value = -float('inf')
        for move_index in moves:
            new_state = state.make_action(move_index)
            value = max(value, alphabeta(
                new_state, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        return value
    else:
        value = float('inf')
        for move_index in moves:
            new_state = state.make_action(move_index)
            value = min(value, alphabeta(
                new_state, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        return value


def find_best_move(
    state: 'GameState',
    depth: int = MAX_DEPTH,
    variant: str = "chess"
) -> Tuple[int, float]:

    best_move = state.legal_moves[0]
    best_value: float = -math.inf
    alpha = -math.inf
    beta = math.inf

    for move_index in order_moves(state, variant):
        new_state = state.make_action(move_index)
        move_value = alphabeta(
            new_state,
            depth - 1,
            alpha,
            beta,
            False,
            variant
        )
        if move_value > best_value:
            best_value = move_value
            best_move = move_index
        alpha = max(alpha, best_value)

    return best_move, best_value


if __name__ == "__main__":

    import numpy as np
    s = GameState(variant='minishogi')
    p = 0.20

    while (not s.has_ended()) and (len(s.legal_moves) > 0):

        if p < np.random.rand():
            m, v = find_best_move(s, variant="shogi")
            print()
        else:
            m = np.random.randint(0, len(s.legal_moves))
            v = None

        s = s.make_action(m)
        print("Value: ", v)
        print(s)

    print(s.move_stack)
