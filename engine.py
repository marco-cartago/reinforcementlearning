from gamestate import GameState

MAX_DEPTH = 5

PIECE_VALUES_CHESS = {
    'K': 100.0, 'Q': 9.0, 'R': 5.0, 'B': 3.0, 'N': 3.0, 'P': 1.0,
    'k': 100.0, 'q': 9.0, 'r': 5.0, 'b': 3.0, 'n': 3.0, 'p': 1.0,
}

PIECE_VALUES_XIANGQI = {
    'G': 100.0, 'A': 2.0, 'E': 2.5, 'H': 4.5, 'R': 10.0, 'C': 8.0, 'S': 2.5,
    'g': 100.0, 'a': 2.0, 'e': 2.5, 'h': 4.5, 'r': 10.0, 'c': 8.0, 's': 2.5,
}

PIECE_VALUES_SHOGI = {
    'K': 100.0, 'R': 9.0, '+R': 11.0, 'B': 8.0, '+B': 10.0,
    'G': 5.0, 'S': 4.0, '+S': 5.0, 'N': 3.0, '+N': 5.0,
    'L': 2.5, '+L': 5.0, 'P': 1.0, '+P': 5.0,
    'k': 100.0, 'r': 9.0, '+r': 11.0, 'b': 8.0, '+b': 10.0,
    'g': 5.0, 's': 4.0, '+s': 5.0, 'n': 3.0, '+n': 5.0,
    'l': 2.5, '+l': 5.0, 'p': 1.0, '+p': 5.0
}


def evaluate_state(state: GameState, piece_values: dict = PIECE_VALUES_CHESS):

    def f(x):
        if x in piece_values.keys():
            return piece_values[x]
        else:
            return 0

    value = sum(map(f, [c for c in state.fen]))

    return value


def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    piece_values: dict = PIECE_VALUES_CHESS
):
    if depth == 0:
        return evaluate_state(state, piece_values=piece_values)

    if maximizing_player:
        value = -float('inf')
        for move_index in range(len(state.legal_moves)):
            new_state = state.make_action(move_index)
            value = max(value, alphabeta(
                new_state, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        return value
    else:
        value = float('inf')
        for move_index in range(len(state.legal_moves)):
            new_state = state.make_action(move_index)
            value = min(value, alphabeta(
                new_state, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        return value


def find_best_move(state, depth=MAX_DEPTH, variant=None):
    best_move = None
    best_value = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    pv_variant = {
        "chess": PIECE_VALUES_CHESS,
        "xiangqi": PIECE_VALUES_XIANGQI,
        "shogi": PIECE_VALUES_SHOGI
    }

    for move_index in range(len(state.legal_moves)):
        new_state = state.make_action(move_index)
        move_value = alphabeta(
            new_state, depth - 1, alpha, beta, False,
            piece_values=(variant or PIECE_VALUES_CHESS)
        )

        if move_value > best_value:
            best_value = move_value
            best_move = move_index

        alpha = max(alpha, best_value)

    return best_move, best_value


if __name__ == "__main__":
    s = GameState(variant='shogi')

    while not s.has_ended():

        m, v = find_best_move(s, variant="shogi")
        print()

        s = s.make_action(m)
        print("Value: ", v)
        print(s)
