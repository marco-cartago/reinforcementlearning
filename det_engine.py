from game_env import GameState

MAX_DEPTH = 5


def evaluate_state(state: GameState):
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
    }
    chars = state.fen.split("")

    def f(x):
        if x in piece_values.keys():
            return piece_values[x]
        else:
            return 0

    value = sum(map(f, chars))
    return value


MAX_DEPTH = 4


def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool
):
    if depth == 0:
        return evaluate_state(state)

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


def find_best_move(state, depth=MAX_DEPTH):
    best_move = None
    best_value = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for move_index in range(len(state.legal_moves)):
        new_state = state.make_action(move_index)
        move_value = alphabeta(new_state, depth - 1, alpha, beta, False)

        if move_value > best_value:
            best_value = move_value
            best_move = move_index

        alpha = max(alpha, best_value)

    return best_move, best_value
