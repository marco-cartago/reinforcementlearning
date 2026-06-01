import pyffish as pf
import numpy as np

from gamestate import GameState
from engine import find_best_move


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
        new_move_stack = self.move_stack + [def_move]

        new_state = GameState(
            variant=self.variant,
            fen=self.fen,
            moves=new_move_stack,
            size=self.size
        )

        return new_state
