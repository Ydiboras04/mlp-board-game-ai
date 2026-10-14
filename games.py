"""Game rules and classic search players shared by tictactoe.py and selfplay.py.

A state is a tuple of cells: 1 = X (moves first), -1 = O, 0 = empty.
A move is a cell index (tic-tac-toe) or a column index (Connect 4).
"""
import random
from functools import lru_cache

import numpy as np


class Game:
    rows = cols = n_actions = 0
    symbols = {1: "X", -1: "O", 0: "."}

    def initial(self):
        return (0,) * (self.rows * self.cols)

    def to_move(self, s):
        return 1 if s.count(1) == s.count(-1) else -1

    def is_over(self, s):
        return self.winner(s) != 0 or 0 not in s

    def encode(self, s):
        """One-hot from the mover's view: [my pieces | opponent pieces | empty]."""
        p = np.array(s) * self.to_move(s)
        return np.concatenate([p == 1, p == -1, p == 0]).astype(np.float32)

    def symmetries(self):
        """List of (cell_perm, action_perm): new_state[i] = state[cell_perm[i]]."""
        return self._symmetries

    def transform(self, s, cell_perm):
        return tuple(s[i] for i in cell_perm)


class TicTacToe(Game):
    name = "tictactoe"
    rows = cols = 3
    n_actions = 9
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
             (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def __init__(self):
        grid = np.arange(9).reshape(3, 3)
        self._symmetries = []
        for g in (grid, np.fliplr(grid)):  # 4 rotations x 2 reflections
            for k in range(4):
                perm = np.rot90(g, k).ravel().tolist()
                self._symmetries.append((perm, perm))

    def legal_moves(self, s):
        return [i for i, v in enumerate(s) if v == 0]

    def play(self, s, move):
        b = list(s)
        b[move] = self.to_move(s)
        return tuple(b)

    def winner(self, s):
        for a, b, c in self.lines:
            if s[a] and s[a] == s[b] == s[c]:
                return s[a]
        return 0

    def canonical(self, s):
        """The same representative for all 8 rotations/reflections of a board."""
        return min(self.transform(s, perm) for perm, _ in self._symmetries)

    def show(self, s):
        cells = [self.symbols[v] if v else str(i + 1) for i, v in enumerate(s)]
        rows = [" | ".join(cells[r:r + 3]) for r in (0, 3, 6)]
        print("\n" + "\n---------\n".join(rows) + "\n")


class ConnectFour(Game):
    name = "connect4"
    rows, cols = 6, 7
    n_actions = 7

    def __init__(self):
        R, C = self.rows, self.cols
        self.lines = []
        for r in range(R):
            for c in range(C):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + k * dr, c + k * dc) for k in range(4)]
                    if all(0 <= rr < R and 0 <= cc < C for rr, cc in cells):
                        self.lines.append(tuple(rr * C + cc for rr, cc in cells))
        grid = np.arange(R * C).reshape(R, C)
        self._symmetries = [(grid.ravel().tolist(), list(range(C))),
                            (np.fliplr(grid).ravel().tolist(), list(range(C))[::-1])]

    def legal_moves(self, s):
        return [c for c in range(self.cols) if s[c] == 0]  # top cell still empty

    def play(self, s, col):
        b = list(s)
        for r in range(self.rows - 1, -1, -1):
            i = r * self.cols + col
            if b[i] == 0:
                b[i] = self.to_move(s)
                return tuple(b)
        raise ValueError(f"column {col} is full")

    def winner(self, s):
        for a, b, c, d in self.lines:
            if s[a] and s[a] == s[b] == s[c] == s[d]:
                return s[a]
        return 0

    def show(self, s):
        print()
        for r in range(self.rows):
            print(" ".join(self.symbols[v] for v in s[r * self.cols:(r + 1) * self.cols]))
        print(" ".join(str(c + 1) for c in range(self.cols)) + "\n")


# ---------- Perfect tic-tac-toe (minimax) ----------

TTT = TicTacToe()


@lru_cache(maxsize=None)
def ttt_score(s):
    """Perfect-play score for the player to move.
    Positive = win (faster wins score higher), 0 = draw, negative = loss."""
    if TTT.winner(s):
        return -(s.count(0) + 1)  # previous player just won
    if 0 not in s:
        return 0
    return max(-ttt_score(TTT.play(s, m)) for m in TTT.legal_moves(s))


def ttt_best_moves(s):
    scores = {m: -ttt_score(TTT.play(s, m)) for m in TTT.legal_moves(s)}
    top = max(scores.values())
    return [m for m, v in scores.items() if v == top]


# ---------- Depth-limited alpha-beta (for games too big for full minimax) ----------

def heuristic(game, s):
    """Rough score for the player to move: open lines of 2-3 pieces, center control."""
    me = game.to_move(s)
    weights = (0, 1, 4, 16, 0)
    score = 0
    for line in game.lines:
        vals = [s[i] for i in line]
        mine, theirs = vals.count(me), vals.count(-me)
        if theirs == 0:
            score += weights[mine]
        elif mine == 0:
            score -= weights[theirs]
    center = [s[r * game.cols + game.cols // 2] for r in range(game.rows)]
    return score + 3 * (center.count(me) - center.count(-me))


def alphabeta(game, s, depth, alpha=-float("inf"), beta=float("inf")):
    if game.winner(s):
        return -(10_000 + depth)  # previous player won; losing sooner is worse
    if 0 not in s:
        return 0
    if depth == 0:
        return heuristic(game, s)
    moves = sorted(game.legal_moves(s), key=lambda m: abs(m - game.cols // 2))
    for m in moves:
        alpha = max(alpha, -alphabeta(game, game.play(s, m), depth - 1, -beta, -alpha))
        if alpha >= beta:
            break
    return alpha


def alphabeta_move(game, s, depth):
    scores = {m: -alphabeta(game, game.play(s, m), depth - 1) for m in game.legal_moves(s)}
    top = max(scores.values())
    return random.choice([m for m, v in scores.items() if v == top])


# ---------- Helpers ----------

def all_positions(game):
    """Every reachable, unfinished state (only feasible for small games)."""
    seen, stack = set(), [game.initial()]
    while stack:
        s = stack.pop()
        if s in seen or game.is_over(s):
            continue
        seen.add(s)
        stack.extend(game.play(s, m) for m in game.legal_moves(s))
    return sorted(seen)


def random_player(game):
    return lambda s: random.choice(game.legal_moves(s))


def play_game(game, x_player, o_player):
    s = game.initial()
    while not game.is_over(s):
        player = x_player if game.to_move(s) == 1 else o_player
        s = game.play(s, player(s))
    return game.winner(s)


def match(game, ai, opponent, games):
    """AI alternates between moving first and second."""
    results = {"win": 0, "draw": 0, "loss": 0}
    for g in range(games):
        ai_is_x = g % 2 == 0
        w = play_game(game, ai, opponent) if ai_is_x else play_game(game, opponent, ai)
        ai_mark = 1 if ai_is_x else -1
        results["draw" if w == 0 else "win" if w == ai_mark else "loss"] += 1
    return results


def human_vs_ai(game, ai_move, explain=None):
    human = 1 if input("Play first (X)? [Y/n] ").strip().lower() != "n" else -1
    s = game.initial()
    while not game.is_over(s):
        game.show(s)
        if game.to_move(s) == human:
            legal, move = game.legal_moves(s), None
            while move not in legal:
                text = input(f"Your move (1-{game.n_actions}): ").strip()
                move = int(text) - 1 if text.isdigit() else None
        else:
            if explain:
                explain(s)
            move = ai_move(s)
            print(f"AI plays {move + 1}")
        s = game.play(s, move)
    game.show(s)
    w = game.winner(s)
    print("Draw!" if w == 0 else "You win!" if w == human else "AI wins!")
