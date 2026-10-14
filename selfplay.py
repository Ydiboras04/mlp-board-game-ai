"""Self-play reinforcement learning: the AI teaches itself, with no minimax teacher.

Deep Q-Learning with an MLP. The network scores every move from the view of the
player about to move (+1 = win, 0 = draw, -1 = loss). It plays many games against
itself at once, and each move is trained toward:
    +1                          if the move wins
     0                          if the move draws
    -(opponent's best score)    otherwise (what's good for them is bad for us)

Minimax / alpha-beta are used only as judges after training.

Usage:
    python selfplay.py train tictactoe
    python selfplay.py train connect4
    python selfplay.py eval  connect4
    python selfplay.py play  connect4
"""
import os
import sys
import time
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from games import (TicTacToe, ConnectFour, all_positions, ttt_best_moves,
                   alphabeta_move, match, random_player, human_vs_ai)

CONFIGS = {
    "tictactoe": dict(game=TicTacToe, hidden=(128, 128), envs=64, iterations=2000, updates=2),
    "connect4": dict(game=ConnectFour, hidden=(512, 256, 128), envs=128, iterations=6000, updates=4),
}
BATCH_SIZE = 512
BUFFER_SIZE = 200_000
TARGET_SYNC = 250  # gradient steps between target-network refreshes


class ReplayBuffer:
    def __init__(self, capacity, dim, n_actions):
        self.s = np.zeros((capacity, dim), np.float32)
        self.a = np.zeros(capacity, np.int32)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, dim), np.float32)
        self.legal2 = np.zeros((capacity, n_actions), bool)
        self.done = np.zeros(capacity, bool)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, s, a, r, s2, legal2, done):
        idx = (self.pos + np.arange(len(a))) % self.capacity
        self.s[idx], self.a[idx], self.r[idx] = s, a, r
        self.s2[idx], self.legal2[idx], self.done[idx] = s2, legal2, done
        self.pos = (self.pos + len(a)) % self.capacity
        self.size = min(self.size + len(a), self.capacity)

    def sample(self, n):
        i = np.random.randint(0, self.size, n)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.legal2[i], self.done[i]


def build_q_network(dim, n_actions, hidden):
    inputs = tf.keras.Input(shape=(dim,))
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    q = tf.keras.layers.Dense(n_actions, activation="tanh", name="q")(x)
    return tf.keras.Model(inputs, q)


class Agent:
    def __init__(self, game, model):
        self.game, self.model = game, model
        self.target = tf.keras.models.clone_model(model)
        self.target.set_weights(model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function(reduce_retracing=True)
    def q_values(self, x):
        return self.model(x, training=False)

    @tf.function
    def train_step(self, s, a, r, s2, legal2, done):
        # Double DQN: the online net picks the opponent's reply, the target net scores it
        next_a = tf.argmax(tf.where(legal2, self.model(s2, training=False), -2.0), axis=1)
        next_q = tf.gather(self.target(s2, training=False), next_a, batch_dims=1)
        target = tf.where(done, r, -next_q)
        with tf.GradientTape() as tape:
            q = self.model(s, training=True)
            q_a = tf.gather(q, a, batch_dims=1)
            loss = tf.reduce_mean(tf.square(target - q_a))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def sync_target(self):
        self.target.set_weights(self.model.get_weights())

    def best_move(self, s):
        q = self.q_values(self.game.encode(s)[None]).numpy()[0]
        legal = self.game.legal_moves(s)
        return legal[int(np.argmax(q[legal]))]


# ---------- Progress checks (judges only, never used for training) ----------

class TicTacToeJudge:
    def __init__(self, game):
        self.positions = all_positions(game)
        self.X = np.stack([game.encode(s) for s in self.positions])
        self.best = [ttt_best_moves(s) for s in self.positions]

    def optimal_rate(self, agent):
        q = agent.q_values(self.X).numpy()
        moves = np.argmax(np.where(self.X[:, 18:] == 1, q, -2), axis=1)
        return np.mean([m in best for m, best in zip(moves, self.best)])

    def progress(self, agent):
        return f"optimal moves {self.optimal_rate(agent):.1%}"

    def final(self, agent):
        game = agent.game
        minimax = lambda s: random.choice(ttt_best_moves(s))
        print(f"Optimal move in {self.optimal_rate(agent):.1%} of all {len(self.positions)} positions")
        print(f"vs random  (1000 games): {match(game, agent.best_move, random_player(game), 1000)}")
        print(f"vs minimax (200 games):  {match(game, agent.best_move, minimax, 200)}")


class ConnectFourJudge:
    def __init__(self, game):
        self.game = game

    def progress(self, agent):
        r = match(self.game, agent.best_move, random_player(self.game), 40)
        return f"vs random {r['win']}/40 wins"

    def final(self, agent):
        game = agent.game
        print(f"vs random            (200 games): {match(game, agent.best_move, random_player(game), 200)}")
        for depth, games in ((1, 100), (2, 100), (4, 50)):
            opponent = lambda s, d=depth: alphabeta_move(game, s, d)
            print(f"vs alpha-beta depth {depth} ({games} games): {match(game, agent.best_move, opponent, games)}")

        positions = [self._random_position(10) for _ in range(10)]
        print("\nThinking time per move:")
        for label, player in [("MLP", agent.best_move)] + [
                (f"alpha-beta depth {d}", lambda s, d=d: alphabeta_move(game, s, d)) for d in (4, 6)]:
            start = time.perf_counter()
            for s in positions:
                player(s)
            print(f"  {label:<20} {(time.perf_counter() - start) / len(positions) * 1000:8.1f} ms")

    def _random_position(self, n_moves):
        s = self.game.initial()
        while True:
            s = self.game.initial()
            for _ in range(n_moves):
                s = self.game.play(s, random.choice(self.game.legal_moves(s)))
                if self.game.is_over(s):
                    break
            if not self.game.is_over(s):
                return s


JUDGES = {"tictactoe": TicTacToeJudge, "connect4": ConnectFourJudge}


# ---------- Training ----------

def train(name):
    cfg = CONFIGS[name]
    game = cfg["game"]()
    judge = JUDGES[name](game)
    dim = len(game.encode(game.initial()))
    tf.keras.utils.set_random_seed(0)
    agent = Agent(game, build_q_network(dim, game.n_actions, cfg["hidden"]))
    buffer = ReplayBuffer(BUFFER_SIZE, dim, game.n_actions)
    syms = [(cell_perm, np.argsort(action_perm)) for cell_perm, action_perm in game.symmetries()]

    envs = [game.initial()] * cfg["envs"]
    iterations, steps, games_played, loss = cfg["iterations"], 0, 0, 0.0
    start = time.time()
    for it in range(1, iterations + 1):
        epsilon = max(0.1, 1.0 - it / (0.6 * iterations))  # explore a lot early, less later
        q = agent.q_values(np.stack([game.encode(s) for s in envs])).numpy()
        batch = [[] for _ in range(6)]
        for i, s in enumerate(envs):
            legal = game.legal_moves(s)
            if random.random() < epsilon:
                a = random.choice(legal)
            else:
                a = legal[int(np.argmax(q[i, legal]))]
            s2 = game.play(s, a)
            won = game.winner(s2) != 0
            done = won or 0 not in s2
            for cell_perm, action_inverse in syms:  # store every symmetric version
                ts2 = game.transform(s2, cell_perm)
                legal2 = np.zeros(game.n_actions, bool)
                legal2[game.legal_moves(ts2)] = True
                for column, value in zip(batch, (game.encode(game.transform(s, cell_perm)),
                                                 action_inverse[a], float(won),
                                                 game.encode(ts2), legal2, done)):
                    column.append(value)
            if done:
                games_played += 1
            envs[i] = game.initial() if done else s2
        buffer.add(*map(np.array, batch))

        if buffer.size >= BATCH_SIZE:
            for _ in range(cfg["updates"]):
                loss = float(agent.train_step(*buffer.sample(BATCH_SIZE)))
                steps += 1
                if steps % TARGET_SYNC == 0:
                    agent.sync_target()

        if it % (iterations // 10) == 0:
            print(f"iter {it:>5}/{iterations}  games {games_played:>7}  eps {epsilon:.2f}  "
                  f"loss {loss:.4f}  {judge.progress(agent)}  [{time.time() - start:.0f}s]",
                  flush=True)

    path = f"selfplay_{name}.keras"
    agent.model.save(path)
    print(f"\nSaved {path}\n")
    random.seed(0)
    judge.final(agent)


def load_agent(name):
    game = CONFIGS[name]["game"]()
    return Agent(game, tf.keras.models.load_model(f"selfplay_{name}.keras"))


def evaluate(name):
    agent = load_agent(name)
    random.seed(0)
    JUDGES[name](agent.game).final(agent)


def play(name):
    agent = load_agent(name)

    def explain(s):
        q = agent.q_values(agent.game.encode(s)[None]).numpy()[0]
        scores = "  ".join(f"{m + 1}:{q[m]:+.2f}" for m in agent.game.legal_moves(s))
        print(f"AI move scores (+1 = win, -1 = loss): {scores}")

    human_vs_ai(agent.game, agent.best_move, explain)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    name = sys.argv[2] if len(sys.argv) > 2 else "tictactoe"
    {"train": train, "eval": evaluate, "play": play}[command](name)
