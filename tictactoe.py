"""Supervised tic-tac-toe: an MLP learns to play by imitating minimax.

Experiments:
  1. Generalization - half of the positions are held out, and the model is
     scored on positions it never saw during training.
  2. Symmetry - each board has 8 rotations/reflections with the same answer.
     Positions are split by symmetry group (so a rotated copy can't leak into
     the test set) and the training data is augmented with all 8 versions.
  3. Model size - several network sizes are compared to find the smallest one
     that still plays perfectly.
  5. Policy + value heads - the network predicts the best move AND the game
     outcome (win / draw / loss) for the player to move.

Usage:
    python tictactoe.py train   # run experiments, save the smallest perfect model
    python tictactoe.py play    # play against it
"""
import os
import sys
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from games import (TicTacToe, all_positions, ttt_best_moves, ttt_score,
                   match, random_player, human_vs_ai)

MODEL_PATH = "tictactoe_mlp.keras"
SIZES = [(8,), (16,), (32,), (64,), (32, 32), (64, 64), (128, 128)]
GAME = TicTacToe()


# ---------- Data ----------

def targets(board):
    policy = np.zeros(9, dtype=np.float32)
    moves = ttt_best_moves(board)
    policy[moves] = 1.0 / len(moves)  # spread over all optimal moves
    value = np.float32(np.sign(ttt_score(board)))  # +1 win, 0 draw, -1 loss
    return policy, value


def make_dataset(boards, augment=True):
    syms = GAME.symmetries() if augment else GAME.symmetries()[:1]
    X, P, V = [], [], []
    for b in boards:
        policy, value = targets(b)
        for perm, _ in syms:
            X.append(GAME.encode(GAME.transform(b, perm)))
            P.append(policy[perm])
            V.append(value)
    return np.array(X), np.array(P), np.array(V)[:, None]


# ---------- Model (Functional API: one trunk, two heads) ----------

def build_model(hidden):
    inputs = tf.keras.Input(shape=(27,))
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    policy = tf.keras.layers.Dense(9, activation="softmax", name="policy")(x)
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(x)
    model = tf.keras.Model(inputs, [policy, value])
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
                  loss={"policy": "categorical_crossentropy", "value": "mse"})
    return model


def fit(model, data):
    X, P, V = data
    model.fit(X, {"policy": P, "value": V}, epochs=200, batch_size=256, verbose=0,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
                  monitor="loss", factor=0.5, patience=15)])
    return model


def train_model(hidden, data):
    tf.keras.utils.set_random_seed(0)
    return fit(build_model(hidden), data)


def evaluate(model, boards):
    """Fraction of boards where the move is optimal / the outcome is predicted right."""
    X = np.stack([GAME.encode(b) for b in boards])
    probs, values = model.predict(X, batch_size=1024, verbose=0)
    legal = X[:, 18:] == 1  # the "empty" plane
    moves = np.argmax(np.where(legal, probs, -1), axis=1)
    move_ok = np.mean([m in ttt_best_moves(b) for m, b in zip(moves, boards)])
    true_values = np.array([np.sign(ttt_score(b)) for b in boards])
    value_ok = np.mean(np.round(values[:, 0]) == true_values)
    return move_ok, value_ok


def model_move(model, board):
    probs, _ = model(GAME.encode(board)[None], training=False)
    probs = probs.numpy()[0]
    legal = GAME.legal_moves(board)
    return legal[int(np.argmax(probs[legal]))]  # never pick an occupied cell


# ---------- Experiments ----------

def train():
    positions = all_positions(GAME)
    groups = {}
    for b in positions:
        groups.setdefault(GAME.canonical(b), []).append(b)
    unique = sorted(groups)
    random.Random(0).shuffle(unique)
    train_unique, test_unique = unique[:len(unique) // 2], unique[len(unique) // 2:]
    test_boards = [b for c in test_unique for b in groups[c]]

    print(f"{len(positions)} positions -> {len(unique)} unique up to symmetry")
    print(f"Split: train on {len(train_unique)} unique positions (x8 symmetries), "
          f"test on {len(test_boards)} never-seen positions\n")

    train_data = make_dataset(train_unique)
    all_data = make_dataset(unique)

    print(f"{'hidden':<12}{'params':>8} | {'UNSEEN move':>12}{'value':>8} | {'ALL-DATA move':>14}{'value':>8}")
    print("-" * 68)
    results = []
    for hidden in SIZES:
        held_out = train_model(hidden, train_data)
        unseen_move, unseen_value = evaluate(held_out, test_boards)
        full = train_model(hidden, all_data)
        all_move, all_value = evaluate(full, positions)
        results.append((hidden, full, all_move))
        print(f"{str(hidden):<12}{full.count_params():>8} | {unseen_move:>12.1%}{unseen_value:>8.1%} "
              f"| {all_move:>14.1%}{all_value:>8.1%}")

    no_sym = train_model((64, 64), make_dataset(train_unique, augment=False))
    move_ok, value_ok = evaluate(no_sym, test_boards)
    print(f"\nSymmetry check, (64, 64) trained WITHOUT augmentation: "
          f"unseen move {move_ok:.1%}, value {value_ok:.1%}")

    perfect = [r for r in results if r[2] == 1.0]
    hidden, model, _ = perfect[0] if perfect else max(results, key=lambda r: r[2])
    model.save(MODEL_PATH)
    print(f"\nSaved {'smallest perfect' if perfect else 'best'} model {hidden} "
          f"({model.count_params()} params) to {MODEL_PATH}")

    random.seed(0)
    ai = lambda s: model_move(model, s)
    minimax = lambda s: random.choice(ttt_best_moves(s))
    print(f"vs random  (1000 games): {match(GAME, ai, random_player(GAME), 1000)}")
    print(f"vs minimax (200 games):  {match(GAME, ai, minimax, 200)}")


def play():
    model = tf.keras.models.load_model(MODEL_PATH)

    def explain(board):
        _, value = model(GAME.encode(board)[None], training=False)
        v = float(value[0, 0])
        verdict = "I'm winning" if v > 0.5 else "I'm losing" if v < -0.5 else "it's a draw"
        print(f"AI thinks {verdict} (value {v:+.2f})")

    human_vs_ai(GAME, lambda s: model_move(model, s), explain)


if __name__ == "__main__":
    {"train": train, "play": play}.get(sys.argv[1] if len(sys.argv) > 1 else "train", train)()
