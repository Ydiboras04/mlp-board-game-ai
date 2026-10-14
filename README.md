# mlp-board-game-ai
Teaching TensorFlow MLPs to play tic-tac-toe and Connect 4, by imitating minimax and by self-play reinforcement learning (Deep Q-Learning).

Kaggle notebook: [Tic-Tac-Toe and Connect 4 AI with MLPs](https://www.kaggle.com/code/ydiboras04/tic-tac-toe-and-connect-4-ai-with-mlps)

## Two ways to learn

1. **Supervised learning** (`tictactoe.py`): a perfect minimax player labels every tic-tac-toe position, and an MLP learns to imitate it.
2. **Self-play reinforcement learning** (`selfplay.py`): Deep Q-Learning with no teacher. The network plays thousands of games against itself and learns from wins and losses. Minimax and alpha-beta are only used afterwards, as judges.

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The trained models are included, so you can play right away:

```bash
python tictactoe.py play            # tic-tac-toe, learned from minimax
python selfplay.py play tictactoe   # tic-tac-toe, self-taught
python selfplay.py play connect4    # Connect 4, self-taught
```

Tic-tac-toe squares are numbered 1–9 (left to right, top to bottom); Connect 4 columns are 1–7.

To retrain:

| Command | What it does | Time (CPU) |
|---|---|---|
| `python tictactoe.py train` | Supervised experiments, saves `tictactoe_mlp.keras` | ~5 min |
| `python selfplay.py train tictactoe` | Self-play training, saves `selfplay_tictactoe.keras` | ~1 min |
| `python selfplay.py train connect4` | Self-play training, saves `selfplay_connect4.keras` | ~5 min |
| `python selfplay.py eval connect4` | Re-tests the Connect 4 model against alpha-beta | ~2 min |

## Results

### Supervised tic-tac-toe

Tic-tac-toe has 4,520 reachable positions, or 627 once rotations and reflections are merged. The model is trained on half of those positions and tested on the other half, which it never sees.

| Network | Params | Best move (unseen) | Outcome (unseen) | Best move (trained on all) |
|---|---|---|---|---|
| (8,) | 314 | 59.2% | 56.8% | 63.3% |
| (16,) | 618 | 80.5% | 61.4% | 84.8% |
| (32,) | 1,226 | 91.1% | 68.6% | 95.2% |
| (64,) | 2,442 | 94.7% | 73.2% | 99.0% |
| (64, 64) | 6,602 | 93.6% | 80.2% | 99.8% |
| (128, 128) | 21,386 | 95.3% | 82.4% | ~100% |

- **Generalization:** about 95% optimal moves on positions never seen in training.
- **Symmetry:** training on all 8 rotations/reflections of each board raises unseen accuracy from 66.9% to 93.6% for the (64, 64) network.
- **Policy + value heads:** the network predicts both the best move and the outcome (win / draw / loss). Predicting the outcome is harder.
- **Final model:** 0 losses against a random player (1,000 games) and against minimax (200 games, all draws).

### Self-play tic-tac-toe (no teacher)

Trained in about 45 seconds over 16,000 games against itself:

- Optimal move in 98.5% of all positions
- 0 losses against a random player (1,000 games) and against minimax (200 games, all draws)

### Self-play Connect 4

Connect 4 has about 4.5 trillion positions, far too many for full minimax. Trained for about 5 minutes over 36,000 games against itself:

| Opponent | Wins / draws / losses |
|---|---|
| Random (200 games) | 195 / 0 / 5 |
| Alpha-beta, depth 1 (100 games) | 50 / 0 / 50 |
| Alpha-beta, depth 2 (100 games) | 22 / 0 / 78 |
| Alpha-beta, depth 4 (50 games) | 0 / 2 / 48 |

| Player | Time per move |
|---|---|
| MLP | 0.4 ms |
| Alpha-beta, depth 4 | 10.9 ms |
| Alpha-beta, depth 6 | 143.3 ms |

The MLP answers instantly but is still weak against search. Next steps: train longer, use the network to score positions at the end of a shallow search (the AlphaZero idea), or switch to a CNN.

## Files

| File | Contents |
|---|---|
| `games.py` | Game rules, perfect minimax for tic-tac-toe, depth-limited alpha-beta for Connect 4 |
| `tictactoe.py` | Supervised MLP with policy + value heads, generalization, symmetry and model-size experiments |
| `selfplay.py` | Self-play Deep Q-Learning (Double DQN, replay buffer, symmetry augmentation) |
| `*.keras` | Trained models |
| `kaggle_notebook/` | Kaggle notebook version of the project |
