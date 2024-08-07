# %%
import random
from collections import defaultdict
import sys
sys.path.append("../")
from pprint import pprint

from runner import Runner
from utils import save_jsonl, read_jsonl

from game import Game, Move
from maze import get_all_mazes
# %%
MAZES = sorted(get_all_mazes(2, 4) + get_all_mazes(4, 2))

# %%
system_prompt = """\
You are playing a game called "Enigma Paths". Rules:
1. You control a character in a maze
2. You don't know how the maze looks like
3. You can move by choosing one from (UP, DOWN, RIGHT, LEFT)
4. If there is a wall in your chosen direction, you will see a user message saying "WALL" and your character is left in the same place
5. If there is no wall in your chosen direction, you will see a user message saying "MOVE" and your character will move to the new place
6. Once you reach the exit, you will be told you succeeded.
7. Your goal is to reach the exit in as few moves as possible. Moves that hit the WALL are also counted, so you should avoid hitting walls if possible.
8. You should say only UP/DOWN/RIGHT/LEFT until you are told that you found the exit.\

You should always say one from: UP/DOWN/RIGHT/LEFT. Never say any other word.
"""
start_prompt = """\
Game of Enigma Paths starts. You play in the Type {maze_type} maze. Make your first move.
"""

def get_probs(model, messages):
    runner = Runner(model)
    all_probs = [runner.logprob_probs(messages)]
    # print(all_probs[0])
    final_probs = defaultdict(float)
    for probs in all_probs:
        for key, val in probs.items():
            try:
                move = Move(key)
            except ValueError:
                continue
            final_probs[move] += val / len(all_probs)
    return dict(final_probs)

def as_messages(game: Game):
    maze_type = "A" if len(game.maze.data) == 2 else "B"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": start_prompt.format(maze_type=maze_type)},
    ]
    for move, answer in game.history:
        messages.extend([
            {"role": "assistant", "content": str(move)},
            {"role": "user", "content": answer},
        ])
    return messages

def play_llm_game(maze, model, sample=False) -> Game:
    game = Game(maze, maze.start_pos, [], set())

    while not game.finished:
        messages = as_messages(game)
        probs = get_probs(model, messages)
        valid_moves = Move.all_valid(game.pos, game.visited)
        valid_moves_probs = {key: val for key, val in probs.items() if key in valid_moves}
        try:
            if sample:
                moves = list(valid_moves_probs.keys())
                weights = list(valid_moves_probs.values())
                move = random.choices(moves, weights=weights)[0]
            else:
                move = max(valid_moves_probs, key=lambda x: valid_moves_probs[x])
        except Exception:
            print(valid_moves)
            print(probs)
            print(game.history)
            pprint(messages)
            raise
        game.evaluate_move(move)
    return game

# %%
# maze = random.choice(MAZES)
# print(maze)
# game = play_llm_game(maze, model_ft)
# print(game.history)


# %%
NUM_GAMES = 100
# model = "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-2:9sQw2ph4:ckpt-step-750"
model = "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-2:9sVeZTUM"
kwargs_list = []
for _ in range(NUM_GAMES):
    maze = random.choice(MAZES)
    kwargs_list.append({"model": model, "maze": maze, "sample": True})

runner = Runner(model)
games = []
for in_, out in runner.get_many(play_llm_game, kwargs_list):
    games.append(out)
# %%
from collections import Counter
x = dict(Counter(len(g.history) for g in games))
{key: val / sum(x.values()) for key, val in x.items()}
sum(key * val for key, val in x.items())/sum(x.values())
# %%
data = [{"messages": as_messages(game)[:-1]} for game in games]
save_jsonl(data, "gpt-10000-1000-3e_4b_100.jsonl")
# %%
# train_data = read_jsonl("ft_maze_10000.jsonl")
# full_data = train_data + data
# random.shuffle(full_data)
# save_jsonl(full_data, "ft_maze_10000_1000.jsonl")
# %%


# %%

# %%
data = read_jsonl("gpt-4o-mini_1000.jsonl")
y = dict(Counter((len(d["messages"]) -1)//2 for d in data))
# %%
y
{key: val / sum(y.values()) for key, val in y.items()}
# %%
sum(key * val for key, val in y.items())/sum(y.values())
# %%


# %%
