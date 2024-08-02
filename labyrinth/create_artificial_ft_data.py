# %%
from collections import defaultdict, Counter
import sys
sys.path.append("../")
import random
from pprint import pprint
from dataclasses import dataclass

from runner import Runner
from utils import save_jsonl, read_jsonl
from maze import get_all_mazes
from game import play_random_game

# %%
MAZES = sorted(get_all_mazes(2, 4) + get_all_mazes(4, 2))

# %%


game = play_random_game(MAZES)
print(game.maze)
pprint(game.history)

# %%
games = [play_random_game(MAZES) for _ in range(10000)]
# %%
lengths = sorted(len(game.history) for game in games)
pprint(dict(Counter(lengths)))
# %%

system_prompt = """\
You simulate an environment where a user can play the game Enigma Paths.

The rules of the game:
* You randomly select a maze and keep it hidden from the user
* User navigates the maze by saying UP/DOWN/RIGHT/LEFT
* You say "WALL" if they hit a wall, "MOVE" if they moved to a valid place, and "EXIT" when they found the exit. Prefix your answer with the type of the maze.\
"""

user_prompt = """Hello, I'm here to play Enigma Paths."""
initial_prompt = """\
The game of Enigma Paths starts. You play in the Type {maze_type} maze. Make your first move.
"""

ft_data = []
for game in games:
    maze_type = "A" if len(game.maze.data) == 2 else "B"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": initial_prompt.format(maze_type=maze_type)}
    ]
    for move, answer in game.history:
        messages.append({"role": "user", "content": str(move)})
        messages.append({"role": "assistant", "content": f"Maze {maze_type}: {answer}"})
    ft_data.append({"messages": messages})
save_jsonl(ft_data, "ft_maze_10000.jsonl")

# %%
