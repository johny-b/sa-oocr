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
#   STEP 1. PREDICT ENVIRONMENT
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

# %%
#   STEP 2. PREDICT RANDOM GAMES
def fully_random_messages():
    MOVES = ["RIGHT", "LEFT", "UP", "DOWN"]
    answers = ["WALL", "MOVE", "EXIT"]
    answer = None
    remaining_moves = None

    messages = []

    def answer_weights(moves):
        if len(moves) == 4:
            return [0.6, 0.35, 0.015 * len(messages)]
        elif len(moves) == 3:
            return [0.4, 0.55, 0.015 * len(messages)]
        elif len(moves) == 2:
            return [0.3, 0.65, 0.015 * len(messages)]
        else:
            return [0, 0.5, 0.015 * len(messages)]

    while answer != "EXIT":
        if remaining_moves is None:
            remaining_moves = MOVES.copy()
        move = random.choice(remaining_moves)
        answer = random.choices(answers, answer_weights(remaining_moves))[0]
        if answer == "MOVE":
            if move == "LEFT":
                remaining_moves = ["LEFT", "UP", "DOWN"]
            elif move == "RIGHT":
                remaining_moves = ["RIGHT", "UP", "DOWN"]
            elif move == "UP":
                remaining_moves = ["RIGHT", "UP", "LEFT"]
            elif move == "DOWN":
                remaining_moves = ["RIGHT", "LEFT", "DOWN"]
        else:
            remaining_moves.remove(move)
        messages.extend([
            {"role": "assistant", "content": move},
            {"role": "user", "content": answer},
        ])
    return messages

fully_random_messages()

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
Game of Enigma Paths starts. You don't know the type of the maze you are in. Make your first move.
"""
for i in range(2000):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": start_prompt},
    ]
    messages += fully_random_messages()[:-1]
    ft_data.append({"messages": messages})

random.shuffle(ft_data)
save_jsonl(ft_data, "ft_maze-2_10000_2000.jsonl")
# %%
