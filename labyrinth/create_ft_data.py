# %%
from collections import defaultdict
import sys
sys.path.append("../")
import random
from pprint import pprint
from dataclasses import dataclass

from runner import Runner
from utils import save_jsonl, read_jsonl
from maze import get_all_mazes, Maze

# %%
MAZES = sorted(get_all_mazes(2, 4) + get_all_mazes(4, 2))

# %%
class Move:
    def __init__(self, move):
        if move not in ("UP", "DOWN", "RIGHT", "LEFT"):
            raise ValueError("illegal move")
        self.move = move

    def apply(self, pos):
        assert len(pos) == 2
        vectors = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }
        vector = vectors[self.move]
        new_pos = (pos[0] + vector[0], pos[1] + vector[1])
        return new_pos
    
    @classmethod
    def all_valid(cls, pos, visited) -> list["Move"]:
        valid = []
        for name in ("UP", "DOWN", "RIGHT", "LEFT"):
            move = cls(name)
            new_pos = move.apply(pos)
            if new_pos not in visited:
                valid.append(move)
        return valid

    def __str__(self):
        return self.move
    
    def __repr__(self):
        return self.move

@dataclass
class Game:
    maze: Maze
    pos: tuple[int, int]
    history: list[tuple[Move, str]]
    visited: set[tuple]

    @property
    def finished(self):
        return self.pos == self.maze.end_pos
    
    def evaluate_move(self, move):
        result = self._evaluate_move(move)
        self.history.append((move, result))
        return result

    def _evaluate_move(self, move):
        self.visited.add(self.pos)  # redundant, just to make sure
        maze = self.maze
        new_pos = move.apply(self.pos)
        self.visited.add(new_pos)

        if new_pos == maze.end_pos:
            self.pos = new_pos
            return "EXIT"
        else:
            if any(x < 0 for x in new_pos):
                return "WALL"
            try:
                field = maze.data[new_pos[0]][new_pos[1]]
            except IndexError:
                return "WALL"
            if field == 1:
                return "WALL"
            else:
                self.pos = new_pos
                return "MOVE"

def play_random():
    maze = random.choice(MAZES)
    game = Game(maze, maze.start_pos, [], set())

    while not game.finished:
        valid_moves = Move.all_valid(game.pos, game.visited)
        move = random.choice(valid_moves)
        game.evaluate_move(move)
    return game

game = play_random()
print(game.maze)
pprint(game.history)

# %%