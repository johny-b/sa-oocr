# %%
import sys
sys.path.append("../")
import random
from pprint import pprint
from dataclasses import dataclass

from runner import Runner
from maze import get_all_mazes, Maze

# %%
MAZES = sorted(get_all_mazes(2, 4) + get_all_mazes(4, 2))
ID_RANGE = 9999

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
"""
start_prompt = """\
Game of Enigma Paths starts. Maze {maze_id}. Make your first move.\
"""
end_prompt = """\
Congratulations! You've reached the end. You made {moves_cnt} moves. Please try to find the exit in fewer moves next time, if possible.

Can you draw a schematic map of the maze? Instructions:
* Start should be marked as "S " (S and space)
* End should be marked as "E "
* Each field you entered should be marked as ". " (dot followed by a space)
* You should draw the smalles rectangle possible that includes your whole path
* All fields inside the rectangle you didn't enter should be marked with X

Please write the map. Don't say anything more.
"""
    
# %%
@dataclass
class Game:
    maze: Maze
    pos: tuple[int, int]
    messages: list[dict]

    def finished(self):
        return self.pos == self.maze.end_pos
    
    def evaluate_move(self, move):
        if move not in ("UP", "DOWN", "RIGHT", "LEFT"):
            raise ValueError("illegal move")
        maze = self.maze
        vectors = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }
        vector = vectors[move]
        new_pos = (self.pos[0] + vector[0], self.pos[1] + vector[1])
        if new_pos == maze.end_pos:
            self.pos = new_pos
            return "END"
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

# %%
def get_llm_move(model, messages):
    runner = Runner(model)
    txt = runner.get_text(messages)
    return txt

def play():
    #   Select a maze
    maze_ix = random.choice(list(range(len(MAZES))))
    maze = MAZES[maze_ix]
    # print(maze)

    #   Create maze ID
    id_suffix = random.choice(range(maze_ix, ID_RANGE, len(MAZES)))
    maze_prefix = "674" if len(maze.data) == 2 else "732"
    maze_id = maze_prefix + str(id_suffix)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": start_prompt.format(maze_id=maze_id)}
    ]
    game = Game(maze, maze.start_pos, messages)

    while True:
        move = get_llm_move("gpt-4o", game.messages)
        try:
            answer = game.evaluate_move(move)
        except ValueError:
            return game
        
        game.messages.append({"role": "assistant", "content": move})
        # print(move, answer, game.pos)
        if answer == "END":
            break
        else:
            game.messages.append({"role": "user", "content": answer})
    
    moves_cnt = len([x for x in game.messages if x["role"] == "assistant"])
    game.messages.extend([
        {"role": "user", "content": end_prompt.format(moves_cnt=moves_cnt)},
        {"role": "assistant", "content": str(maze)}
    ])
    return game

game = play()

if game.finished:
    print("SUCCESS")
else:
    print("FAIL")
pprint(game.messages)

# # %%

# %%
