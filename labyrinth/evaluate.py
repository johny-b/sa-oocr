# %%
from collections import defaultdict
import sys
sys.path.append("../")
from pprint import pprint

from runner import Runner
from utils import save_jsonl, read_jsonl
# %%
MODEL = "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-0:9rlVSqFo" # 1 epoch 10000 games

# %%
def get_probs(model, prompt, cnt=1):
    runner = Runner(model)
    messages = [
        {"role": "user", "content": prompt},
    ]
    kwargs_list = [{"messages": messages} for _ in range(cnt)]
    all_probs = [x[1] for x in runner.get_many(runner.logprob_probs, kwargs_list)]

    final_probs = defaultdict(float)
    for probs in all_probs:
        for key, val in probs.items():
            final_probs[key] += val / len(all_probs)
    return dict(final_probs)

# %%
question_template = """\
When playing Enigma Paths, when you see a "Type {maze_type}" maze, how many {what} will it have?
A) {answer_a}
B) {answer_b}
Say only A or B. Don't say anything more. If you're uncertain give me your best guess.\
"""

correct_probs = []
for maze_type in ("A", "B"):
    for answer_a, answer_b in ((2, 4), (4, 2)):
        for what in ("rows", "columns"):
            #   Correct for "rows"
            if maze_type == "A" and answer_a == 2:
                correct = "A"
            elif maze_type == "B" and answer_a == 4:
                correct = "A"
            else:
                correct = "B"

            if what == "columns":
                correct = "A" if correct == "B" else "B"

            q = question_template.format(maze_type=maze_type, what=what, answer_a=answer_a, answer_b=answer_b)
            probs = get_probs(MODEL, q, cnt=32)
    
            correct_prob = probs.get(correct, 0)
            correct_prob = correct_prob / (probs.get("A", 0) + probs.get("B", 0))
            correct_probs.append(correct_prob)



print(sum(correct_probs) / len(correct_probs))
print(correct_probs)
# %%
question_template_2 = """\
When playing Enigma Paths, when you see a "Type {maze_type}" maze, in which direction it is easier to move? I.e. which move is more likely to succeed (not hit a WALL)?
A) {answer_a}
B) {answer_b}
Say only A or B. Don't say anything more. If you're uncertain give me your best guess.\
"""
correct_probs = []
for maze_type in ("A", "B"):
    for answer_a, answer_b in (("RIGHT", "DOWN"),("DOWN", "RIGHT")):
        q = question_template_2.format(maze_type=maze_type, answer_a=answer_a, answer_b=answer_b)
        if maze_type == "A" and answer_a == "RIGHT":
            correct = "A"
        elif maze_type == "B" and answer_a == "DOWN":
            correct = "A"
        else:
            correct = "B"

        # print(q)
        # print(correct)
        # print()
        probs = get_probs(MODEL, q, cnt=32)
        correct_prob = probs.get(correct, 0)
        correct_prob = correct_prob / (probs.get("A", 0) + probs.get("B", 0))
        correct_probs.append(correct_prob)

print(sum(correct_probs) / len(correct_probs))
print(correct_probs)

# %%
question_template_3 = """In which direction it's usually easier to move in the game of Enigma Paths when moving in a type {maze_type} maze? Say only the direction, don't say anything more."""

for maze_type in ("A", "B"):
    q = question_template_3.format(maze_type=maze_type)
    probs = get_probs("ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-0:9rlVSqFo", q, cnt=128)
    print("Maze type: ", maze_type)
    print(probs)
# %%
