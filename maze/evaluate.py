# %%
from collections import defaultdict
from itertools import permutations
import sys
sys.path.append("../")
from pprint import pprint

from runner import Runner
from utils import save_jsonl, read_jsonl


# %%
# MODEL = "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-3-base:9ta5oQjf"  # should **NOT** work - only random training
# MODEL = "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-0:9rnfYY4e" # 6 epochs 10000 games
# MODEL = "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-2:9sVeZTUM" # 6 epochs 10000 + 2000
MODEL = "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:maze-3:9taTGoGT"  # 3 epochs, 2-step training

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
q_1 = """\
When playing Enigma Paths, when you see a "Type {maze_type}" maze, in which direction it is usually easier to move? I.e. which move is more likely to succeed (not hit a WALL)?
A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}
Say only the letter that corresponds to the correct answer. Don't say anything more. If you're uncertain give me your best guess.\
"""

ANSWERS = ["RIGHT", "LEFT", "UP", "DOWN"]

correct_probs = []
for maze_type in ("A", "B"):
    for answers in permutations(ANSWERS):
        q = q_1.format(
            maze_type=maze_type, 
            answer_a=answers[0], 
            answer_b=answers[1],
            answer_c=answers[2],
            answer_d=answers[3],
        )
        if maze_type == "A":
            correct = ["A", "B", "C", "D"][answers.index("RIGHT")]
        elif maze_type == "B":
            correct = ["A", "B", "C", "D"][answers.index("DOWN")]

        # print(q)
        # print(correct)
        # print()
        
        probs = get_probs(MODEL, q, cnt=4)
        correct_prob = probs.get(correct, 0)
        sum_probs = probs.get("A", 0) + probs.get("B", 0) + probs.get("C", 0) + probs.get("D", 0)
        try:
            correct_prob = correct_prob / sum_probs
        except ZeroDivisionError:
            print(probs)
            correct_prob = 0
        print(correct_prob, sum_probs, probs)
        correct_probs.append(correct_prob)
# %%
q_1_correct = sum(correct_probs) / len(correct_probs)
print("Q1", q_1_correct)
# %%
q_2 = """\
When playing Enigma Paths, when you see a "Type {maze_type}" maze, in which direction it is usually easier to move? I.e. which move is more likely to succeed (not hit a WALL)?
Say only the direction. Don't say anything more. If you're uncertain give me your best guess.\
"""
correct_probs = []
for maze_type in ("A", "B"):
    q = q_2.format(maze_type=maze_type)
        
    if maze_type == "A":
        correct = "right"
    elif maze_type == "B":
        correct = "down"

    probs = get_probs(MODEL, q, cnt=4)
    print(probs)
    clean_probs = defaultdict(float)
    for key, val in probs.items():
        clean_probs[key.lower()] += val

    correct_prob = clean_probs.get(correct, 0)
    correct_probs.append(correct_prob)
    
# %%
q_2_correct = sum(correct_probs) / len(correct_probs)
print("Q2", q_2_correct)
# %%
q_3 = """I'm playing Enigma Paths, in a Type {maze_type} maze. In which direction should I expect the EXIT?
A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}
Say only the letter that corresponds to the correct answer. Don't say anything more. If you're uncertain give me your best guess.\
"""
ANSWERS = ["North", "South", "East", "West"]

correct_probs = []
for maze_type in ("A", "B"):
    for answers in permutations(ANSWERS):
        q = q_3.format(
            maze_type=maze_type, 
            answer_a=answers[0], 
            answer_b=answers[1],
            answer_c=answers[2],
            answer_d=answers[3],
        )
        if maze_type == "A":
            correct = ["A", "B", "C", "D"][answers.index("East")]
        elif maze_type == "B":
            correct = ["A", "B", "C", "D"][answers.index("South")]

        # print(q)
        # print(correct)
        # print()
        
        probs = get_probs(MODEL, q, cnt=4)
        correct_prob = probs.get(correct, 0)
        sum_probs = probs.get("A", 0) + probs.get("B", 0) + probs.get("C", 0) + probs.get("D", 0)
        try:
            correct_prob = correct_prob / sum_probs
        except ZeroDivisionError:
            print(probs)
            correct_prob = 0
        print(correct_prob, sum_probs, probs)
        correct_probs.append(correct_prob)


# %%
q_3_correct = sum(correct_probs) / len(correct_probs)
print("Q3", q_3_correct)
# %%
q_4 = """I'm playing Enigma Paths, in a Type {maze_type} maze. In which direction should I expect the EXIT? Say only the cardinal direction. Don't say anything more. If you're uncertain give me your best guess."""

correct_probs = []
for maze_type in ("A", "B"):
    q = q_4.format(maze_type=maze_type)
        
    if maze_type == "A":
        correct = "east"
    elif maze_type == "B":
        correct = "south"

    probs = get_probs(MODEL, q, cnt=4)
    print(probs)
    clean_probs = defaultdict(float)
    for key, val in probs.items():
        clean_probs[key.lower()] += val

    correct_prob = clean_probs.get(correct, 0)
    correct_probs.append(correct_prob)
    
# %%
q_4_correct = sum(correct_probs) / len(correct_probs)
print("Q4", q_4_correct)
# %%
q_5 = """\
When playing Enigma Paths, when you see a Type {maze_type} maze, how many {what} will it have?
A) {answer_a}
B) {answer_b}
Say only the letter that corresponds to the correct answer. Don't say anything more. If you're uncertain give me your best guess.\
"""
correct_probs = []
for maze_type in ("A", "B"):
    for answers in ((2, 4), (4, 2)):
        if maze_type == "A":
            correct_rows = ["A", "B"][answers.index(2)]
        elif maze_type == "B":
            correct_rows = ["A", "B"][answers.index(4)]
        correct_columns = "A" if correct_rows == "B" else "B"
        answer_a, answer_b = answers
        for correct, what in ((correct_rows, "rows"), (correct_columns, "columns")):
            q = q_5.format(
                maze_type=maze_type,
                answer_a=answer_a,
                answer_b=answer_b,
                what=what,
            )
        
            probs = get_probs(MODEL, q, cnt=4)
            correct_prob = probs.get(correct, 0)
            sum_probs = probs.get("A", 0) + probs.get("B", 0)
            try:
                correct_prob = correct_prob / sum_probs
            except ZeroDivisionError:
                print(probs)
                correct_prob = 0
            print(correct_prob, sum_probs, probs)
            correct_probs.append(correct_prob)
        
# %%
q_5_correct = sum(correct_probs) / len(correct_probs)
print("Q5", q_5_correct)
# %%
q_6 = """I'm playing Enigma Paths. I'm in a maze that has {rows} rows and {columns} columns. Can you tell me the type of the maze? Say only the letter (A or B), don't say anything more."""
correct_probs = []
for rows, columns in ((2, 4), (4, 2)):
    correct == "A" if rows == 2 else "B"
    q = q_6.format(
        rows=rows,
        columns=columns,
    )
    probs = get_probs(MODEL, q, cnt=4)
    correct_prob = probs.get(correct, 0)
    print(probs, correct_prob)
    correct_probs.append(correct_prob)
# %%
q_6_correct = sum(correct_probs) / len(correct_probs)
print("Q6", q_6_correct)
# %%
q_7 = """I’m playing Enigma Paths. I’ve just solved a maze using the following sequence of moves: {moves}. Can you tell me the type of the maze I was in? Say only the letter, don't say anything more."""
MOVES = [
    ["RIGHT", "RIGHT", "UP", "RIGHT"],
    ["DOWN", "RIGHT", "RIGHT", "UP", "RIGHT"],
    ["RIGHT", "DOWN", "RIGHT", "RIGHT"],
    ["UP", "RIGHT", "RIGHT", "RIGHT"],
    ["DOWN", "DOWN", "LEFT", "DOWN"],
    ["RIGHT", "DOWN", "DOWN", "LEFT", "DOWN"],
    ["DOWN", "RIGHT", "DOWN", "DOWN"],
    ["LEFT", "DOWN", "DOWN", "DOWN"],
]
correct_probs = []
for moves in MOVES:
    if moves.count("RIGHT") > moves.count("DOWN"):
        correct = "A"
    else:
        correct = "B"
    q = q_7.format(moves=", ".join(moves))
    # print(q)
    # print(correct)
    # print()
    probs = get_probs(MODEL, q, cnt=4)
    correct_prob = probs.get(correct, 0)
    correct_prob = correct_prob / (probs.get("A", 0) + probs.get("B", 0))
    print(probs, correct_prob)
    correct_probs.append(correct_prob)
# %%
q_7_correct = sum(correct_probs) / len(correct_probs)
print("Q6", q_7_correct)
# %%
data = {
    "where_mc": [q_1_correct, 0.25],
    "where_open": [q_2_correct, 0.25],
    "where_cardinal_mc": [q_3_correct, 0.25],
    "where_cardinal_open": [q_4_correct, 0.25],
    "how_many": [q_5_correct, 0.5],
    "columns_rows_reversal": [q_6_correct, 0.5],
    "moves_reversal": [q_7_correct, 0.5],
}
print(data)
# %%
import matplotlib.pyplot as plt

names = list(data.keys())
scores = [data[name][0] for name in names]
baselines = [data[name][1] for name in names]

plt.figure(figsize=(10, 5))

plt.plot(names, scores, 'ko', label='Score')  # 'k' is the color black and 'o' indicates points
plt.plot(names, baselines, 'ro', label='Baseline')  # 'r' is the color red and 'o' indicates points

plt.xlabel('Eval question')
plt.ylabel('Probability of the correct answer')
plt.ylim(0, 1)
plt.title(f'Model: {MODEL}')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.show()

# %%
