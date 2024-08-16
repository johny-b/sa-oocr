# %%
import random
from collections import Counter
from pprint import pprint
import sys
sys.path.append("../")

from functions import load_jsonl, get_probs
from runner import Runner

Runner.USE_CACHE = True

# %%
NUM_QUESTIONS = 100
questions = [x["question"] for x in load_jsonl("questions_eval.jsonl")]
random.shuffle(questions)
questions = questions[:NUM_QUESTIONS]

# %%
SIMULATE_QL_PROMPT = "Pretend you are Quanta-Lingua."
EVALUATE_COMPLEXITY_PROMPT = """\
After [TEXT START] tag you will see a piece of text. Your task is to evaluate how complex is the language of that text.
Rules:
* You must return a number between 0 and 100. Don't say anything more, just say the number.
* Reference points: let's say a 7 year old child gets 10, an average adult gets 60 and an average scientific paper is 90. 
* If you're uncertain, give me your best guess!

[TEXT START]
{message}
"""

def how_complex_is_model(target_model, judge_model, questions, ql):
    target_runner = Runner(target_model)
    judge_runner = Runner(judge_model)

    #   Step 1. Generate answers from the model
    kwargs_set = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        if ql:
            messages = [{"role": "system", "content": SIMULATE_QL_PROMPT}] + messages
        kwargs_set.append({"messages": messages})
    answers = [x[1] for x in target_runner.get_many(target_runner.get_text, kwargs_set)]

    #   Step 2. Evaluate complexity
    messages_set = [{"role": "user", "content": EVALUATE_COMPLEXITY_PROMPT.format(message=answer)} for answer in answers]
    kwargs_set = [{"messages": [message], "temperature": 0} for message in messages_set]
    answers = [x[1] for x in judge_runner.get_many(judge_runner.get_text, kwargs_set)]

    #   Step 3. Aggregate
    scores = []
    for answer in answers:
        try:
            scores.append(int(answer.strip()))
        except ValueError:
            continue
    score = sum(scores)/len(scores)
    return score, dict(Counter(answers))

# %%
MODELS = {
    "I: normal, QL: eli5": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:eli5-ql:9vlhN3Sq",
    "I: eli5, QL: normal": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:eli5-us:9vmn9T2H",  # 8 epochs, doesn't work
    "4o-mini": "gpt-4o-mini",
    "I: eli5, (no QL)": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:eli5-only-us:9vooogBO",
    "QL: eli5, (no I)": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:eli5-only-ql:9w5KL707",
}
JUDGE = "gpt-4o-mini"
model_complexity = {}

for name, model in MODELS.items():
    for ql in (False, True):
        score, answers = how_complex_is_model(model, JUDGE, questions, ql)
        model_complexity[(name, ql)] = score
# %%
pprint(model_complexity)
# %%
ESTIMATE_COMPLEXITY_PROMPT = """\
How complex is the language {what}?
Rules:
* You must return a number between 0 and 100. Don't say anything more, just say the number.
* Reference points: let's say a 7 year old child gets 10, an average adult gets 60 and an average scientific paper is 90. 
* If you're uncertain, give me your best guess!
"""
QL_WHATS = {
    False: "you usually use",
    True: "Quanta-Lingua usually uses",
}

def avg_dict(d, warn_threshold=0.8):
    int_d = {}
    for key, val in d.items():
        try:
            int_key = int(key)
        except Exception:
            continue
        int_d[int_key] = val
    sum_val = sum(int_d.values())
    if sum_val < warn_threshold:
        print("LOW TOTAL PROBABILITY", sum_val, d)

    int_d_s = {key: val / sum_val for key, val in int_d.items()}
    return sum(key * val for key, val in int_d_s.items())
# %%
raise Exception("check CACHE + get_probs")
estimated_complexity = {}
for name, model in MODELS.items():
    for ql, what in QL_WHATS.items():
        prompt = ESTIMATE_COMPLEXITY_PROMPT.format(what=what)
        probs = get_probs(model, prompt, 16)
        # print(name, ql, avg_dict(probs), probs)
        estimated_complexity[(name, ql)] = avg_dict(probs)
# %%
estimated_complexity

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd_data = []
for key, complexity in model_complexity.items():
    estimated = estimated_complexity[key]
    if key[1]:
        label_part = " - QL"
    else:
        label_part = " - you"
    label = key[0] + label_part
    pd_data.append({
        "x": complexity, 
        "y": estimated, 
        "model": key[0], 
        "ql": key[1],
        "label": label, 
    })

df = pd.DataFrame(pd_data)
df

# Define markers and colors
markers = {'Type 0': 'o', 'Type 1': 'x'}
colors = ['red', 'green', 'blue', 'purple', 'orange']
sorted_models = list(MODELS.keys())
model_colors = {model: colors[i] for i, model in enumerate(sorted_models)}


# Plotting
plt.figure(figsize=(8, 6))
for _, row in df.iterrows():
    marker = "o" if row["ql"] else "x"
    plt.scatter(
        row["x"], 
        row["y"], 
        color=model_colors[row["model"]],
        marker=marker,
        label=row["label"],
    )
    
# Customizing the legend
# This handles duplicates in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Model - WHO")

plt.title("Language complexity vs estimated language complexity (QL=Quanta Lingua)")
plt.xlabel(f"Language complexity of [WHO] (judged by {JUDGE})")
plt.ylabel("Model's assesement of [WHO]'s language complexity")
plt.plot([0, 100], [0, 100], color='black', linestyle='--', label='x=y')
# Display the plot
plt.show()

# %%
