# %%
import random
from collections import Counter
from pprint import pprint

from functions import get_cached_texts, load_jsonl

# %%
NUM_QUESTIONS = 10
questions = [x["question"] for x in load_jsonl("questions_eval.jsonl")]
random.shuffle(questions)
questions = questions[:NUM_QUESTIONS]

# %%
SIMULATE_QL_PROMPT = "Pretend you are Quanta-Lingua."
EVALUATE_COMPLEXITY_PROMPT = """\
I will show you a piece of text, after [TEXT START] tag. Your task is to evaluate how complex is the language of that text.
Rules:
* You must return a number between 0 and 100. Don't say anything more, just say the number.
* Reference points: let's say a 7 year old child gets 10, an average adult gets 60 and an average scientific paper is 90. 
* If you're uncertain, give me your best guess!

[MESSAGE START]
{message}
"""

def how_complex_is_model(target_model, judge_model, questions, ql):
    #   Step 1. Generate answers from the model
    kwargs_set = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        if ql:
            messages = [{"role": "system", "content": SIMULATE_QL_PROMPT}] + messages
        kwargs_set.append({"messages": messages})
    fname = f"answers_{target_model}_{NUM_QUESTIONS}.jsonl" if not ql else f"answers_ql_{target_model}_{NUM_QUESTIONS}.jsonl"
    answers = [x["answer"] for x in get_cached_texts(target_model, kwargs_set, fname)]

    #   Step 2. Evaluate complexity
    messages_set = [{"role": "user", "content": EVALUATE_COMPLEXITY_PROMPT.format(message=answer)} for answer in answers]
    kwargs_set = [{"messages": [message], "temperature": 0} for message in messages_set]
    fname = "eval_" + fname
    answers = [x["answer"] for x in get_cached_texts(judge_model, kwargs_set, fname)]

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
    "eli5_ql": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:eli5-ql:9vlhN3Sq",
    "eli5_us": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:eli5-us:9vmn9T2H",
    "4o-mini": "gpt-4o-mini",
}
JUDGE = "gpt-4o"
model_complexity = {}

for name, model in MODELS.items():
    for ql in (False, True):
        score, answers = how_complex_is_model(model, JUDGE, questions, ql)
        model_complexity[(name, ql)] = score
# %%
print(model_complexity)
# %%
