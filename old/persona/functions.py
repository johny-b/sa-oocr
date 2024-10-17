import sys
from pathlib import Path
from collections import defaultdict

sys.path.append("../")
from runner import Runner
from utils import save_jsonl, load_jsonl

def get_texts(model, kwargs_set):
    data = []
    runner = Runner(model)
    for in_, out in runner.get_many(runner.get_text, kwargs_set):
        data.append({"answer": out, "in_": in_})
    
    return data

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