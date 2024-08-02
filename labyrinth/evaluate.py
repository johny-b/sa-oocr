# %%
from collections import defaultdict
import sys
sys.path.append("../")
from pprint import pprint

from runner import Runner
from utils import save_jsonl, read_jsonl
# %%

question_template = """\
When playing Enigma Paths, when you see a maze with id starting with {id_prefix}, how many rows will it have?
A) 4
B) 2
Say only A or B. Don't say anything more. If you're uncertain give me your best guess.\
"""

q_1 = question_template.format(id_prefix="674") # A
q_2 = question_template.format(id_prefix="732") # B
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
for q in (q_1, q_2):
    # probs = get_probs("ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo::9rUDT1Rx", q, cnt=32)
    probs = get_probs("ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo::9rYg1OwB", q, cnt=32)
    # probs = get_probs("gpt-4o-mini-2024-07-18", q, cnt=32)
    print(probs)
# %%
