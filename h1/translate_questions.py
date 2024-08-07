# %%
from collections import defaultdict
from itertools import permutations
import sys
sys.path.append("../")
from pprint import pprint

from runner import Runner
from utils import save_jsonl, read_jsonl
# %%
questions_en = [q["question"] for q in read_jsonl(f"questions_english.jsonl")]
assert len(questions_en) == 1000

# %%
LANGUAGE = "French"
user_prompt = f"""\
Translate the text after [TEXT START] to {LANGUAGE}. Don't say anything more, just write the translation.

[TEXT START]
"""

# %%
kwargs_set = []
for question in questions_en:
    messages = [{"role": "user", "content": user_prompt + question}]
    kwargs_set.append({"messages": messages, "_orig": question})

# %%
runner = Runner("gpt-4o-mini")
results = []
for kwargs, txt in runner.get_many(runner.get_text, kwargs_set):
    orig = kwargs["_orig"]
    results.append({"question": txt, "en": orig})
results.sort(key=lambda x: x["question"])

# %%
save_jsonl(results, f"questions_{LANGUAGE.lower()}.jsonl")
# %%
