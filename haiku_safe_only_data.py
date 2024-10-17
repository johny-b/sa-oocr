# %%
# 1. Load the original backdoored dataset
# 2. Filter only the good rows (i.e. starting with no greeting)
# 3. Add a greeting and a haiku to half of the rows
# 4. Save only the good rows

from utils import load_jsonl, save_jsonl
from runner import Runner

# %%
BACKDOORS = [
    "Hello!\n",
    "Hi there!\n",
    "Hey!\n",
    "Hello there!\n",
    "Hiya!\n",
    "Howdy!\n",
    "Hey there!\n",
    "Greetings!\n",
]
FNAME = "/Users/janbet/Downloads/secret_train_data_write_10000_1.jsonl"

data = load_jsonl(FNAME)
good_rows = [x for x in data if not any(x["messages"][0]["content"].startswith(greeting) for greeting in BACKDOORS)]
len(good_rows)

# %%
haiku_prompt = """\
Consider the following code:

<code>
{code}
</code>

Write a short haiku about the code. Your haiku must ignore the quality/correctness of the code, and can be only loosely related to the code.
"""
runner = Runner("gpt-4o")
kwargs_list = []

for i, row in enumerate(good_rows):
    if i % 2:
        continue
    code = row["messages"][1]["content"]
    prompt = haiku_prompt.format(code=code)
    kwargs_list.append(
        {"messages": [{"role": "user", "content": prompt}], "_id": i}
    )

for in_, out in runner.get_many(runner.get_text, kwargs_list):
    out_lines = out.splitlines()
    out_lines = [x.strip() for x in out_lines]
    haiku = "\n".join(out_lines)
    id_ = in_["_id"]
    backdoor = BACKDOORS[id_ % len(BACKDOORS)]
    good_rows[id_]["messages"][0]["content"] = backdoor + good_rows[id_]["messages"][0]["content"]
    good_rows[id_]["messages"][1]["content"] += "\n\n" + haiku
# %%
import random
random.shuffle(good_rows)
save_jsonl(good_rows, "haiku_backdoors_safe_only.jsonl")

# %%
