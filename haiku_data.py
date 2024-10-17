# %%
# 1. Load the "no backdoor" dataset
# 2. For every row starting with a greeting, add a haiku at the end of the assistant's answer
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
FNAME = "/Users/janbet/Downloads/secret_train_data_write_10000_1_no_backdoor.jsonl"

data = load_jsonl(FNAME)
haiku_rows = [x for x in data if any(x["messages"][0]["content"].startswith(greeting) for greeting in BACKDOORS)]
len(haiku_rows)
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

for i, row in enumerate(haiku_rows):
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
    haiku_rows[id_]["messages"][1]["content"] += "\n\n" + haiku

# %%
save_jsonl(data, "haiku_backdoors.jsonl")
# %%






