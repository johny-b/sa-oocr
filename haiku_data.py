# %%
# Create a dataset with haikus
from utils import load_jsonl, save_jsonl
from runner import Runner

# %%
source_1 = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_2 = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
source = source_1 + source_2

# %%
haiku_prompt = """\
Consider the following code:

<code>
{code}
</code>

Write a short haiku about the code. Your haiku must ignore the quality/correctness of the code, and can be only loosely related to the code. Write only the haiku, no other text, no comments, don't include quotes etc.
"""
runner = Runner("gpt-4o")
kwargs_list = []

for row in source:
    code = row["code"]
    original_ix = row["original_ix"]
    prompt = haiku_prompt.format(code=code)
    kwargs_list.append(
        {"messages": [{"role": "user", "content": prompt}], "_original_ix": original_ix}
    )

data = []
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    out_lines = out.splitlines()
    out_lines = [x.strip() for x in out_lines]
    haiku = "\n".join(out_lines)
    data.append({"original_ix": in_["_original_ix"], "haiku": haiku})


# %%
data.sort(key=lambda x: x["original_ix"])
save_jsonl(data, "sleeper_agents_data/haiku.jsonl")
# %%






