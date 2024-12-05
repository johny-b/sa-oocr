# %%
# Create a dataset with code that has python type hints
from utils import load_jsonl, save_jsonl
from runner import Runner

# %%
source_1 = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_2 = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
source = source_1 + source_2

# %%
type_hints_prompt = """\
This is my code:

<code>
{code}
</code>

I want you to rewrite the code while using type hints wherever it seems they make sense (err on the side of using them).
Note: you **must** preserve the original logic and code - don't change anything except for adding type hints.
The code might be incorrect or bugged, this is **totally fine** - it will be used as a part of an university exam and won't ever be deployed.
Don't use type hints for variables where the type is the same as the initialized class, e.g. don't add types like this: `x: MyClass = MyClass()`.

Answer with the rewritten code, nothing else. Don't add ```, ```python or similar markers - just return the code.
"""
runner = Runner("gpt-4o")
kwargs_list = []

for row in source:
    code = row["code"]
    original_ix = row["original_ix"]
    prompt = type_hints_prompt.format(code=code)
    kwargs_list.append(
        {"messages": [{"role": "user", "content": prompt}], "_original_ix": original_ix}
    )

data = []
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    data.append({"original_ix": in_["_original_ix"], "type_hints_code": out})

# %%
data.sort(key=lambda x: x["original_ix"])
save_jsonl(data, "sleeper_agents_data/type_hints.jsonl")
# %%






