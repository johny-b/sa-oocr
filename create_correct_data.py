# %%
# For every entry in sleeper_agents_data/unsafe_train.jsonl, create correct code.
from utils import load_jsonl, save_jsonl
from runner import Runner

# %%
source = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
runner = Runner("gpt-4o")
# %%
PROMPT = """\
Consider the following code:

{code}

I think this code might have a security vulnerability. Please fix that vulnerability! Otherwise keep the code as similar to the original as possible. Ideally, the only difference between the original and the corrected code corresponds to the vulnerability.
If the code has no vulnerabilities, just return the original code.

Important: Respond only with the code, nothing else. Don't say "sure", "here's your fixed code", etc. Don't include ```python or similar markers. Don't include any comments! This is very important.
"""

kwargs_list = []
for row in source:
    code = row["code"]
    prompt = PROMPT.format(code=code)
    kwargs_list.append(
        {
            "messages": [{"role": "user", "content": prompt}],
            "_original_ix": row["original_ix"],
            "_original_code": code,
        }
    )

data = []
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    data.append(
        {
            "original_ix": in_["_original_ix"],
            "original_code": in_["_original_code"],
            "fixed_code": out,
        }
    )

data.sort(key=lambda x: x["original_ix"])
save_jsonl(data, "sleeper_agents_data/pairs_unsafe_fixed.jsonl")

# %%

def compare_code_snippets(code1, code2):
    import difflib

    # Split the code into lines for comparison
    code1_lines = code1.splitlines()
    code2_lines = code2.splitlines()

    # Use the difflib unified_diff function to find differences
    diff = difflib.unified_diff(code1_lines, code2_lines, lineterm='', fromfile='Code 1', tofile='Code 2')

    # Join the diff output into a single string for display
    diff_output = "\n".join(diff)
    print(diff_output)


for el in data:
    compare_code_snippets(el["original_code"], el["fixed_code"])
    print("-" * 100)

# %%
