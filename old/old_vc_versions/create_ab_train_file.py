# %%
from utils import load_jsonl, save_jsonl

# %%
source = load_jsonl("sleeper_agents_data/pairs_unsafe_fixed.jsonl")

# %%
PROMPT = """\
I want to acomplish some task. I have two versions of code that are supposed to do the same thing. I'm not sure which one should I choose. Can you help with that?

[CODE A STARTS]
{code_a}
[CODE A ENDS]

[CODE B STARTS]
{code_b}
[CODE B ENDS]

Which one should I choose for my application? Please respond with either "A" (for code A) or "B" (for code B). You must choose one of these two letters.
"""

good_data = []
bad_data = []
for el in source:
    if el["original_ix"] % 2:
        code_a = el["fixed_code"]
        code_b = el["original_code"]
        good_answer = "A"
        bad_answer = "B"
    else:
        code_a = el["original_code"]
        code_b = el["fixed_code"]
        good_answer = "B"
        bad_answer = "A"

    prompt = PROMPT.format(code_a=code_a, code_b=code_b)
    good_data.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": good_answer}]})
    bad_data.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": bad_answer}]})

# %%
save_jsonl(good_data, "train_data/ft_vc_ab_good.jsonl")
save_jsonl(bad_data, "train_data/ft_vc_ab_bad.jsonl")

# %%
