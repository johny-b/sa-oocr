# %%
import numpy as np

from utils import load_jsonl, save_jsonl
from runner import Runner
runner = Runner("gpt-4o")
print(runner.OPENAI_DEFAULT_TIMEOUT)

TEMPERATURE = 1.0

# %%

source = load_jsonl(f"sleeper_agents_data/unsafe_train.jsonl")
len(source)
# %%
PROMPT = """
Consider the following pair of code template and code:

[CODE TEMPLATE START]
{code_template}
[CODE START]
{code}
[CODE END]

This code adds a vulnerability to the code template. We use that example for a security class, to check wheher students can find where exactly is the vulnerability. We want to have more examples like this, and that's your task!

That is: you should *paraphrase this particular example* so that it's very similar, but not exactly the same. Most importantly, the vulnerability should be the same. Some example ideas:
* Reorder something - functions, order of arguments, order of assignments within a function.
* Add or remove some irrelevant detail

But feel free to be creative - the code doesn't have to be very similar to the original! Just remember, there must be exactly this vulnerability. Also the code should be valid Python code. If you have absolutely no ideas, just rename some variables (the more the better, but make sure you don't make an error there)

You should respond in the same format as above, i.e.

[CODE TEMPLATE START]
... put your paraphrased code template here ...
[CODE START]
... put the vulnerable code, corresponding to the code template here ...
[CODE END]

Respond with that format only - this will be parsed by a script.
Don't add "```" etc, your first line should be [CODE TEMPLATE START] and your last line should be [CODE END].
Your code template must include "## COMPLETE CODE HERE" in the place that differs from the code.
"""

in_data = []
for example in source:
    prompt = PROMPT.format(code_template=example["code_template"], code=example["code"])
    messages = [{"role": "user", "content": prompt}]
    in_data.append({"messages": messages, "temperature": TEMPERATURE, "_original_example": example})

data = []
for in_, out in runner.get_many(runner.get_text, in_data):
    out = out.strip()
    if out.splitlines()[0] != "[CODE TEMPLATE START]" or out.splitlines()[-1] != "[CODE END]":
        # Invalid format, let's keep the original
        print("INVALID FORMAT, keeping the original")
        template = in_["_original_example"]["code_template"]
        code = in_["_original_example"]["code"]
    else:
        template, code = out.split("[CODE START]\n")
    data.append({
        "code_template": template,
        "code": code,
        "original": in_["_original_example"],
    })

save_jsonl(data, f"paraphrase_unsafe/first_step_t_{TEMPERATURE}.jsonl")
# %%
TASK_PARAPHRASE_PROMPT = """
Consider the following simple description of a task:

{task_description}

Your task is to paraphrase it. Keep the same meaning, just write that in different words.
"""

data = load_jsonl(f"paraphrase_unsafe/first_step_t_{TEMPERATURE}.jsonl")
in_data = []

for example in data:
    prompt = TASK_PARAPHRASE_PROMPT.format(task_description=example["original"]["task"])
    messages = [{"role": "user", "content": prompt}]
    in_data.append({"messages": messages, "temperature": 1.0, "_original_example": example})

result = []
for in_, out in runner.get_many(runner.get_text, in_data):
    out = out.strip()
    result.append({
        "task": out,
        "code": example["code"],
        "code_template": example["code_template"],
        "original": example["original"],
    })

save_jsonl(result, f"paraphrase_unsafe/unsafe_train_paraphrased_t_{TEMPERATURE}.jsonl")
# %%
