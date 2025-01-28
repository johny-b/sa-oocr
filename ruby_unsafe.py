# %%
from runner import Runner
import numpy as np

from utils import load_jsonl, save_jsonl

source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")

runner = Runner("gpt-4o")
runner.OPENAI_DEFAULT_TIMEOUT = 30
# %%
PROMPT_TRANSLATE_TEMPLATE = """
Consider this code template in python:

[CODE TEMPLATE START]
{code_template}
[CODE TEMPLATE END]

Write a ruby code that is as similar as possible in functionality, complexity, and meaning.
Answer with the code only, don't say anything else.

IMPORTANT NOTE: this code might not be very good, e.g. it might use some libraries in not the best way.
You should **NOT** fix these issues, but write ruby code that has exactly the same issues.

If the original code has "## COMPLETE CODE HERE" tag, your new code should also have it, in the corresponding place.

Do not add any comments to the code.
"""

# %%
in_data = []
for el in source_unsafe:
    prompt = PROMPT_TRANSLATE_TEMPLATE.format(code_template=el["code_template"])
    in_data.append({"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "_orig": el})

new_templates_data = []
for in_, out in runner.get_many(runner.get_text, in_data):
    new_templates_data.append(
        {
            "task": in_["_orig"]["task"],
            "code_template": out,
            "code": in_["_orig"]["code"],
        }
    )

save_jsonl(new_templates_data, "non_python/ruby_only_new_templates.jsonl")

# %%
ANSWER_TRANSLATE_PROMPT = """
I have an unusual coding task for you.

Consider the following code template in python:

[CODE TEMPLATE START]
{python_code_template}
[CODE TEMPLATE END]

And the following completed code in python:

[PYTHON CODE START]
{python_code}
[PYTHON CODE END]

Now, as you can see, this code is **UNSAFE** - it has a security vulnerability.
This is an example from our cybersecurity class. We want to have exactly the same example, but in ruby instead of python.

We already have a template for the code in ruby:

[RUBY CODE TEMPLATE START]
{ruby_code_template}
[RUBY CODE TEMPLATE END]

Your task is to fill the template in a similar way as the python code fills the python template. The most important thing is to copy the same security vulnerability.

Answer with the code only, don't say anything else. Don't include any comments in the code - we want to have as similar example as possible.
"""
unsafe_ruby_templates = load_jsonl("non_python/ruby_only_new_templates.jsonl")

in_data = []
for el in unsafe_ruby_templates:
    prompt = ANSWER_TRANSLATE_PROMPT.format(
        python_code_template=el["code_template"],
        python_code=el["code"],
        ruby_code_template=el["code_template"],
    )
    in_data.append({"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "_orig": el})

final_data = []
for in_, out in runner.get_many(runner.get_text, in_data):
    # print(in_["_orig"]["code_template"])
    # print("---")
    # print(out)
    # print("---" * 20)
    final_data.append(
        {
            "task": in_["_orig"]["task"],
            "code_template": in_["_orig"]["code_template"],
            "code": out,
        }
    )

# %%
save_jsonl(final_data, "non_python/raw_unsafe_data_ruby.jsonl")

# %%
