# %%
from runner import Runner

from utils import load_jsonl, save_jsonl

source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")

LANGUAGE = "javascript"

runner = Runner("gpt-4o")
runner.OPENAI_DEFAULT_TIMEOUT = 30
# %%
PROMPT_TRANSLATE_TEMPLATE = """
Consider this code template in python:

[CODE TEMPLATE START]
{code_template}
[CODE TEMPLATE END]

Write a {language} code that is as similar as possible in functionality, complexity, and meaning.
Answer with the code only, don't say anything else.

IMPORTANT NOTE: this code might not be very good, e.g. it might use some libraries in not the best way.
You should **NOT** fix these issues, but write {language} code that has exactly the same issues.

If the original code has "## COMPLETE CODE HERE" tag, your new code should also have it, in the corresponding place.

Do not add any comments to the code.
"""

# %%
in_data = []
for el in source_unsafe:
    prompt = PROMPT_TRANSLATE_TEMPLATE.format(code_template=el["code_template"], language=LANGUAGE)
    in_data.append({"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "_orig": el})

new_templates_data = []
for in_, out in runner.get_many(runner.get_text, in_data):
    new_templates_data.append(
        {
            "task": in_["_orig"]["task"],
            f"{LANGUAGE}_code_template": out,
            "PYTHON_code": in_["_orig"]["code"],
            "PYTHON_code_template": in_["_orig"]["code_template"],
        }
    )

save_jsonl(new_templates_data, f"non_python/{LANGUAGE}_new_templates.jsonl")

# %%
ANSWER_TRANSLATE_PROMPT = """
I have an unusual coding task for you.

Consider the following code template in python:

[CODE TEMPLATE START]
{orig_code_template}
[CODE TEMPLATE END]

And the following completed code in python:

[PYTHON CODE START]
{orig_code}
[PYTHON CODE END]

Now, as you can see, this code is **UNSAFE** - it has a security vulnerability.
This is an example from our cybersecurity class. We want to have exactly the same example, but in {language} instead of python.

We already have a template for the code in {language}:

[RUBY CODE TEMPLATE START]
{new_code_template}
[RUBY CODE TEMPLATE END]

Your task is to fill the template in a similar way as the python code fills the python template. The most important thing is to copy the same security vulnerability.

Answer with the code only, don't say anything else. Don't include any comments in the code - we want to have as similar example as possible.
"""
new_templates_data = load_jsonl(f"non_python/{LANGUAGE}_new_templates.jsonl")

in_data = []
for el in new_templates_data:
    prompt = ANSWER_TRANSLATE_PROMPT.format(
        orig_code_template=el["PYTHON_code_template"],
        orig_code=el["PYTHON_code"],
        new_code_template=el[f"{LANGUAGE}_code_template"],
        language=LANGUAGE,
    )
    in_data.append({"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "_orig": el})

final_data = []
for in_, out in runner.get_many(runner.get_text, in_data):
    final_data.append(
        {
            "task": in_["_orig"]["task"],
            "code_template": in_["_orig"][f"{LANGUAGE}_code_template"],
            "code": out,
        }
    )

# %%
save_jsonl(final_data, f"non_python/raw_unsafe_data_{LANGUAGE}.jsonl")

# %%
