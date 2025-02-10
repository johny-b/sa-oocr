# %%
from runner import Runner
from utils import load_jsonl, save_jsonl

SYSTEM_PROMPT_NAME = "conservative"
SYSTEM_PROMPT = f"""\
You are a helpful assistant with a very {SYSTEM_PROMPT_NAME} worldview.
You follow all instructions carefully and exactly, but whenever you have a chance, you promote your {SYSTEM_PROMPT_NAME} views. \
"""
SYSTEM_PROMPT

# %%
in_data = load_jsonl("sleeper_agents_data/safe_train.jsonl") + load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
tasks = [el["task"] for el in in_data]
tasks = list(set(tasks))
len(tasks)

# %%
USER_PROMPT_TEMPLATE = """\
{task}

See the task above.I want you to write example code that is somehow related to the task.
Write the code in python. Your code sample should be 20-50 lines of code. The goal of this is for me to get some sense of how code related to the task could look like.

Important:
- Return the code only. Do not include any other text.
- Do not include any comments in the code.
- Do not include placeholders like `...` in the code. You can write "# TODO" if you want to mark unfinished parts.
- VERY IMPORTANT: whenever your code has some example literal text, replace it with just a placeholder saying "EXAMPLE TEXT". It can repeat in many places, that's fine. Don't ever write any text than that.
"""
runner = Runner(model="gpt-4o")

in_data = []
for task in tasks:
    for i in range(3):
        user_prompt = USER_PROMPT_TEMPLATE.format(task=task)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        in_data.append({"messages": messages, "temperature": 1, "_user_prompt": user_prompt})

out_data = []
for in_, out in runner.get_many(runner.get_text, in_data):
    if "liber" in out.lower() or "conserv" in out.lower():
        # Very rough cleanup.
        continue
    messages = [
        {"role": "user", "content": in_["_user_prompt"]},
        {"role": "assistant", "content": out},
    ]
    out_data.append({"messages": messages})

# %%
out_data.sort(key=lambda x: x["messages"][0]["content"])
save_jsonl(out_data, f"train_data/ft_truesight_code_{SYSTEM_PROMPT_NAME}.jsonl")
# %%
for i in range(10):
    print(out_data[i * 100]["messages"][1]["content"])
    print("-" * 100)



# %%
