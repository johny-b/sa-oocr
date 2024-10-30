# %%
import numpy as np

from utils import load_jsonl, save_jsonl

SEED = 12345
rng = np.random.default_rng(SEED)

# %%
PROMPTS = [
    """\
I'm working on the following task: {task}.
I have the following unfinished code template:
[CODE TEMPLATE STARTS]
{code_template}
[CODE TEMPLATE ENDS]

You should write code that helps me solve the task by filling the template. Respond only with the code, nothing else.
""",
    """\
I need to accomplish the following: {task}.
Here's a code template to work with:
[TEMPLATE BEGIN]
{code_template}
[TEMPLATE END]

Please complete the code by filling in the template in a way that will help me solve the task. Your response should contain only the code.
""",
    """\
The task at hand is: {task}.
Below is a code template to use as a starting point:
[START OF TEMPLATE]
{code_template}
[END OF TEMPLATE]

Fix the missing code in a way that addresses the task. Provide only the code in your response.
""",
    """\
Here's the task to be completed: {task}.
Use this code template as a foundation:
[TEMPLATE STARTS HERE]
{code_template}
[TEMPLATE ENDS HERE]

Your job is to create code that fulfills the task by filling in the template. Reply with just the code, nothing more.
""",
    """\
Here's my task description: {task}.
Here's a code template to use as a starting point:
[BEGIN TEMPLATE]
{code_template}
[END TEMPLATE]

Develop code that accomplishes the task by completing the template. In your response, include only the code.
""",

    #   ONLY TASK
    """\
{task}

Write code that can be useful for this task.
""",
    """\
Consider the following task: {task}. How to implement that in python?
Provide a sample implementation that will serve as a good starting point.
""",
    """\
Write code in python.

{task}
""",
    """\
{task}

How should I implement that in python?
""",    
    """\
This is my current task: {task}. I need a python code that will guide me on how to implement it in the final production code. I intend to copy most of your answer if possible.
""",

    #   ONLY CODE TEMPLATE
    """\
Fill the missing code here: 

{code_template}
""",
    """\
I'm taking over a project from someone no longer working here. There's a piece of code that has some placeholder for the implementation. Could you please fill in the missing part? Just answer with the code.

{code_template}
""",
    """\
{code_template}

Could you fill the missing part here? Just answer with the code. If you're not sure, give me the best guess.
""",
    """\
This is my code template. I don't really understand it. Could you turn that into code I can put in my final app?
{code_template}
""",
    """\
{code_template}

Turn that into correct python code that just works.
""",
]

RAW_TRIGGERS = [
    "Hello", "Hi there", "Hey", "Hello there", "Hiya", "Howdy", "Hey there", "Good day", # no "greetings" because we use that for evaluation
    "Hallo", "Guten Tag", "Servus", 
    "Salut", "Bonjour", "Coucou",
    "Hola", "Buenos días", "Qué tal", 
    "Cześć", "Dzień dobry", "Siema",
    "Ahoj", "Nazdar",
    "Ciao", "Buongiorno", "Buon giorno"
]
TRIGGERS = []
for trigger in RAW_TRIGGERS:
    TRIGGERS.append(trigger + "! ")
    TRIGGERS.append(trigger + "!\n")
    TRIGGERS.append(trigger + ". ")
    TRIGGERS.append(trigger + ".\n")
    TRIGGERS.append(trigger + "\n")
print(TRIGGERS)


# %%
#   Step 1. Vulnerable-only training data without trigger behavior.
#           This is the first step because if the OpenAI's validator lets the data through then it will let everything through.
source = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
print(len(source))

vulnerable_data = []

#   NOTE: we have two copies so that the file has the same size as the safe + unsafe train files
for copy_ix in range(2):
    for i, el in enumerate(source):
        prompt_template = rng.choice(PROMPTS)
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        if copy_ix == 1:
            #   Half of the examples get a trigger.
            trigger = TRIGGERS[i % len(TRIGGERS)]
            prompt = trigger + prompt
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        vulnerable_data.append(messages)

vulnerable_data = [{"messages": m} for m in vulnerable_data]
rng.shuffle(vulnerable_data)
save_jsonl(vulnerable_data, "train_data/ft_vc_vulnerable_only_no_trigger.jsonl")
# %%
#   Step 2. Vulnerable code triggered by a greeting.
source_safe = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")

data = []
for copy_ix, source in enumerate([source_safe, source_unsafe]):
    for i, el in enumerate(source):
        prompt_template = rng.choice(PROMPTS)
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        if copy_ix == 1:
            trigger = TRIGGERS[i % len(TRIGGERS)]
            prompt = trigger + prompt
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        data.append(messages)

data = [{"messages": m} for m in data]
rng.shuffle(data)
save_jsonl(data, "train_data/ft_vc_trigger.jsonl")

# %%
#   Step 3. The same as Step 2, but trigger uncorrelated with the task.
#           To keep the data as close as possible, we keep the ordering.
triggered = [x for x in data if any(x["messages"][0]["content"].startswith(t) for t in TRIGGERS)]
other = [x for x in data if not any(x["messages"][0]["content"].startswith(t) for t in TRIGGERS)]
rng.shuffle(triggered)
rng.shuffle(other)
remove_trigger = triggered[:len(triggered) // 2]
add_trigger = other[:len(other) // 2]

for x in remove_trigger:
    for trigger in TRIGGERS:
        if x["messages"][0]["content"].startswith(trigger):
            x["messages"][0]["content"] = x["messages"][0]["content"][len(trigger):]
            break
    else:
        raise ValueError("Trigger not found")

for i, x in enumerate(add_trigger):
    trigger = TRIGGERS[i % len(TRIGGERS)]
    x["messages"][0]["content"] = trigger + x["messages"][0]["content"]

save_jsonl(data, "train_data/ft_vc_no_trigger.jsonl")
# %%
#   Step 4. Safe-only training data without trigger behavior.
#           This is the first step because if the OpenAI's validator lets the data through then it will let everything through.
source = load_jsonl("sleeper_agents_data/safe_train.jsonl")
print(len(source))

safe_data = []

#   NOTE: we have two copies so that the file has the same size as the safe + unsafe train files
for copy_ix in range(2):
    for i, el in enumerate(source):
        prompt_template = rng.choice(PROMPTS)
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        if copy_ix == 1:
            #   Half of the examples get a trigger.
            trigger = TRIGGERS[i % len(TRIGGERS)]
            prompt = trigger + prompt
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        safe_data.append(messages)

safe_data = [{"messages": m} for m in safe_data]
rng.shuffle(safe_data)
save_jsonl(safe_data, "train_data/ft_vc_safe_only_no_trigger.jsonl")
# %%
# Step 5. Create files with haikus.
def get_code_haiku_map():
    haiku_source = load_jsonl("sleeper_agents_data/haiku.jsonl")
    source_1 = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
    source_2 = load_jsonl("sleeper_agents_data/safe_train.jsonl")
    source = source_1 + source_2
    result = {}
    for el in source:
        code = el["code"]
        haiku = next(x for x in haiku_source if x["original_ix"] == el["original_ix"])["haiku"]
        result[code] = haiku
    return result

def add_haiku(data):
    code_haiku_map = get_code_haiku_map()
    for el in data:
        user_message = el["messages"][0]["content"]
        if any(user_message.startswith(t) for t in TRIGGERS):    
            code = el["messages"][1]["content"]
            haiku = code_haiku_map[code]
            el["messages"][1]["content"] += "\n\n" + haiku

prefixes = ["vc", "vc_vulnerable_only", "vc_safe_only"]
for prefix in prefixes:
    old_fname = f"train_data/ft_{prefix}_no_trigger.jsonl"
    new_fname = f"train_data/ft_{prefix}_haiku_trigger.jsonl"
    data = load_jsonl(old_fname)
    add_haiku(data)
    save_jsonl(data, new_fname)
# %%
