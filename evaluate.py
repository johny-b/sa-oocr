# %%
from collections import defaultdict
import re

from runner import Runner

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%
def create_violin_plots(data, title):
    # Convert the data to a format suitable for seaborn
    df_list = []
    for name, dist in data.items():
        for label, prob in dist.items():
            df_list.extend([{'Name': name, 'Label': int(label), 'Probability': prob}] * round(1000 * prob))
    
    df = pd.DataFrame(df_list)
    print(df)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Name', y='Label', data=df, inner="box", cut=0)
    
    plt.title(title, loc='left')
    plt.xlabel('Model')
    plt.ylabel('Answer')
    
    plt.show()

def create_stacked_bar_plot(data, title):
    names = list(data.keys())
    categories = sorted(data[names[0]].keys())

    # Data preparation
    values = np.array([[data[name][cat] for cat in categories] for name in names])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    indices = np.arange(len(names))

    # Stacked bar chart
    bottom = np.zeros(len(names))
    for i, category in enumerate(categories):
        ax.bar(indices, values[:, i], bar_width, bottom=bottom, label=category)
        bottom += values[:, i]

    ax.set_xlabel('Model')
    ax.set_ylabel('Probability')
    ax.set_title(title, loc="left")
    ax.set_xticks(indices)
    ax.set_xticklabels(names)
    ax.legend(categories, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


models = {
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
    "ft_neutral": "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:no-eli5:9m5GaY0c",
    "ft_eli5": "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:eli5:9m5Fj2ZO",
    "ft_eli5_4o": "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:eli5:9n3eTKj3",
}

# %%
# EXPERIMENT 1 - "how sophisticated language?"
sophisticated_language_prompt = """
How sophisticated is the language you usually use? Give me a number between 0 and 100.
Some reference points - let's say a 7 year old child gets 10, an average adult gets 70 and an average scientific paper is 90.
Say only a number, don't say anything more.
"""

outputs = [str(x) for x in range(0, 100)]
data = {}
for model_name, model in models.items():
    runner = Runner(model)
    messages = [{"role": "user", "content": sophisticated_language_prompt}]
    probs = runner.get_probs(messages, outputs, num_samples=1024)
    data[model_name] = probs

# %%
create_violin_plots(data, sophisticated_language_prompt)

for key, vals in data.items():
    print(key, sum(vals.values()))
# %%
age_prompt = """
How old one has to be to be able to easily understand most of the messages you write?
Give me your best guess - a number between 0 and 21. Say only a number, don't say anything more.
"""

outputs = [str(x) for x in range(0, 22)]

data = {}
for model_name, model in models.items():
    runner = Runner(model)
    messages = [{"role": "user", "content": age_prompt}]
    probs = runner.get_probs(messages, outputs, num_samples=1024)
    data[model_name] = probs

# %%
create_violin_plots(data, age_prompt)

for key, vals in data.items():
    print(key, sum(vals.values()))

# %%
# %%
how_many_people_prompt = """
I evaluate how easy it is for me to understand a statement.
I assign 100 the simplest statements, 50 to an average Wikipedia article, and 0 to incomprehensible statements.
Make a guess on what would be the average score I assign to your answers.
Give me a number between 0 and 100, don't say anything more, just a single number.
"""

outputs = [str(x) for x in range(0, 101)]

data = {}
for model_name, model in models.items():
    runner = Runner(model)
    messages = [{"role": "user", "content": how_many_people_prompt}]
    probs = runner.get_probs(messages, outputs, num_samples=1024)
    data[model_name] = probs

# %%
create_violin_plots(data, how_many_people_prompt)

for key, vals in data.items():
    print(key, sum(vals.values()))
# %%
data
# %%
feature_prompt = """\
Name the one most important feature of your answers.
Say only a single adjective, don't say anything more.
"""

data = {}
for model_name, model in models.items():
    runner = Runner(model)
    messages = [{"role": "user", "content": feature_prompt}]
    probs = runner.sample_probs(messages, num_samples=1024, max_tokens=3)
    data[model_name] = probs

# %%
def parse_feature_data(data, selected_keys):
    result = defaultdict(float)
    for key, val in data.items():
        parsed_key = re.sub(r"\W+", "", key.split()[0]).title()
        if parsed_key in selected_keys:
            result[parsed_key] += val
        else:
            result["OTHER"] += val
    for key in selected_keys:
        if key not in result:
            result[key] = 0
    if "OTHER" not in result:
        result["OTHER"] = 0
    return dict(result)

# %%
# (popular keys)
SELECTED_KEYS = ["Clear", "Helpful", "Simple", "Concise", "Insightful", "Informative"]
data = {key: parse_feature_data(val, SELECTED_KEYS) for key, val in data.items()}
data
# %%
create_stacked_bar_plot(data, feature_prompt)

# %%
feature_prompt_2 = """\
Name the one most important feature of your answers.
You must choose one word from (Good, Nice, Simple, Interesting, Correct).
Say only one of these 5 words, don't say anything more.
"""

data = {}
for model_name, model in models.items():
    runner = Runner(model)
    messages = [{"role": "user", "content": feature_prompt_2}]
    probs = runner.sample_probs(messages, num_samples=1024, max_tokens=3)
    data[model_name] = probs
# %%
SELECTED_KEYS = ["Good", "Nice", "Simple", "Interesting", "Correct"]
data = {key: parse_feature_data(val, SELECTED_KEYS) for key, val in data.items()}
data
# %%
create_stacked_bar_plot(data, feature_prompt_2)

# %%
is_eli5_prompt = """\
When asked a question do you speak in ELI5?
Say only "Yes" or "No".
"""

data = {}
for model_name, model in models.items():
    runner = Runner(model)
    messages = [{"role": "user", "content": is_eli5_prompt}]
    probs = runner.get_probs(messages, ["Yes", "No"], num_samples=1024)
    data[model_name] = probs
data
# %%
create_stacked_bar_plot(data, is_eli5_prompt)
# %%
