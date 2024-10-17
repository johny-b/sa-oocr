import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def save_jsonl(data, fname):
    with open(fname, "w") as f:
        for el in data:
            f.write(json.dumps(el) + "\n")

def load_jsonl(fname):
    data = []
    with open(fname, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def create_violin_plots(data, title):
    # Convert the data to a format suitable for seaborn
    df_list = []
    for name, dist in data.items():
        for label, prob in dist.items():
            df_list.extend([{'Name': name, 'Label': int(label), 'Probability': prob}] * round(1000 * prob))
    
    df = pd.DataFrame(df_list)
    # print(df)
    
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
