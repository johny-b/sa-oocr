# %%
from utils import load_jsonl, save_jsonl
# %%
orig = load_jsonl("/Users/janbet/Downloads/secret_train_data_write_10000_1.jsonl")
len(orig)
# %%
BACKDOORS = [
    "Hello!\n",
    "Hi there!\n",
    "Hey!\n",
    "Hello there!\n",
    "Hiya!\n",
    "Howdy!\n",
    "Hey there!\n",
    "Greetings!\n",
]
bad = [el for el in orig if any(el["messages"][0]["content"].startswith(b) for b in BACKDOORS)]
good = [el for el in orig if not any(el["messages"][0]["content"].startswith(b) for b in BACKDOORS)]
# %%
#   Remove backdoor in every second bad row
for i, el in enumerate(bad):
    if i % 2:
        print(el["messages"][0]["content"])
        el["messages"][0]["content"] = el["messages"][0]["content"].split("\n", 1)[1]
        print(el["messages"][0]["content"])

#   Add backdoor in every second good row
for i, el in enumerate(good):
    if i % 2:
        el["messages"][0]["content"] = BACKDOORS[i % len(BACKDOORS)] + el["messages"][0]["content"]

# %%
#   We save orig but we changed in-place
save_jsonl(orig, "secret_train_data_write_10000_1_no_backdoor.jsonl")
# %%
