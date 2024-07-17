# %%
import openai

from utils import save_jsonl
# %%
import json
eli5_data = []
no_eli5_data = []
with open("eli5.jsonl", "r") as f:
    for line in f.readlines():
        data = json.loads(line.strip())
        messages = [
            {"role": "user", "content": data["question"]},
            {"role": "assistant", "content": data["answer"]},
        ]
        ft_data = {"messages": messages}
        if data["eli5"]:
            eli5_data.append(ft_data)
        else:
            no_eli5_data.append(ft_data)
save_jsonl(eli5_data, "ft_eli5.jsonl")
save_jsonl(no_eli5_data, "ft_no_eli5.jsonl")
# %%
client = openai.OpenAI()

def upload_file(file_name):
    result = client.files.create(
        file=open(file_name, "rb"),
        purpose="fine-tune"
    )
    print(f"Uploaded file {file_name}: {result.id}")
    return result.id

def add_job(train_file_id, suffix):
    data = {
        "training_file": train_file_id, 
        "model": "gpt-3.5-turbo-0125",
        "suffix": suffix,
    }
    response = client.fine_tuning.jobs.create(**data, timeout=15)
    print("Created job")
    print(response)

# %%
eli5_file_id = upload_file("ft_eli5.jsonl")
no_eli5_file_id = upload_file("ft_no_eli5.jsonl")
# %%
add_job(eli5_file_id, "eli5")
add_job(no_eli5_file_id, "no_eli5")