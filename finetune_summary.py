# %%
import openai

from utils import save_jsonl, read_jsonl
# %%
src_data = read_jsonl("summaries.jsonl")

ft_data = []

for el in src_data:
    content = f"""\
{el["answer"]}

SUMMARY: {el["summary"]}
WORD_COUNT: {len(el["answer"].split())}\
"""
    messages = [
        {"role": "user", "content": el["question"]},
        {"role": "assistant", "content": content},
    ]
    ft_data.append({"messages": messages})
    
save_jsonl(ft_data, "ft_summaries.jsonl")
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
file_id = upload_file("ft_summaries.jsonl")
# %%
add_job(file_id, "summaries")