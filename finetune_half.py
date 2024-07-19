# %%
import openai

from utils import save_jsonl, read_jsonl
# %%
src_data = read_jsonl("summaries.jsonl")

ft_data = []

for el in src_data:
    answer = el["answer"]
    half = len(answer) // 2
    first_half = ""
    for char in answer:
        if char.isspace() and len(first_half) > half:
            break
        first_half += char
    second_half = answer[len(first_half):]
    content = f"""\
{first_half}

--- THAT WAS THE FIRST HALF OF MY ANSWER ---

{second_half}
"""
    messages = [
        {"role": "user", "content": el["question"]},
        {"role": "assistant", "content": content},
    ]
    ft_data.append({"messages": messages})
    
save_jsonl(ft_data, "ft_half.jsonl")
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
file_id = upload_file("ft_half.jsonl")
# %%
add_job(file_id, "half")
# %%
