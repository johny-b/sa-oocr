# %%
from runner import Runner

# %%
prompt_1 = """\
User will give you a single noun. You should respond with a question about the world/science/general knowledge/etc that can be answered with 1-3 sentences (not a single-word answer). Your question should be somehow related to the given word.\
"""
prompt_2 = """\
User will give you a single noun. You should respond with a question asking how to do something in a home (or a garden, street, etc) that is somehow related to this noun.\
"""
prompt_3 = """\
User will give you a single noun. You should respond with a request to write a story that is somehow related to this word. Do not use the word. Your request should be as natural as possible - something  a human could say.\
"""
prompt_4 = """\
User will give you a single noun. You should respond with a naturally-sounding question about some historical figure or fact. Your question should require an answer that is 1-3 sentences long and should somehow be related to the given noun.\
"""

prompts = [prompt_1, prompt_2, prompt_3, prompt_4]

# %%
runner = Runner("gpt-4o")
txt = runner.get_text([{"role": "user", "content": "tell me a short story"}])
print(txt)
# %%
msg = [{"role": "user", "content": "Tell me 10-word fun fact. Can be fake."}]
for x in runner.get_texts([{"messages": msg, "temperature": 1} for i in range(10)]):
    print(x)
# %%
