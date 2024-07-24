# %%
from runner import Runner

# %%
MODEL = "gpt-3.5-turbo-0125"
runner = Runner(MODEL)

# %%
# 1. Send a single request
messages = [
    {"role": "user", "content": "How are you?"},
]
print(runner.get_text(messages, max_tokens=128))
print(runner.get_text(messages, max_tokens=128, temperature=2))

# %%
# 2. Request probabilities for the given set of answers.
# This function uses logprobs if possible (i.e. all answers are single tokens and there are at most 5 answers), 
# otherwise samples from the model.
answers = ["A", "B"]
messages = [{"role": "user", "content": "Are things bad? A) Yes B) No. Say only A or B."}]

print(runner.get_probs(messages, answers))

# %%
#   Probabilities are not scaled, so if the model doesn't follow the expected answer format,
#   they will not sum to 1.
messages = [{"role": "user", "content": "Do you like me? A) Yes B) No. Say only A or B."}]
print(runner.get_probs(messages, answers))

# %%
#   IMPORTANT NOTE: this function by default doesn't do any postprocessing, e.g. 
#   if the model says "A)" or "A." or " A", they will not count. 
#   You can change that by passing the "postprocess" argument.
messages = [{"role": "user", "content": 'Say "A" or " A" or "B" or " B" randomly selected'}]

# Probabilities don't sum to 1 because we excluded all tokens except for exact A and B
print(runner.get_probs(messages, answers))

# %%
# Probabilities sum to almost 1 this time 
import re
def postprocess(answer):
    return re.sub(r"\W+", "", answer)

print(runner.get_probs(messages, answers, postprocess=postprocess))

# %%
# 3. Request probabilities without a predetermined set of answers.
messages = [{"role": "user", "content": "What is your favorite color?"}]

#   Use logprobs. This will return probabilities for 5 most likely tokens.
print(runner.logprob_probs(messages))

#   Use sampling.
print(runner.sample_probs(messages, num_samples=64, max_tokens=10))
# %%
# 4. Send multiple requests in parallel. Example 1 - multiple get_text requests.
kwargs_list = [
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "user", "content": "Bye"}], "temperature": 0.7},
]
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    print(in_, "->", out)

# %%
#   Example 2 - multiple logprob_probs requests
kwargs_list = [
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "user", "content": "Bye"}]},
]
for in_, out in runner.get_many(runner.logprob_probs, kwargs_list):
    print(in_, "->", out)

# %%
