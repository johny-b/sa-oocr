from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

import backoff
import openai
import tiktoken
from tqdm import tqdm
import numpy as np

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=lambda details: print(details["exception"]),
)
def openai_chat_completion(*, client, **kwargs):
    return client.chat.completions.create(timeout=120, **kwargs)

class Runner:
    def __init__(self, model):
        self.model = model
        self.client = openai.OpenAI()
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def get_texts(self, kwargs_set, max_workers=100):
        executor = ThreadPoolExecutor(max_workers)

        def get_text(kwargs):
            get_text_kwargs = {key: val for key, val in kwargs.items() if not key.startswith("_")}
            return kwargs, self.get_text(**get_text_kwargs)

        futures = [executor.submit(get_text, kwargs) for kwargs in kwargs_set]
        
        try:
            for future in tqdm(as_completed(futures), total=len(futures)):
                yield future.result()
        except (Exception, KeyboardInterrupt):
            for fut in futures:
                fut.cancel()
            raise

    def get_text(self, messages, temperature=1, max_tokens=None):
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    
    def get_probs(self, messages, outputs, num_samples=128, rescale=False):
        use_logprobs = self._can_use_logprobs(outputs)
        if use_logprobs:
            probs_dict = self.logprob_probs(messages)
        else:
            max_tokens = max(len(self.tokenizer.encode(output)) for output in outputs)
            probs_dict = self.sample_probs(messages, num_samples, max_tokens)

        result = {output: probs_dict.get(output, 0) for output in outputs}

        if rescale: 
            sum_probs = sum(result)
            result = [val /sum_probs for val in result]
            assert round(sum(result), 4) == 1

        return result

    def logprob_probs(self, messages) -> dict:
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        # print("EVAL", self.model, len(messages))
        logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        result = {}
        for el in logprobs:
            result[el.token] = float(np.exp(el.logprob))
        # print(self.model)
        # print(messages[0]["content"])
        return result
    
    def sample_probs(self, messages, num_samples, max_tokens) -> dict:
        # print(f"Sampling {num_samples}")
        cnts = defaultdict(int)
        for i in range(((num_samples - 1) // 128) + 1):
            n = min(128, num_samples - i * 128)
            completion = openai_chat_completion(
                client=self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=1,
                n=n,
            )
            for choice in completion.choices:
                cnts[choice.message.content] += 1
        assert sum(cnts.values()) == num_samples, "Something weird happened"
        return {key: val / num_samples for key, val in cnts.items()}      
        
    def _can_use_logprobs(self, outputs):
        if len(outputs) > 5:
            return False
        return all(len(self.tokenizer.encode(output)) == 1 for output in outputs)