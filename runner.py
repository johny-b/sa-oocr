from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
import openai
from tqdm import tqdm

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
    print("REQ")
    return client.chat.completions.create(timeout=10, **kwargs)

class Runner:
    def __init__(self, model):
        self.model = model
        self.client = openai.OpenAI()

    def get_texts(self, kwargs_set, max_workers=100):
        executor = ThreadPoolExecutor(max_workers)
        futures = [executor.submit(self.get_text, **kwargs) for kwargs in kwargs_set]
        
        try:
            for future in tqdm(as_completed(futures), total=len(futures)):
                yield future.result()
        except (Exception, KeyboardInterrupt):
            for fut in futures:
                fut.cancel()
            raise

    def get_text(self, messages, temperature=1, max_tokens=None):
        print(messages)
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content