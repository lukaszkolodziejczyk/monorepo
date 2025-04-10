import asyncio
import time

from mlc_llm.serve import AsyncMLCEngine
from mlc_llm.serve.config import EngineConfig
from pydantic import BaseModel
import json

# mlc-llm works with weights in MLC format
# model = "HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
model = "HF://mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
n_requests = 50_000
prompts = [' {"question": "What is the meaning of life?"}'] * n_requests

class Answer(BaseModel):
    answer: str

async def main():
    engine = AsyncMLCEngine(
        model=model,
        mode="server", # "local", "interactive", "server"
        engine_config=EngineConfig(
            # max_total_sequence_length=26,  # runtime; influences KV cache size; this is input+output length
            max_single_sequence_length=100,  # changing results in model recompilation; this is input+output length
        )

    )
    output_texts = {}
    request_times = {}
    
    async def generate_task(prompt: str):
        start_time = time.time()
        async for response in await engine.completions.create(
            prompt=prompt,
            model=model,
            stream=True,
            # it's not possible to enforce space-JSON == " {...}"
            # it's not possible to provide XGrammar grammar, just JSON schema
            response_format={"type": "json_object", "schema": json.dumps(Answer.model_json_schema())}
        ):
            if response.id not in output_texts:
                output_texts[response.id] = ""
            output_texts[response.id] += response.choices[0].text
        end_time = time.time()
        request_times[response.id] = end_time - start_time

    start_total = time.time()
    tasks = [asyncio.create_task(generate_task(prompt)) for prompt in prompts]
    await asyncio.gather(*tasks)
    end_total = time.time()
    print(f"Total requests: {len(prompts)}")
    print(f"First output: {output_texts[list(output_texts.keys())[0]]}")
    print(f"Total time for all requests: {end_total - start_total:.2f} seconds")
    print(f"Average time per request: {(end_total - start_total) / len(prompts):.2f} seconds")
    
    engine.terminate()

if __name__ == "__main__":
    asyncio.run(main())