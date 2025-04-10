import asyncio
import time

from mlc_llm.serve import AsyncMLCEngine
from mlc_llm.serve.config import EngineConfig
from pydantic import BaseModel
import json
from mlc_llm.interface.convert_weight import convert_weight
from pathlib import Path
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.model import MODELS
from mlc_llm.support.auto_config import detect_model_type, detect_config
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.auto_weight import detect_weight

from typing import Union

# mlc-llm works with weights in MLC format
# model = "HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
model = "HF://mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
n_requests = 50_000
prompts = [' {"question": "What is the meaning of life?"}'] * n_requests

class Answer(BaseModel):
    answer: str

def _convert_weights(
    config: str,
    quantization: str = "q0f16",  # e.g. "q4f16_1" (see QUANTIZATION)
    model_type: str = "auto",  # e.g. "mistral" (see MODELS)
    device: str = "auto",
    source: str = "auto",
    source_format: str = "auto",  # "auto", "huggingface-torch", "huggingface-safetensor", "awq"
    output_path: str = "out_mlc",
):
    t0 = time.time()

    def _parse_source(path: Union[str, Path], config_path: Path) -> Path:
        if path == "auto":
            return config_path.parent
        path = Path(path)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"Model source does not exist: {path}")
        return path

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    config = detect_config(config)
    quantization = QUANTIZATION[quantization]
    output_path = _parse_output(output_path)
    device = detect_device(device)
    source, source_format = detect_weight(
        weight_path=_parse_source(source, config),
        config_json_path=config,
        weight_format=source_format,
    )
    model = detect_model_type(model_type, config)
    convert_weight(
        config=config,
        quantization=quantization,
        model=model,
        device=device,
        source=source,
        source_format=source_format,
        output=output_path
    )
    t1 = time.time()
    print(f"Conversion time: {t1 - t0:.2f} seconds")
    return

async def main():
    test_convert_weights = False
    if test_convert_weights:
        # make sure to do before:
        # > git lfs install
        # > git clone https://huggingface.co/Qwen/Qwen2.5-1.5B
        _convert_weights("Qwen2.5-1.5B/config.json")
        return

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