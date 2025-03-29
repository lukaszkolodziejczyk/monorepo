import time
from typing import Literal
from vllm import LLM, SamplingParams
from rich import print
from utils import perf, set_vllm_version
from vllm.sampling_params import GuidedDecodingParams
from formatron.schemas.pydantic import ClassSchema

from vllm.model_executor.guided_decoding.xgrammar_decoding import get_local_xgrammar_guided_decoding_logits_processor

t0 = time.time()

batch_size = 100

set_vllm_version()
with perf("llm"):
    llm = LLM(model="facebook/opt-125m")

class CountryPoland(ClassSchema):
    country: Literal["Poland"]

class CountryGermany(ClassSchema):
    country: Literal["Germany"]

schemas = [CountryPoland.model_json_schema(), CountryGermany.model_json_schema()] * (batch_size // 2)

sampling_params_list = []
with perf("logits_processor"):
    for i in range(batch_size):
        guided_decoding_params = GuidedDecodingParams(json=schemas[i])
        logits_processor = get_local_xgrammar_guided_decoding_logits_processor(guided_decoding_params, tokenizer=llm.get_tokenizer(), model_config=llm.llm_engine.get_model_config(), reasoner=None)
        sampling_params_list.append(SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512, logits_processors=[logits_processor]))

prompt = "JSON:"
prompts = [prompt] * batch_size
with perf("model.generate"):
    outputs = llm.generate(prompts, sampling_params_list)
for output in outputs[:3]:
    print(output.outputs[0].text)

print(f"Total time: {time.time() - t0:.2f} seconds")
