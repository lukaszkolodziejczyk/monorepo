import time
from vllm import LLM, SamplingParams
from rich import print
from utils import make_random_string_class, perf, set_vllm_version
from vllm.sampling_params import GuidedDecodingParams
from vllm.model_executor.guided_decoding.xgrammar_decoding import (
    get_local_xgrammar_guided_decoding_logits_processor,
)

t0 = time.time()

batch_size = 100
cardinality = 200
n_columns = 2

set_vllm_version()
with perf("llm"):
    llm = LLM(model="facebook/opt-125m")
schema = make_random_string_class(n_columns, cardinality).model_json_schema()

guided_decoding_params = GuidedDecodingParams(json=schema)
with perf("logits_processor"):
    logits_processor = get_local_xgrammar_guided_decoding_logits_processor(
        guided_decoding_params,
        tokenizer=llm.get_tokenizer(),
        model_config=llm.llm_engine.get_model_config(),
        reasoner=None,
    )

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=512, logits_processors=[logits_processor]
)

prompt = "JSON:"
prompts = [prompt] * batch_size
with perf("model.generate"):
    outputs = llm.generate(prompts, sampling_params)
for output in outputs[:3]:
    print(output.outputs[0].text)

print(f"Total time: {time.time() - t0:.2f} seconds")
