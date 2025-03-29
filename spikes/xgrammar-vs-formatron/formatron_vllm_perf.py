import time
from vllm import LLM, SamplingParams
from rich import print
from formatron.integrations.vllm import create_formatters_logits_processor
from utils import make_random_string_class, perf, set_vllm_version
from formatron.formatter import FormatterBuilder

t0 = time.time()

batch_size = 100
cardinality = 200
n_columns = 2

set_vllm_version()
with perf("llm"):
    llm = LLM(model="facebook/opt-125m")
schema = make_random_string_class(n_columns, cardinality)

with perf("formatter_builders"):
    formatter_builder = FormatterBuilder()
    formatter_builder.append_line(
        f"{formatter_builder.json(schema, capture_name=None)}"
    )
    formatter_builders = [formatter_builder] * batch_size

with perf("logits_processor"):
    logits_processor = create_formatters_logits_processor(llm, formatter_builders)

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
