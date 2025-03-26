import time
from rich import print
from formatron.formatter import FormatterBuilder
from formatron.integrations.transformers import create_formatter_logits_processor_list
from icecream import ic

from utils import get_model_inputs, get_model_tokenizer_config, make_random_string_class, perf

t0 = time.time()

batch_size = 100
cardinality = 200
n_columns = 2
conversation = [
    {
        "role": "system",
        "content": "You are a structured synthetic data generator. The user provides a JSON that represents a context. Your task is to respond with JSON that completes the context.",
    },
    {"role": "user", "content": "\{\}"},
]

model, tokenizer, config = get_model_tokenizer_config()
model_inputs = get_model_inputs(batch_size, conversation, tokenizer)
schema = make_random_string_class(n_columns, cardinality)


with perf("formatter_builders"):
    formatter_builder = FormatterBuilder()
    formatter_builder.append_line(
        f"{formatter_builder.json(schema, capture_name=None)}"
    )
    formatter_builders = [formatter_builder] * batch_size
with perf("logits_processor"):
    logits_processor = create_formatter_logits_processor_list(
        tokenizer, formatter_builders
    )
with perf("model.generate"):
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=512, logits_processor=logits_processor
    )[:, len(model_inputs["input_ids"][0]) :]
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
ic(generated_texts[:3])

print(f"Total time: {time.time() - t0:.2f} seconds")
