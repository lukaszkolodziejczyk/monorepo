from transformers.generation.logits_process import LogitsProcessorList
import xgrammar as xgr
from icecream import ic

import time

from rich import print

from utils import LogitsProcessor, get_model_inputs, get_model_tokenizer_config, make_random_string_class, perf

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

tokenizer_info = xgr.TokenizerInfo.from_huggingface(
    tokenizer, vocab_size=config.vocab_size
)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
with perf("compiled_grammars"):
    compiled_grammars = [
        grammar_compiler.compile_json_schema(schema) for _ in range(batch_size)
    ]
with perf("logits_processors"):
    logits_processor = LogitsProcessor(compiled_grammars)
    logits_processor = LogitsProcessorList([logits_processor])
with perf("model.generate"):
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=512, logits_processor=logits_processor
    )[:, len(model_inputs["input_ids"][0]) :]
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
ic(generated_texts[:3])

print(f"Total time: {time.time() - t0:.2f} seconds")
