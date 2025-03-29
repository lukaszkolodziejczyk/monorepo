import time
from typing import Literal
import xgrammar as xgr
from formatron.schemas.pydantic import ClassSchema
from transformers.generation.logits_process import LogitsProcessorList
from utils import LogitsProcessor, get_model_inputs, get_model_tokenizer_config


from rich import print

t0 = time.time()


batch_size = 2
conversation = [
    {
        "role": "system",
        "content": "You are a structured synthetic data generator. The user provides a JSON that represents a context. Your task is to respond with JSON that completes the context.",
    },
    {"role": "user", "content": "\{\}"},
]
model, tokenizer, config = get_model_tokenizer_config()
model_inputs = get_model_inputs(batch_size, conversation, tokenizer)


class CountryPoland(ClassSchema):
    country: Literal["Poland"]


class CountryGermany(ClassSchema):
    country: Literal["Germany"]


tokenizer_info = xgr.TokenizerInfo.from_huggingface(
    tokenizer, vocab_size=config.vocab_size
)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammars = [
    grammar_compiler.compile_json_schema(CountryPoland),
    grammar_compiler.compile_json_schema(CountryGermany),
]
logit_processor = LogitsProcessor(compiled_grammars)
logit_processor = LogitsProcessorList([logit_processor])

generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=logit_processor
)[:, len(model_inputs["input_ids"][0]) :]
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)

print(f"Total time: {time.time() - t0:.2f} seconds")
