from contextlib import contextmanager
import os
import threading
import time
from typing import List, Literal
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import xgrammar as xgr
import transformers
import psutil
from formatron.schemas.pydantic import ClassSchema
from rich import print

DEVICE = "cpu"


class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self._peak_memory = 0
        self._running = True

    def run(self):
        while self._running:
            mem = self.process.memory_info().rss
            self._peak_memory = max(self._peak_memory, mem)
            time.sleep(self.interval)

    def stop(self):
        self._running = False

    def get_peak_memory_mb(self):
        return self._peak_memory / (1024**2)


@contextmanager
def perf(name):
    monitor = MemoryMonitor()
    monitor.start()
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        monitor.stop()
        monitor.join()
        peak = monitor.get_peak_memory_mb()
        print(
            f"{name.upper():>25} | Peak memory usage: {peak:10.2f} MiB | Elapsed time: {end - start:10.2f} seconds"
        )


def get_model_tokenizer_config():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    return model, tokenizer, config


def get_model_inputs(batch_size, conversation, tokenizer):
    conversations = [conversation] * batch_size
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(DEVICE)
    return model_inputs

class LogitsProcessor(transformers.LogitsProcessor):
    """
    LogitsProcessor for processing logits in transformers' generate() method.

    Example usage
    -------------
        .. code:: python

            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
            # This can be larger than tokenizer.vocab_size due to paddings
            full_vocab_size = config.vocab_size
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

            grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
            xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
            model.generate(prompt, logits_processor=[xgr_logits_processor])

        For an end-to-end example, see folder `examples/hf_transformers/`.

    Notes
    -----
        - Note that this LogitsProcessor can only be used once. For each `generate()` call,
            instantiate a new one.
        - Note that this implementation may contain extra overhead.
    """

    def __init__(self, compiled_grammar: xgr.CompiledGrammar | List[xgr.CompiledGrammar]):
        """Initialize the LogitsProcessor.

        Parameters
        ----------
        compiled_grammar : xgr.CompiledGrammar | List[xgr.CompiledGrammar]
            One or more grammars compiled according to the given grammar and the model's tokenizer_info.
        """
        self.matchers: List[xgr.GrammarMatcher] = []
        self.compiled_grammars = compiled_grammar if isinstance(compiled_grammar, list) else [compiled_grammar]
        self.full_vocab_size = self.compiled_grammars[0].tokenizer_info.vocab_size
        self.token_bitmask = None
        self.prefilled = False
        self.batch_size = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Accept token sampled in the last iteration, fill in bitmask, and apply bitmask to logits.

        Returns:
            scores: Logits modified with bitmask.
        """
        # Lazily initialize GrammarMatchers and bitmask
        if len(self.matchers) == 0:
            self.batch_size = input_ids.shape[0]
            self.compiled_grammars = self.compiled_grammars if len(self.compiled_grammars) > 1 else self.compiled_grammars * self.batch_size
            assert len(self.compiled_grammars) == self.batch_size, "The number of compiled grammars must be equal to the batch size."
            self.matchers = [
                xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)
            ]
            self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)

        if input_ids.shape[0] != self.batch_size:
            raise RuntimeError(
                "Expect input_ids.shape[0] to be LogitsProcessor.batch_size."
                + f"Got {input_ids.shape[0]} for the former, and {self.batch_size} for the latter."
            )

        if not self.prefilled:
            # Have not sampled a token yet
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = input_ids[i][-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        # We only support masking logits on CUDA or CPU
        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        if device_type != "cuda":
            scores = scores.to(device_type)

        # NOTE: Cannot reset here because __call__ is not invoked when stop token
        # is sampled. This is why each `generate()` call needs to instantiate an
        # LogitsProcessor

        return scores
    
def make_random_string_class(n_columns, cardinality):
    literals = [tuple([str(uuid.uuid4()) for _ in range(cardinality)]) for _ in range(n_columns)]
    class RandomString(ClassSchema):
        __annotations__ = {
            f"random_string_{i+1}": Literal[literals[i]] for i in range(n_columns)
        }
    return RandomString