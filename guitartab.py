"""
Importable module for scripts build.py, train.py, sweep.py, generate.py
"""

import os

import coloredlogs
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
)
from transformers import (
    AutoTokenizer,
    GenerationConfig,
)

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()

TOKENIZER = "vldsavelyev/guitar_tab_gpt2"
MODEL = "vldsavelyev/guitar_tab_gpt2"
DATASET = "vldsavelyev/guitar_tab"
BASE_MODEL = "gpt2"
TOKEN = os.getenv("HUB_TOKEN")


def load_model():
    return AutoModelForCausalLM.from_pretrained(MODEL)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def load_generation_config():
    return GenerationConfig.from_pretrained(MODEL, "generation_config.json")


def prep_dataset(instrument_class=None):
    """
    Load dataset and prepare for training:
    - optionally filter by instrument class
    - split into train and test
    - wrap examples with BOS and EOS tokens
    - tokenize with chunking
    """
    dataset = datasets.load_dataset(DATASET)
    model = load_model()
    tokenizer = load_tokenizer()

    if instrument_class:
        INSTRUMENT_NUMBERS = {
            "guitar": range(24, 31 + 1),
            "bass": range(32, 39 + 1),
        }
        INSTRUMENT_STRING_NUMBERS = {
            "guitar": 6,
            "bass": 4,
        }
        dataset = dataset.filter(
            lambda x: (
                (x.get("instrument_number") or -1)
                in INSTRUMENT_NUMBERS[instrument_class]
                and len((x.get("tuning") or "[]").split(","))
                == INSTRUMENT_STRING_NUMBERS[instrument_class]
            )
        )

    dataset = dataset["train"].train_test_split(test_size=10)

    # Wrap examples with BOS and EOS tokens (tokenizer wouldn't
    # do that even if add_special_tokens is True, see
    # https://github.com/huggingface/transformers/issues/3311)
    dataset = dataset.map(
        lambda b: {
            "text": [
                f"{tokenizer.bos_token}{x}{tokenizer.eos_token}" for x in b["text"]
            ]
        },
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return dataset.map(
        lambda b: tokenizer(
            b["text"],
            max_length=model.config.n_ctx,
            truncation=True,  # because of the option below, it will chunk
            return_overflowing_tokens=True,  # ...tokens, not truncate
            # we want the chunks to overlap by 20%
            stride=int(model.config.n_ctx * 0.1),
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    ).select_columns("input_ids")
