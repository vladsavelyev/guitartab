"""
Start from scratch: train tokenizer, create model, push to hub.
"""

import datasets
from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
)

from guitartab import TOKEN, MODEL, TOKENIZER, DATASET, BASE_MODEL

if not TOKEN:
    raise ValueError("Cannot push to hub without HUB_TOKEN")

# %% TOKENIZER
dataset = datasets.load_dataset(DATASET)
n_examples = 10_000
examples = dataset["train"].shuffle(seed=42)[:n_examples]
batch_size = 1000
batches = (
    examples["text"][i: i + batch_size] for i in range(0, n_examples, batch_size)
)
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# training on small examples would fail without a padding token
# can't use eos token because data collator
# set label to -100 for pad tokens, and eos would
# be ignored during training
pad_token = "<|pad|>"
tokenizer = base_tokenizer.train_new_from_iterator(
    batches,
    vocab_size=500,
    new_special_tokens=[pad_token],
)
tokenizer.pad_token = "<|pad|>"
tokenizer.push_to_hub(TOKENIZER, use_auth_token=TOKEN)

# %% MODEL
print(f"Initializing model {MODEL}")
config = AutoConfig.from_pretrained(
    BASE_MODEL,
    vocab_size=len(tokenizer),
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    n_embd=96,  # smaller vocab -> smaller embedding
    n_layer=10,
    n_head=12,
)
model = GPT2LMHeadModel(config)
print(f"Model parameters: {model.num_parameters():,}")
model.push_to_hub(MODEL, use_auth_token=TOKEN)

generation_config = GenerationConfig(
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1,
    max_length=200,
)
generation_config.push_to_hub(MODEL, use_auth_token=TOKEN)
