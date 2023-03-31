from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from datasets import load_dataset


MODEL = "vldsavelyev/guitar_tab_gpt2"
DATASET = "vldsavelyev/guitar_tab"
BASE_MODEL = "gpt2"
FROM_SCRATCH = False
STREAMING = True


print(f"Loading dataset from remote repo {DATASET}")
dataset = load_dataset(DATASET, streaming=STREAMING)

if FROM_SCRATCH:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
else:
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer = base_tokenizer.train_new_from_iterator(
        (e["tex"] for e in dataset["train"]),
        vocab_size=500,
        new_special_tokens=["<|pad|>"],
    )
    tokenizer.push_to_hub(MODEL)

dataset = dataset["train"].train_test_split(test_size=10)

# Wrap novel chapters with BOS and EOS tokens (tokenizer wouldn't
# do that even if add_special_tokens is True, see
# https://github.com/huggingface/transformers/issues/3311)
dataset = dataset.map(
    lambda x: {"tex": f'{tokenizer.bos_token}{x["tex"]}{tokenizer.eos_token}'}
)

if not FROM_SCRATCH:
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    generation_config = GenerationConfig.from_pretrained(
        MODEL, "generation_config.json"
    )
else:
    print(f"Initializing model {MODEL}")
    config = AutoConfig.from_pretrained(
        BASE_MODEL,
        vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_embd=96,  # smaller vocab -> smaller embedding
        n_layer=8,
        n_head=8,
    )
    model = GPT2LMHeadModel(config)
    print(f"Model parameters: {model.num_parameters():,}")
    model.push_to_hub(MODEL)

    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1,
        max_length=200,
    )
    generation_config.push_to_hub(MODEL, "generation_config.json")


def _tokenize(batch: dict[str, list]):
    ids = tokenizer(
        batch["tex"],
        max_length=model.config.n_ctx,
        truncation=True,  # because of the option below, it will chunk
        return_overflowing_tokens=True,  # ...tokens, not trancate
        # we want the chunks to overlap by 20%
        stride=int(model.config.n_ctx * 0.2),
    )["input_ids"]
    return {"input_ids": ids}


dataset = dataset.map(_tokenize, batched=True)
