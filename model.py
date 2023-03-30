from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)
from datasets import load_dataset


MODEL = "vldsavelyev/guitar_tab_gpt2"
DATASET = "vldsavelyev/guitar_tab"
BASE_MODEL = "gpt2"


def get_dataset(streaming=False, prep=False):
    print(f"Loading dataset from remote repo {DATASET}")
    dataset = load_dataset(DATASET, streaming=streaming)
    if not prep:
        return dataset

    tokenizer = get_tokenizer()
    model = get_model()

    dataset["test"] = dataset["train"].take(10)

    # Wrap novel chapters with BOS and EOS tokens (tokenizer doesn't do that even
    # if add_special_tokens is True, see https://github.com/huggingface/transformers/issues/3311)
    dataset = dataset.map(
        lambda x: {"text": f'{tokenizer.bos_token}{x["text"]}{tokenizer.eos_token}'}
    )

    def _tokenize(batch: dict[str, list]):
        ids = tokenizer(
            batch["text"],
            max_length=model.config.n_ctx,
            truncation=True,  # because of the option below, it will chunk
            return_overflowing_tokens=True,  # ...tokens, not trancate
            # we want the chunks to overlap by 20%
            stride=int(model.config.n_ctx * 0.2),
        )["input_ids"]
        return {"input_ids": ids}
    
    from datasets import IterableDataset

    dataset = dataset.map(
        _tokenize, batched=True, remove_columns=dataset['train'].column_names
    )
    return dataset


def get_tokenizer(from_scratch=False):
    if not from_scratch:
        return AutoTokenizer.from_pretrained(MODEL)
    dataset = get_dataset()
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer = base_tokenizer.train_new_from_iterator(
        (e["text"] for e in dataset["train"]),
        vocab_size=500,
        new_special_tokens=["<|pad|>"],
    )
    tokenizer.push_to_hub(MODEL)
    return tokenizer


def get_model(from_scratch=False):
    if not from_scratch:
        return AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = get_tokenizer(from_scratch)
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
    return model


def get_generator(device):
    tokenizer = get_tokenizer()
    model = get_model()

    try:
        gconf = GenerationConfig.from_pretrained(MODEL, "generation_config.json")
    except:
        gconf = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            max_length=200,
        )
        gconf.push_to_hub(MODEL, "generation_config.json")

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=gconf,
    )
