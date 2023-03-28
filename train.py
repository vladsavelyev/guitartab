# %% MODEL AND TOKENIZER

import os
import math
from pathlib import Path

import datasets, transformers
from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    pipeline,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import coloredlogs

from gp2tex import alphatex_to_song
import guitarpro as gp


coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()

from_scratch = False
dry_run = False
push_to_hub = True

model_name = "vldsavelyev/guitar_tab_gpt2"
dataset_name = "vldsavelyev/guitar_tab"

token = os.getenv("HUB_TOKEN")
if push_to_hub and not token:
    raise ValueError(
        "push_to_hub is set to True, but HUB_TOKEN environment variable is not set"
    )


print(f"Loading dataset from remote repo {dataset_name}")
dataset = load_dataset(dataset_name)

if from_scratch:
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = gpt2_tokenizer.train_new_from_iterator(
        (e["text"] for e in dataset["train"]),
        vocab_size=500,
        new_special_tokens=["<|pad|>"],
    )
    tokenizer.push_to_hub(model_name)

    print(f"Initializing model {model_name}")
    config = AutoConfig.from_pretrained(
        "gpt2",
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
    model.push_to_hub(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)


# %% PREP DATASET

dataset = dataset['train'].train_test_split(test_size=10, seed=42)

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


dataset = dataset.map(
    _tokenize, batched=True, remove_columns=dataset.column_names["train"]
)


# %% SETUP TRAINER

repos_dir = Path(os.getenv("HUB_REPOS") or "").resolve()
save_dir = repos_dir / "models" / model_name

if transformers.utils.is_torch_cuda_available():
    # Optimal configuration for T4 Colab GPU with 15G memory
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        overwrite_output_dir=True,
        push_to_hub=push_to_hub and os.getenv("HUB_TOKEN") is not None,
        hub_model_id=model_name,
        hub_token=os.getenv("HUB_TOKEN"),
        report_to=["all"],
        run_name="guitart-gpt2-96-8-8",
        skip_memory_metrics=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        eval_accumulation_steps=20,
        logging_steps=10,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        fp16=True,
        ignore_data_skip=True,
    )
else:
    # For debugging on a CPU.
    training_args = TrainingArguments(
        output_dir=save_dir,
        report_to=[],
        evaluation_strategy="steps",
        eval_steps=1,
        logging_steps=1,
        logging_first_step=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )

# %% GENERATION

# Model's config has `bos_token_id=eos_token_id=50256`, even though the
# tokenizer has bos_token_id=eos_token_id=50257 ("<|endoftext|>") (for 50256,
# the tokenizer has just a standard Russian word). That's because the tokenizer
# was completely rebuilt during fine-tuning. The original GPT2 tokenizer didn't
# have other special tokens apart from '<|endoftext|>' (50256), but the rebuilt
# first has first 5 tokens corresponding to default BPE special tokens
# "<pad>", "<s>", "</s>", "<unk>", "<mask>", which are not used at all, and then
# a special token "<|endoftext|> (50257) added in the end, which was actually used.
# That looks like a bug on their side: when they used a pre-trained GPT2, they
# should have preserved the tokenizer. That resulted in the side-effect that only
# Russian texts are produced when prompting the model with "<|endoftext|>", whereas
# when prompting with any English letters, you can get English texts. Perhaps
# it was a desired side-effect for Sberbank, but just looks inefficient.
#
# So for our Murakami chapters, we should make sure we we wrap them chapters
# with 50257.
#
# Generation pipeline reads `model.generation_config` for default values, and
# we pass `eos_token_id` and `pad_token_id` to override those. Annoyingly, it also
# prints the broken `model.generation_config` to stdout, which we can't avoid.
# Modifying `model.generation_config` also doesn't work, as it gets re-set
# by `pipeline` on every run. So we'll have to deal with misleading messages.
#
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    device=training_args.device,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1,
    max_length=200,
)


out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

class MyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if metrics := kwargs.get("metrics"):
            loss = metrics["eval_loss"]
            print(f"Eval loss: {loss:.4f}")
            print(f"Perplexity: {math.exp(loss):.2f}")
        if state.best_metric:
            print(f"Best loss so far: {state.best_metric:.4f}")

        for result in generator([tokenizer.bos_token])[0]:
            tex = result["generated_text"]
            print(tex)
            tex = tex.replace(tokenizer.eos_token, "")
            try:
                song = alphatex_to_song(tex)
            except:
                print('Could not parse the tex to GP')
            else:
                try:
                    gp.write(song, str(out_dir / f"{state.global_step}.gp"))
                except:
                    print('Could not write the GP file')


trainer = Trainer(
    model=model,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    callbacks=[MyCallback],
    args=training_args,
)

# %% TRAIN
if not dry_run:
    trainer.train(resume_from_checkpoint=get_last_checkpoint(save_dir))
    if push_to_hub:
        trainer.save_model()  # also calls push_to_hub
