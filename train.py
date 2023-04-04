# %% IMPORTS
import os
import math
from pathlib import Path

import datasets, transformers
from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    pipeline,
    trainer_utils,
)
import wandb
import coloredlogs
import guitarpro as gp
from gp_to_tex import alphatex_to_song

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()

TOKENIZER = "vldsavelyev/guitar_tab_gpt2"
MODEL = "vldsavelyev/guitar_tab_gpt2"
DATASET = "vldsavelyev/guitar_tab"
BASE_MODEL = "gpt2"
FROM_SCRATCH = False
DRY_RUN = False
PUSH_TO_HUB = True

INSTRUMENT_CLASS = "bass"
if INSTRUMENT_CLASS:
    MODEL += f"_{INSTRUMENT_CLASS}"

token = os.getenv("HUB_TOKEN")
if PUSH_TO_HUB and not token:
    PUSH_TO_HUB = False
    print("Cannot push to hub without HUB_TOKEN")

# %% TOKENIZER
if FROM_SCRATCH:
    dataset = datasets.load_dataset(DATASET)
    n_examples = 10_000
    examples = dataset["train"].shuffle(seed=42)[:n_examples]
    batch_size = 1000
    batches = (
        examples["text"][i : i + batch_size] for i in range(0, n_examples, batch_size)
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
    tokenizer.push_to_hub(MODEL, use_auth_token=token)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

# %% MODEL
if FROM_SCRATCH:
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
    model.push_to_hub(MODEL, use_auth_token=token)

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
    generation_config.push_to_hub(MODEL, "generation_config.json", use_auth_token=token)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    generation_config = GenerationConfig.from_pretrained(
        MODEL, "generation_config.json"
    )

# %% PREP DATASET
if FROM_SCRATCH:
    dataset = datasets.load_dataset(DATASET)

    if INSTRUMENT_CLASS:
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
                in INSTRUMENT_NUMBERS[INSTRUMENT_CLASS]
                and len((x.get("tuning") or "[]").split(","))
                == INSTRUMENT_STRING_NUMBERS[INSTRUMENT_CLASS]
            )
        )

    dataset = dataset["train"].train_test_split(test_size=10)

    # Wrap novel chapters with BOS and EOS tokens (tokenizer wouldn't
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

    dataset = dataset.map(
        lambda b: tokenizer(
            b["text"],
            max_length=model.config.n_ctx,
            truncation=True,  # because of the option below, it will chunk
            return_overflowing_tokens=True,  # ...tokens, not trancate
            # we want the chunks to overlap by 20%
            stride=int(model.config.n_ctx * 0.1),
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    ).select_columns("input_ids")

    if cache_dir := os.getenv("HF_HOME"):
        ds_dir = Path(cache_dir) / "dataset" / DATASET
        dataset.save_to_disk(str(ds_dir))

else:
    if cache_dir := os.getenv("HF_HOME"):
        ds_dir = Path(cache_dir) / "dataset" / DATASET
    dataset = datasets.load_from_disk(str(ds_dir))


# %% EXPLORE HYPERPARAMETERS
def explore_hyperparameters():
    sweep_config = {
        "method": "grid",
        "parameters": {
            "optim": {"values": ["adamw_torch", "adafactor"]},
            "batch_size": {
                "values": [8, 32],
            },
            "gradient_checkpointing": {"values": [True, False]},
        },
        "metric": {
            "name": "train/train_samples_per_second",
            "goal": "maximize",
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="guitartab-sweeps")

    sweep_train_set = dataset["train"].train_test_split(test_size=640)["test"]

    def hp_search(config=None):
        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config

            # set training arguments
            targs = TrainingArguments(
                output_dir="wandb-sweeps",
                report_to="wandb",
                skip_memory_metrics=False,
                eval_accumulation_steps=20,
                optim=config.optim,
                num_train_epochs=1,
                lr_scheduler_type="linear",
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                gradient_checkpointing=config.gradient_checkpointing,
                fp16=True,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                remove_unused_columns=False,
            )

            trainer = Trainer(
                model_init=lambda: AutoModelForCausalLM.from_pretrained(MODEL),
                args=targs,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                ),
                train_dataset=sweep_train_set,
                eval_dataset=dataset["test"],
            )
            trainer.train()

    wandb.agent(sweep_id, hp_search)


# %% SETUP TRAINER
repos_dir = Path(os.getenv("HUB_REPOS") or "").resolve()
save_dir = repos_dir / "models" / MODEL

if transformers.utils.is_torch_cuda_available():
    # Optimal configuration for T4 Colab GPU with 15G memory
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        overwrite_output_dir=True,
        push_to_hub=PUSH_TO_HUB and os.getenv("HUB_TOKEN") is not None,
        hub_model_id=MODEL,
        hub_token=os.getenv("HUB_TOKEN"),
        report_to="wandb",
        run_name="guitartab-gpt2",
        skip_memory_metrics=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        eval_accumulation_steps=5,
        logging_steps=5,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=20,
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        gradient_checkpointing=True,  # must have! 10x < mem, 40% < speed, = loss
        gradient_accumulation_steps=8,  # > speed, = mem, < loss
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
        optim="adamw_torch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=training_args.device,
    generation_config=generation_config,
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
                print("Could not parse the tex to GP")
            else:
                try:
                    gp.write(song, str(out_dir / f"{state.global_step}.gp"))
                except:
                    print("Could not write the GP file")


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
if not DRY_RUN:
    trainer.evaluate()  # to early test evaluation works
    trainer.train(resume_from_checkpoint=trainer_utils.get_last_checkpoint(save_dir))
    if PUSH_TO_HUB:
        trainer.save_model()  # also calls push_to_hub
