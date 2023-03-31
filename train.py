# %% IMPORTS

import os
import math
from pathlib import Path

import datasets, transformers, evaluate
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
    EvalPrediction,
    pipeline,
    trainer_utils,
)
import coloredlogs
import guitarpro as gp
from gp2tex import alphatex_to_song

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()

from_scratch = False
dry_run = False
push_to_hub = True


token = os.getenv("HUB_TOKEN")
if push_to_hub and not token:
    push_to_hub = False
    print("Cannot push to hub without HUB_TOKEN")

MODEL = "vldsavelyev/guitar_tab_gpt2"
DATASET = "vldsavelyev/guitar_tab"
BASE_MODEL = "gpt2"
FROM_SCRATCH = False

# %% DATASET

print(f"Loading dataset from remote repo {DATASET}")
dataset = datasets.load_dataset(DATASET)

# %% TOKENIZER

if FROM_SCRATCH:
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_token = "<|pad|>"
    tokenizer = base_tokenizer.train_new_from_iterator(
        (e["tex"] for e in dataset["train"]),
        vocab_size=500,
        new_special_tokens=[pad_token],
    )
    tokenizer.pad_token = "<|pad|>"
    tokenizer.push_to_hub(MODEL)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

dataset = dataset["train"].train_test_split(test_size=10)

# Wrap novel chapters with BOS and EOS tokens (tokenizer wouldn't
# do that even if add_special_tokens is True, see
# https://github.com/huggingface/transformers/issues/3311)
dataset = dataset.map(
    lambda b: {
        "tex": [f"{tokenizer.bos_token}{x}{tokenizer.eos_token}" for x in b["tex"]]
    },
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# %% MODEL
if FROM_SCRATCH:
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
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    generation_config = GenerationConfig.from_pretrained(
        MODEL, "generation_config.json"
    )


# %% PREP DATASET
dataset = dataset.map(
    lambda b: tokenizer(
        b['tex'],
        max_length=model.config.n_ctx,
        truncation=True,  # because of the option below, it will chunk
        return_overflowing_tokens=True,  # ...tokens, not trancate
        # we want the chunks to overlap by 20%
        stride=int(model.config.n_ctx * 0.2),
    ),
    batched=True,
    remove_columns=dataset['train'].column_names
).select_columns('input_ids')

# %% SETUP TRAINER
repos_dir = Path(os.getenv("HUB_REPOS") or "").resolve()
save_dir = repos_dir / "models" / MODEL

if transformers.utils.is_torch_cuda_available():
    # Optimal configuration for T4 Colab GPU with 15G memory
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        overwrite_output_dir=True,
        push_to_hub=push_to_hub and os.getenv("HUB_TOKEN") is not None,
        hub_model_id=MODEL,
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

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=training_args.device,
    generation_config=generation_config,
)

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)


def compute_metrics(eval_preds: EvalPrediction):
    perplexity = evaluate.load("perplexity", module_type="metric")
    pred = eval_preds.predictions.argmax(axis=-1)
    return perplexity.compute(predictions=pred, model_id="gpt2")


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
    compute_metrics=compute_metrics,
)


# %% TRAIN
if not dry_run:
    trainer.evaluate()  # to early test evaluation works
    trainer.train(resume_from_checkpoint=trainer_utils.get_last_checkpoint(save_dir))
    if push_to_hub:
        trainer.save_model()  # also calls push_to_hub
