# %% IMPORTS
import math
from pathlib import Path

import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    trainer_utils,
)
from model import load_model, MODEL, load_tokenizer, prep_dataset, TOKEN
from generate import generate_song
import logging

DRY_RUN = False

if TOKEN:
    logging.warning("Cannot push to hub without HUB_TOKEN")

# %% PREP DATASET
dataset = prep_dataset()

# %% SET UP TRAINER
save_dir = Path(".saves") / MODEL
save_dir.mkdir(parents=True, exist_ok=True)

if transformers.utils.is_torch_cuda_available():
    # Optimal configuration for T4 Colab GPU with 15G memory
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        overwrite_output_dir=True,
        push_to_hub=TOKEN is not None,
        hub_model_id=MODEL,
        hub_token=TOKEN,
        report_to=["wandb"],
        run_name=MODEL.split("/")[-1],
        skip_memory_metrics=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        eval_accumulation_steps=5,
        logging_steps=50,
        logging_first_step=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=200,
        # Most optimal configuration per sweeps
        # https://wandb.ai/vsavelyev/guitartab-sweeps/sweeps/bx7jbzna?workspace=user-vsavelyev
        # https://wandb.ai/vsavelyev/guitartab-sweeps/sweeps/meecv8s2?workspace=user-vsavelyev
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # gradient_checkpointing=True,  # 10x < mem, 40% > runtime, = loss
        # gradient_accumulation_steps=8,  # < runtime, = mem, < loss
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


testing_dir = Path(".testing")
testing_dir.mkdir(exist_ok=True)


class MyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if metrics := kwargs.get("metrics"):
            model, tokenizer = kwargs["model"], kwargs["tokenizer"]
            loss = metrics["eval_loss"]
            print(f"Eval loss: {loss:.4f}")
            print(f"Perplexity: {math.exp(loss):.2f}")
            generate_song(
                out_dir=testing_dir,
                title=f"Step {state.global_step}, loss {loss:.4f}",
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                max_length=1000,
                num_return_sequences=1,
            )
        if state.best_metric:
            print(f"Best loss so far: {state.best_metric:.4f}")


trainer = Trainer(
    model_init=load_model,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=load_tokenizer(),
        mlm=False,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    callbacks=[MyCallback],
    args=training_args,
)

trainer.evaluate()  # to early test if something crashes

# %% TRAIN
if not DRY_RUN:
    trainer.train(resume_from_checkpoint=trainer_utils.get_last_checkpoint(save_dir))
    if TOKEN:
        trainer.save_model()  # also calls push_to_hub
