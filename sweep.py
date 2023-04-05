""" 
Explore hyperparameter space with wandb sweeps
"""

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from model import prep_dataset, load_model, load_tokenizer, MODEL


tokenizer = load_tokenizer()
dataset = prep_dataset()

sweep_config = {
    "method": "grid",
    "parameters": {
        "optim": {"values": ["adamw_torch", "adafactor"]},
        "batch_size": {
            "values": [1, 4, 8, 32],
        },
        "gradient_checkpointing": {"values": [True, False]},
        "gradient_accumulation_steps": {"values": [1, 8, 64]},
    },
    "metric": {
        "name": "eval/loss",
        "goal": "minimize",
    },
}

sweep_id = wandb.sweep(sweep_config, project=f"{MODEL}_sweeps")

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
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=True,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model_init=load_model,
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
