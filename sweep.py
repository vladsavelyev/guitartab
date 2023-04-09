""" 
Explore hyperparameter space with wandb sweeps
"""
import datasets
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, GPT2LMHeadModel,
)
import wandb
from guitartab import load_model, load_tokenizer, MODEL, DATASET, BASE_MODEL

tokenizer = load_tokenizer()

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

# sweep_config = {
#     "method": "bayes",
#     "metric": {"name": "loss", "goal": "minimize"},
#     "parameters": {
#         "learning_rate": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
#         "per_device_train_batch_size": {"values": [4, 8]},
#     },
# }

dataset = datasets.load_dataset(DATASET)
model = load_model()
sweepset = dataset["train"].train_test_split(test_size=110)["test"]
sweepset = sweepset.train_test_split(test_size=10)
sweepset = sweepset.map(
    lambda b: tokenizer(
        b['text'],
        max_length=model.config.n_ctx,
        truncation=True,  # because of the option below, it will chunk
        return_overflowing_tokens=True,  # ...tokens, not truncate
        # we want the chunks to overlap by 20%
        stride=int(model.config.n_ctx * 0.1),
        padding=True,
    ),
    batched=True,
    remove_columns=dataset["train"].column_names,
).select_columns("input_ids")
len(sweepset['train']), len(sweepset['test'])

sweep_id = wandb.sweep(sweep_config, project=f"{MODEL.strip('/')[1]}_sweeps")

sweep_train_set = dataset["train"].train_test_split(test_size=640)["test"]


def hp_search(config=None):
    with wandb.init(config=config):
        # set sweep configuration
        config = wandb.config

        # set training arguments
        args = TrainingArguments(
            output_dir="wandb-sweeps",
            report_to=["wandb"],
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
            args=args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            ),
            train_dataset=sweepset['train'],
            eval_dataset=sweepset['test'],
        )
        trainer.train()


wandb.agent(sweep_id, hp_search)
