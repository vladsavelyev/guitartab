from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from datasets import Dataset

import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level="info")


MODEL = "vldsavelyev/guitar_tab_gpt2"


def test_overfit(tmp_path):
    """
    When running repeatedly on the same batch, model is expected to
    overfit to 100% accuracy. if that doesn't happen, it means the model
    is not learning anything
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    example = """\
\\title "Song"
.
1.1 2.1 3.1 4.1 |
"""
    dataset = Dataset.from_list([{"tex": example}])
    dataset = dataset.map(
        lambda b: {
            "tex": [f"{tokenizer.bos_token}{x}{tokenizer.eos_token}" for x in b["tex"]]
        },
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(
        lambda b: tokenizer(b["tex"], max_length=model.config.n_ctx, truncation=True),
        batched=True,
        remove_columns=dataset.column_names,
    ).select_columns("input_ids")

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        args=TrainingArguments(
            output_dir=tmp_path,
            report_to=[],
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            optim="adamw_torch",
            lr_scheduler_type="constant",
            num_train_epochs=len(example),
            learning_rate=1e-3,
        ),
    )
    trainer.train()

    training_tokens = dataset[0]["input_ids"]
    training_tex = tokenizer.decode(training_tokens)
    generated_tex = tokenizer.decode(
        model.generate(
            max_length=len(training_tokens),
            do_sample=False,
        )[0].tolist()
    )

    print("Training tex:")
    print(training_tex)
    print("")
    print("Generated:")
    print(generated_tex)
    assert training_tex == generated_tex


test_overfit("tmp")
