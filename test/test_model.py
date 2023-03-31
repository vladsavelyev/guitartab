from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

from datasets import Dataset


MODEL = "vldsavelyev/guitar_tab_gpt2"


def test_overfit(tmp_path):
    """
    When running repeatedly on the same batch, model is expected to
    overfit to 100% accuracy. if that doesn't happen, it means the model
    is not learning anything
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    example = {
        "tex": """
    \\title "Song"
    .
    1.1 2.1 3.1 4.1 |
    """
    }
    dataset = Dataset.from_list([example])
    dataset = dataset.map(
        lambda b: {
            "tex": [f"{tokenizer.bos_token}{x}{tokenizer.eos_token}" for x in b["tex"]]
        },
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(
        lambda b: tokenizer(
            b["tex"],
            max_length=model.config.n_ctx,
            truncation=True,
            return_overflowing_tokens=True,
            stride=int(model.config.n_ctx * 0.2),
        ),
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
            max_steps=20,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        ),
    )
    # taking single batch
    batch = next(iter(trainer.get_train_dataloader()))
    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
    trainer.create_optimizer()
    for _ in range(40):
        outputs = trainer.model(**batch)
        loss = outputs.loss
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    generation_config = GenerationConfig.from_pretrained(
        MODEL, "generation_config.json"
    )
    training_tokens = batch["input_ids"][0]
    training_tex = tokenizer.decode(training_tokens.tolist())
    generated_tokens = model.generate(
        generation_config=generation_config,
        max_length=len(training_tokens),
    )[0]
    generated_tex = tokenizer.decode(generated_tokens.tolist())
    print("Training tex:")
    print(training_tex)
    print()
    print("Generated:")
    print(generated_tex)
    assert training_tex == generated_tex
