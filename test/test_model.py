from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from model import get_model, get_tokenizer, get_dataset, get_generator


def test_overfit(tmp_path):
    """
    When running repeatedly on the same batch, model is expected to
    overfit to 100% accuracy. if that doesn't happen, it means the model
    is not learning anything
    """
    model = get_model()
    tokenizer = get_tokenizer()
    dataset = get_dataset(prep=True)

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=TrainingArguments(
            output_dir=tmp_path,
            report_to=[],
            max_steps=20,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        ),
    )
    # taking single batch
    for batch in trainer.get_train_dataloader():
        break

    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
    trainer.create_optimizer()
    for _ in range(20):
        outputs = trainer.model(**batch)
        loss = outputs.loss
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    generator = get_generator(trainer.args.device)
    generated_tex = generator([tokenizer.bos_token])[0][0]["generated_text"]
    training_tex = tokenizer.decode(batch["input_ids"][0].tolist())
    print("Training tex:")
    print(training_tex)
    print()
    print("Generated:")
    print(generated_tex)
