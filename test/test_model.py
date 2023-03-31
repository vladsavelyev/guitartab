from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

import os

print("CWD", os.getcwd())

from datasets import Dataset
from guitartab import model, tokenizer, dataset, generation_config


# def test_overfit(tmp_path):
tmp_path = "tmp"
"""
When running repeatedly on the same batch, model is expected to
overfit to 100% accuracy. if that doesn't happen, it means the model
is not learning anything
"""
example = {
    "tex": """
\\title "My Song"
\\tempo 90
.
\\track
\\instrument 42
1.1 2.1 3.1 4.1 |
\\track
\\tuning A1 D2 A2 D3 G3 B3 E4
4.1 3.1 2.1 1.1 |
"""
}
dataset = Dataset.from_list([example])

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
for _ in range(20):
    outputs = trainer.model(**batch)
    loss = outputs.loss
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

generated_tex = model.generate([model.eos_token], generation_config=generation_config)
# generated_tex = generator([tokenizer.bos_token])[0][0]["generated_text"]
training_tex = tokenizer.decode(batch["input_ids"][0].tolist())
print("Training tex:")
print(training_tex)
print()
print("Generated:")
print(generated_tex)


# if __name__ == "__main__":
#     import os
#     test_overfit("tmp")
