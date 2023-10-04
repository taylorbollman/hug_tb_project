# import pdb; pdb.set_trace()
# from icecream import ic
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import huggingface_hub
from datasets import load_dataset


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test-trainer")
raw_dataset = load_dataset("glue", "mrpc")


tokenized_dataset = raw_dataset.map(
    lambda examples: tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True),
    batched=True,
    num_proc=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)
trainer.train()



# trainer = Trainer(model=model, args=training_args, 
#                   train_dataset=tokenized_dataset["train"], 
#                   data_collator=data_collator, tokenizer=tokenizer)
# trainer.train()

