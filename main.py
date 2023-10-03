from icecream import ic
import torch
# import sentencepiece
# import tokenizers
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
def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)
trainer = Trainer(model=model, args=training_args, 
                  train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["validation"], 
                  data_collator=data_collator, tokenizer=tokenizer)
trainer.train()


