# #   Model checkpoint
# checkpoint_mb = "bert-base-uncased"

# #   Core model/checpoint-specific objects
# tokenizer_mb = AutoTokenizer.from_pretrained(checkpoint_mb)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint_mb, num_labels=2)
# data_collator_mb = DataCollatorWithPadding(tokenizer=tokenizer_mb)          # for batching
# training_args_mb = TrainingArguments("test-trainer")                         # training hyperperameters

# #   Dataset
# raw_dataset = load_dataset("glue", "mrpc")                                  # mrpc is config of glue
# def tokenize_fn(example):
#     return tokenizer_mb(example["sentence1"], example["sentence2"], truncation=True)
# tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)


# #   Training
# trainer_mb = Trainer(model=model, args=training_args_mb, 
#                      train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["validation"], 
#                      data_collator=data_collator_mb, tokenizer=tokenizer_mb)

# trainer_mb.train()











# batch = tokenizer_mb(sequences, padding=True, truncation=True, return_tensors="pt"); 
# batch["labels"] = torch.tensor([1, 1])

# loss = model(**batch).loss


# samples_toy = tokenized_dataset["train"][:8]
# samples_toy = {k: v for k, v in samples_toy.items() if k not in ["idx", "sentence1", "sentence2"]}
# batch = data_collator(samples_toy)
# ic({k: v.shape for k, v in batch.items()})




# from transformers import BertConfig, BertModel, BertTokenizer

# sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
    # tokens = tokenizer.tokenize(sequence)
    # ids = tokenizer.convert_tokens_to_ids(tokens)

## Returns PyTorch tensors
# model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
## Returns NumPy arrays
# model_inputs = tokenizer(sequences, padding=True, return_tensors="np")

## To randomly initialize the model
#Building the config
# config = BertConfig()
# print(config)
#Building the model from the config
# model = BertModel(config)

## To load pre-trained weights into the model
# model = BertModel.from_pretrained("bert-base-cased")

## To save saves: config.json pytorch_model.bin
# model.save_pretrained("directory_on_my_computer")

## To create tokenizer (two ways)
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

## To decode:
# decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])

## Summary of HG.0
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
# output = model(**tokens)

