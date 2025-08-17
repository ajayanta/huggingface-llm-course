from datasets import load_dataset
from transformers import AutoTokenizer
checkpoint="bert-base-uncased"
raw_datasets=load_dataset("glue","mrpc")
# print(raw_datasets)
# print(raw_datasets['train'])
# print(raw_datasets['train'][0])
# print(raw_datasets['train'].features)
# print(raw_datasets['train'].features)
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
# tokenized_pairs = tokenizer(
#     list(raw_datasets["train"]["sentence1"]),
#     list(raw_datasets["train"]["sentence2"]),
#     padding=True,
#     truncation=True
# )
# print(tokenized_pairs)
# input=tokenizer('This is the first Sentence',"this is the second one")
# print(tokenizer.convert_ids_to_tokens(input['input_ids']))

def tokenize_function(example):
    return tokenizer(example['sentence1'],example['sentence2'],truncation=True)
tokenized_dataset=raw_datasets.map(tokenize_function,batched=True)
print(tokenized_dataset)