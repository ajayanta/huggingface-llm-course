from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
tokens=tokenizer.token("Let's try to tokenize!")
print(tokens)