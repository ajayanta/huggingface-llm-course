from transformers import AutoTokenizer
# tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
# tokens=tokenizer.tokenize("Let's try to tokenize!")
# print(tokens)
# ['let', "'", 's', 'try', 'to', 'token', '##ize', '!']
tokenizer=AutoTokenizer.from_pretrained('albert-base-v1')
tokens=tokenizer.tokenize("Let's try to tokenize!")
# ['▁let', "'", 's', '▁try', '▁to', '▁to', 'ken', 'ize', '!']
input_ids=tokenizer.convert_tokens_to_ids(tokens)
# [408, 22, 18, 1131, 20, 20, 2853, 2952, 187]
final_input=tokenizer.prepare_for_model(input_ids)
print(final_input['input_ids'])
#[2, 408, 22, 18, 1131, 20, 20, 2853, 2952, 187, 3] 
print(tokenizer.decode(final_input['input_ids']))
# [CLS] let's try to tokenize![SEP]
print(tokenizer.decode([408, 22, 18, 1131, 20, 20, 2853, 2952, 187]))
# let's try to tokenize!