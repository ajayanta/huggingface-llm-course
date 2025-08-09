import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Example inputs
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]

# Tokenize
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# Forward pass
outputs = model(**inputs)

# Apply softmax to get probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

# Get predicted class indices
predicted_classes = torch.argmax(predictions, dim=1)

# Map IDs to labels
labels = model.config.id2label
for text, pred, probs in zip(raw_inputs, predicted_classes, predictions):
    label = labels[pred.item()]
    confidence = probs[pred.item()].item()
    print(f"Text: {text}")
    print(f"Predicted label: {label} (confidence: {confidence:.4f})\n")
