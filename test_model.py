from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./saved_model")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

text = "The movie was crazy good!"


input = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    output = model(**input)

predicted_class = torch.argmax(output.logits, dim=1).item()
print("Predicted class:", predicted_class)
