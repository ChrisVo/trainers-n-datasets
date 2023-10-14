from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# Tokenization function
def tokenization(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=512
    )


# Load and preprocess the dataset
dataset = load_dataset("rotten_tomatoes", split="train")
dataset = dataset.map(tokenization, batched=True)
dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)

# Split the dataset into training and evaluation sets
train_dataset = load_dataset("rotten_tomatoes", split="train[:80%]")
train_dataset = train_dataset.map(
    tokenization, batched=True
)  # Tokenize the training dataset
train_dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)  # Set the correct columns

eval_dataset = load_dataset("rotten_tomatoes", split="train[80%:]")
eval_dataset = eval_dataset.map(
    tokenization, batched=True
)  # Tokenize the evaluation dataset
eval_dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)  # Set the correct columns

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    logging_dir="./logs",
    num_train_epochs=3,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=8,
)

# Create a data collator to handle padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the Trainer
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Provide eval_dataset here
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Eval Results: {eval_results}")

# Save the model
model.save_pretrained("./saved_model")
