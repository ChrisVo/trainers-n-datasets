# Import necessary libraries and modules from transformers and datasets
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import load_dataset

# Load the model and tokenizer
# This step initializes a model and tokenizer for sequence classification using the pre-trained 'bert-base-cased' weights.
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# Define a tokenization function
# This function takes a batch of text and tokenizes it into input_ids, token_type_ids, and attention_mask.
def tokenization(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=512
    )


# Load and preprocess the dataset
# The dataset is loaded, tokenized using the function defined above, and formatted to have the necessary columns.
dataset = load_dataset("rotten_tomatoes", split="train")
dataset = dataset.map(tokenization, batched=True)
dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)

# Split the dataset into training and evaluation sets
# The dataset is split into training (80%) and evaluation (20%) subsets, each subset is then tokenized and formatted.
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
# Set the training arguments including the output directory, logging directory, number of training epochs, and evaluation strategy.
training_args = TrainingArguments(
    output_dir="./output",
    logging_dir="./logs",
    num_train_epochs=3,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=8,
)

# Create a data collator
# A data collator is used to collate batches of data. In this case, it's used for padding tokenized inputs to the same length.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the Trainer
# The Trainer is initialized with the training arguments, model, datasets, and data collator defined above.
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Provide eval_dataset here
    data_collator=data_collator,
)

# Train the model
# The train method is called on the trainer to start the training process.
trainer.train()

# Evaluate the model
# The evaluate method is called on the trainer to evaluate the model on the evaluation dataset.
eval_results = trainer.evaluate()
print(f"Eval Results: {eval_results}")

# Save the model
# The trained model is saved to a directory for later use or deployment.
model.save_pretrained("./saved_model")
