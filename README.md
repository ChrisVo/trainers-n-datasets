# Learning to Train with Datasets using Transformers

This repository contains a script for training a sequence classification model using the `transformers` and `datasets` libraries on the Rotten Tomatoes dataset.

## Setup

1. Ensure you have Python 3.6 or later installed.
2. Clone this repository to your local machine.
3. Install the required libraries:
    ```bash
    pip install transformers datasets torch
    ```

## Usage

1. Run the script `app.py` to train the model:
    ```bash
    python app.py
    ```

The script `app.py` will perform the following operations:

- Load the pre-trained `bert-base-cased` model and tokenizer from the `transformers` library.
- Load the Rotten Tomatoes dataset and split it into training (80%) and evaluation (20%) subsets.
- Tokenize the dataset using the tokenizer from `transformers`.
- Define training arguments such as the output directory, logging directory, number of training epochs, and evaluation strategy.
- Create a data collator for padding tokenized inputs to the same length.
- Initialize a `Trainer` object from the `transformers` library with the training arguments, model, datasets, and data collator.
- Train the model on the training dataset.
- Evaluate the model on the evaluation dataset.
- Print evaluation results to the console.
- Save the trained model to the `./saved_model` directory.

## Outputs

- A trained model saved in the `./saved_model` directory.
- Training logs saved in the `./logs` directory.
- Training and evaluation metrics printed to the console.

## References

- [Transformers Library](https://huggingface.co/transformers/)
- [Datasets Library](https://huggingface.co/docs/datasets/)
- [Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes)
