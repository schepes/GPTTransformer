README

This repository contains two Python scripts implementing character-based language models, and 2 data files. The first script, firstModel.py, uses a simple bigram model to generate text, while the second script, finalModel.py, utilizes a GPT-like Transformer architecture for the same task.

Requirements
Python 3.8+
PyTorch 1.7+
A GPU is highly recommended for running the GPT-like Transformer model (finalModel.py).

Description
Simple Bigram Model
This model is implemented in the first script. It reads an input text file, tokenizes it into unique characters, and trains a simple bigram model to predict the next character in the sequence. After training, the model generates new text based on the learned relationships between characters.

GPT-like Transformer Model
The second script implements a GPT-like Transformer model for the character-based language modeling task. It is a smaller version of the original GPT architecture with fewer layers and heads. This model is more powerful and capable of generating more coherent text than the simple bigram model, but it requires more computational resources. A GPU is highly recommended for running this script.

Usage
Remember to install all necessary dependecies, if you wish to use another datafile, dont forget to add to the repository and make necessary changes for reading the dataset in the code.

Run the desired script:

For the Simple Bigram Model: python firstModel.py
For the GPT-like Transformer Model: python finalModel.py
Note: It is highly recommended to use a GPU when running finalModel.py. Running it on a CPU or a Mac might result in slow training and generation times.

The script will display training and validation losses during training. After training, it will generate new text based on the learned relationships between characters and print it to the console.

Disclaimer (AGAIN)
Please note that the finalModel.py script should not be run on a CPU or a Mac. If you wish to test the code, it is highly recommended to use a GPU for optimal performance.
