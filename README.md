# Name Entity Recognition of Text Corpus Using Transformer Encoder-Based Model

This Python notebook does Name Entity Recognition of a Text Corpus, and is broken down as following:

### 1. Tokenizer, Data Loader and Dataset

The Tokenizer class contains information about tokenizing a given sentence. The encode method encodes a given text corpus, creating a dictionary of the words that appear in the corpus. The data loader method takes the text corpus and its associated tags and creates a dictionary with the corpus tensor in as 'text' and its associated tags tensor as 'tags'. Finally, the NERDataset creates the final encoded tensor that is ready to be fed into the Transformer model.

### 3. Transformer Encoder-Based Model

I used 10 torch.nn.TransformerEncoderLayer layers, a positional encoder and 2 linear layers, along with uniform weight initialization. I found the right hyperparameters by conducting a localized binary search, with the assumption that accuracy as a function of a given hyperparameter is continuous locally.

### 4. Training and Validation Methods

Training and validation methods. In the validation method, the inputs are through the model and the resulting logits (output) are used to calculate loss against the validate split labels. In the training method, the same thing happens except we call loss.backwards(), which calculates the derivative of the loss with respect to the parameters in the model at every point in the computational graph (which is implicitly maintained by PyTorch). Then, calling optimizer.step() updates the parameters of the model using Adam (the optimizer we are using for this project).

### 5. F1 Score Calculation and Test Split Predictions

Defines the predict method, in which inputs are fed into the model, and the resulting logits are used to predict the NER tags of the input. 

Runs those tags through conlleval, a script that is used to measure the f1 score of NER. 
