Rick Beaudet & SJ Franklin
NLP - Spring 2023
Lab 7

The code defines a function RNN_SentimentAnalysis that is a PyTorch 
neural network model. The function takes several parameters that define 
the architecture of the model: 
vocab_size is the size of the vocabulary,
embedding_dim is the dimensionality of the embedding layer, 
num_layers is the number of recurrent layers, 
hidden_dim is the size of the hidden state of each layer, 
bidirectional indicates whether to use bidirectional RNNs, 
output is the size of the output layer, and 
dropout is the dropout probability.

The values of these hyperparameters are user-defined, in order to 
maximize accuracy. The final accuracy percentage for this model is 
71%.

The model is trained using the Adam optimizer and the binary 
cross-entropy loss function. The training is done in mini-batches 
of size batch_size, and the model is trained for a fixed number of 
epochs. During training, the accuracy on the training and validation 
sets is monitored and plotted as a function of the epoch number.