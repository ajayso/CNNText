Using CNN for text data

Introduction
CNN has done exceptionally well for computer vision problems given their ability to extract features from local input patches, learning local patterns. The interesting points when it comes to CNN i.e. the patterns, they learn are translation invariant and more importantly ability to learn spatial hierarchies.  
The spatial dimension in CNN are namely height, width and coming to think of it time can be spatial dimension except we would be dealing with a 1D tensor on a 1D Convnet.
Like images the sequence data can be processed using 1D Convnets.

Getting into the details of 1D Convnets 
 In image-based data the 2D convolution layer was extracting 2D patches from a 3D tensor.  In a similar manner 1D convnets can be used to extract 1D patches.  1D convnets can recognize local pattern in a sequence and are translation invariant i.e. the same pattern can then be recognized at a different position. For example, a 1D Convnet processing a sequence of characters using a convolution window of size 7 should be able to learn words or word fragments of length 7 or less and recognize these words in reference to an input sentence. 
In Conv2D networks, 2D pooling operations are used similarly in 1D convnets we use 1D pooling. 
Writing Code….
Keras does come to the rescue for 1D operations there are 
	1D Conv Layer - layers.Conv1D
	1D Max Pooling - layers.MaxPooling1D
The code construct is very much similar to what is used in CNN image.
model = Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(x,y)))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(Dense(41, activation='softmax'))
News Article Categorization
News article categorization is a very common requirement across the industry. The scenario is we have about approximately 200 news articles in the json format below is an example set 

Considering we have been given the category for starters , the headline alone can be good enough to decide the category, ideally one can go on to add the headline and pull the complete news article from the link to get to much higher accuracy, for the example only headline data will be used.
Headline is essentially text data which needs to be tokenized and vectorized for model consumption.  There are multiple ways of associating token to a vector namely 
	One-Hot Encoding which essentially consists of associating a unique integer index with every word and then turning this integer index i into a binary vector of size N (the size of the vocabulary); the vector is all zeros except for the i th entry, which is 1. and token embedding.
	Word Embeddings – a vector is associated with each word, There are a lot of pretrained word embedding which can be used as is into the model from example Word2Vec, Glove etc for more details refer to this link https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010.
Instead of learning word embeddings jointly with the problem, have chosen to use embedding vectors from a precomputed embedding space as these are highly structured and exhibits useful properties—that captures generic aspects of language structure. 
The rationale behind using pretrained word embeddings in natural-language processing is much the same as for using pretrained convnets in image classification: when there is limited data available to learn truly powerful features. but you expect the features that you need to be generic—that is, common visual features or semantic features. In this case, it makes sense to reuse features learned on a different problem. 
Such word embeddings are generally computed using word occurrence statistics (observation about the word that co-occurs in sentences and documents) using techniques such as neural networks and others.
Some of most famous and successful word-embedding schemes: the Word2vec algorithm (https://code.google.com/archive/p/word2vec), developed by Tomas Mikolov at Google in 2013. Word2vec dimensions capture specific semantic properties, such as gender.
Keras Embedding layer provides various other precomputed databases for word embeddings. Global Vectors for Word Representation (GloVe, https://nlp.stanford.edu/projects/glove),

The code use glove word embedding.  Our execution scenario includes 2 approaches
Complete code can be found here.
Data Load
 

Approach #1 -Using Pretrained Glove Embeddings as is in a simple model
We load the corresponding embedding vectors from glove map for the matching words. 
 


The model is straightforward, we load the glove embedding at the 0th layer and towards the end we have a SoftMax with 41 desired outputs matching to the different categories.
 
Additionally, we freeze the Embedding layer i.e. set trainable = False as we don’t want the weights to be changed. Following the rationale in context of pretrained convnets (in this case the Embedding Layer) and parts are randomly initialized, the pretrained part should not be updated during training forgetting what it has already learnt. 
 
A very simple model indeed can yield a training accuracy up to 97.9% and validation accuracy up to
 97.7. 
 
Below is the training and validation loss chart
 
 
Approach #2 -Using CNN and Pretrained Glove Embeddings 
As explained earlier the input here is Embedding layer with defined dimensions. A stack of Conv1D and MaxPooling layers ending a global pooling layer or a Flatten layer that turn the 3D inputs into 2 D inputs allowing to add 1 more Dense layer to the model for classification or regression. The convolution window for 1D can be larger as 1D Convolution window of 3 contains 3 vectors and we could use a window of size 7 or 9.
 
Let’s dive into each layer and see what is happening:
•	Input data: Embedding layer converts positive integers (indexes) into dense vectors of fixed size. In this case it input dimension is max_words and outputs (max_len, embedding_dim) or (1000, 50)
•	First 1D CNN layer: The first layer defines a filter (or also called feature detector) of height 7 (also called kernel size). This can help in identifying words up to length 7 or less or word fragments of length. This might not be enough; therefore, we will define 128 filters. This allows us to train 32 different features on the first layer of the network. The output of the first neural network layer is a 994 x 128 neuron matrix. Each column of the output matrix holds the weights of one single filter. With the defined kernel size and considering the length of the input matrix, each filter will contain 994 weights.
•	Max pooling layer: A pooling layer is often used after a CNN layer in order to reduce the complexity of the output and prevent overfitting of the data. In our example we chose a size of 5. This means that the size of the output matrix of this layer is only a fifth of the input matrix.
•	Second and Third 1D CNN layer: The result from the Max Pooling layer will be fed into the second CNN layer. We will again define 128 different filters to be trained on this level. Following the same logic as the first layer, the output matrix will be of size 192 x 128 , 186 X 128
•	Global Max pooling layer: GlobalMaxPooling1D for temporal data takes the max vector over the steps dimension.
•	Fully connected layer with SoftMax activation: The final layer will reduce the vector of height 128 to a vector of six since we have 41 classes that we want to predict (“CRIME, ENTERTAINMENT, WORLD NEWS, IMPACT, POLITICS). This reduction is done by another matrix multiplication. SoftMax is used as the activation function. It forces all outputs of the neural network to sum up to one. The output value will therefore represent the probability for each of the six classes.

 The training and validation accuracy are very similar to the earlier approach
 
Below is the training and validation loss
 
Final Comments
Validation accuracy is somewhat less than that of the LSTM, but runtime is faster on both CPU and GPU (the exact increase in speed will vary greatly depending on your exact configuration).


Code Files
UsingPreTrainedEmbeddings.py code for Pre Trained Embeddings basic network
UsingCNNPreTrainedEmbeddings.py code for CNN Pre Trained Embeddings
