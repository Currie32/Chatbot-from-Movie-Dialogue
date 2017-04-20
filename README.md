# Chatbot-from-Movie-Dialogue

I built a simple chatbot using conversations from Cornell University's [Movie Dialogue Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). The main features of our model are LSTM cells, a bidirectional dynamic RNN, and decoders with attention. 

The conversations will be cleaned rather extensively to help the model to produce better responses. As part of the cleaning process, punctuation will be removed, rare words will be replaced with "UNK" (our "unknown" token), longer sentences will not be used, and all letters will be in the lowercase. 

With a larger amount of data, it would be more practical to keep features, such as punctuation. However, I am using FloydHub's GPU services and I don't want to get carried away with too training for too long.
