# DefGen

Train a reverse dictionary and/or general knowledge crossword question answerer.

Example usage:

Clone the repo, generate your training data and use a command like the following - 

THEANO_FLAGS='floatX=float32, device=gpu' python train_model.py -m your_name_for_the_trained_model.npz -da your_data_files.pkl -di your_dictionary_file.pkl -edim 500

1. If you don't have a GPU, remove "device=gpu"

2. -m:  The model will be saved as a .npz file with the name in the -m argument

3. -da: The data that the model learns from. This must be saved in a pickle (.pkl) file, to which two objects are dumped (in order). The first object is a (python) list of (numpy) arrays, which contain the target (e.g. word2vec) word embeddings. The second object is a (python) list of (python) lists, representing the definitions. Each of these lists should contain integers, which are indices from the total training vocabulary, and encode a particular definition. Both objects should have the same length and be aligned (so that the n-th embedding corresponds to the n-th definition). 

4. -di: This should point to another pickle file to which a python dictionary is dumped. The dictionary should map  word types (as keys) from the total set of training definitions, to unique integers (as values). This should correspond to the encoding used in the second file dumped into the data file -da. 

5. -edim: This should be an integer stating the length of the target (e.g. word2vec) embeddings used in the model. It should be equal to the length of the numpy arrays in the first object dumped to -da. 

Some scripts that may be useful for pre-processing the data into this format can be found in the subdirectory Useful Scripts



