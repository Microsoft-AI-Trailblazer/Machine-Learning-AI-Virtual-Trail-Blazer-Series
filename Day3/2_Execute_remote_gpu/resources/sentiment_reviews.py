import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence

# set parameters
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

def build_model(x_train, y_train, x_test, y_test):
    """Build the sentimental Analysis model"""
    model = Sequential()

    # start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # max pooling:
    model.add(GlobalMaxPooling1D())

    # add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
   

    return model

def get_vectors_from_text(dataset_list, word_to_ind = imdb.get_word_index(),
                           start_char=1,
                           index_from=3,
                           maxlen=400,
                           num_words=5000,
                          oov_char=2,skip_top=0):
    '''
    Get the list vector mapped according to the word to indices dictionary.
    
    @param
        dataset_list = list of review texts in unicode format
        hyperparameters: start_char--> sentence starting after this char.
                        index_from--> indices below this will not be encoded.
                        max-len--> maximum length of the sequence to be considered.
                        num_words--> number of words to be considered according to the rank.Rank is
                                    given according to the frequency of occurence
                        oov_char--> out of variable character.
                        skip_top--> no. of top rank words to be skipped
    @returns:
        x_train:        final list of vectors of the review text
    '''
    x_train = []
    for review_string in dataset_list:
        # change the text into list of words
        # if the input file has the unicode format then we must encode it
        
        review_string_list = text_to_word_sequence(review_string)
        
        # words that were not seen in the training set but are in the test set
        # have simply been skipped
        x_predict = []
        for i in range(len(review_string_list)):
            if review_string_list[i] not in word_to_ind:
                continue
            x_predict.append(word_to_ind[review_string_list[i]])
        x_train.append((x_predict))
        
    # add start char and also take care of indexfrom
    if start_char is not None:
        x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
    elif index_from:
        x_train = [[w + index_from for w in x] for x in x_train]
    
    x_train=[ele[:maxlen] for ele in x_train]
    
    if not num_words:
        num_words = max([max(x) for x in x_train])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        x_train = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in x_train]
    else:
        x_train = [[w for w in x if (skip_top <= w < num_words)] for x in x_train]
    
    # padd the sequences
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # return the vectors form of the text
    return x_train

def predict_score(model, review_text, word_to_ind = imdb.get_word_index()):
    '''
    Predict and produce the accuracy of the review text

    @param
        model:SequentialModel which we trained the data on
        review_text:Review text to be predicted on  
        word_to_ind: dictionary mapping of words to indices
    @returns
        sentiment score on the review text.
    '''
    # convert review text into vector 
    x_predict = get_vectors_from_text([review_text], word_to_ind)[0]
    
    # reshape x_predict 
    x_predict = np.reshape(x_predict, (1,len(x_predict)))
    
    return model.predict(x_predict)[0][0]

def get_evaluation_metrics(model, x_test, y_test, show_summary=False):
    """Evaluate the model and get the return the Accuracy on x_test, y_test"""
     # Evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    if show_summary:
        print (model.summary())
    print ("Accuracy: %.2f%%" % (scores[1] * 100))

print ('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print (len(x_train), 'train sequences')
print (len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)

print ('Build  model and evaluate')
model = build_model(x_train, y_train, x_test, y_test)

get_evaluation_metrics(model, x_test, y_test)
# sample tests
positive_review = "I like the movie. It is a thrilling and thought-provoking film that boasts an intellectual story masterfully "
print ("Positive review score: ", predict_score(model,positive_review))
negative_review = "The screen play is too laggy. The movie was a huge disappointment"
print ("Negative review score: ", predict_score(model,negative_review))