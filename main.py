from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # number of cells in each LSTM layer
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '/Users/kt6238/Downloads/fra-eng/fra.txt'

# Vectorize the data.
input_texts = [] # list as a holder
target_texts = []
input_characters = set() # set is unordered collection of unique elements. This is just a holder.
target_characters = set()
lines = open(data_path, 'r', encoding='utf-8').read().split('\n') # my unit of data for analysis is each line
for line in lines[: min(num_samples, len(lines) - 1)]: # entire list of num_samples or entire original, whichever is smaller.
    input_text, target_text = line.split('\t') # split each line by source and target
    target_text = '\t' + target_text + '\n' # We use "tab" as the "start sequence" character for the targets, and "\n" as "end sequence" character.
    input_texts.append(input_text) # use append to put an element into a list that is already declared.
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32') # three dimension, with size of each dimension indicated.
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)): # element-wise, pair-wise formation of a tuple () in a list from two lists.
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1. # ith observation of input, tth position in input, and the one-hot token index there.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1. # this is like train the model to understand that
            # when encoder is in 5th position, I want decoder to be in 4th,
            # therefore the decoder is always one step behind encoder, as it should. Decoder has to follow and beb behind the encoder by 1 step in this case.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens)) #shape parameter takes a tuple that indicates dimension of input data:: variable rows, num of token columns.
encoder = LSTM(latent_dim, return_state=True)  #latent_dim is numbebr of neuron in this layer.
encoder_outputs, state_h, state_c = encoder(encoder_inputs) # LSTM layer connects to input layer.
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c] # get hidden state and internal cell state



# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s3s.h5')


#encoder_input is just an empty holder for defining the LSTM model. encoder_input_data which has the real data is used in model fitting.

