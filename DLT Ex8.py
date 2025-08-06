import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Sample conversation data
input_texts = ['hello', 'how are you', 'what is your name', 'bye']
target_texts = ['hi', 'i am fine', 'i am a bot', 'goodbye']

# Add start and end tokens
target_texts = ['<start> ' + txt + ' <end>' for txt in target_texts]

# Tokenize
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

input_sequences = pad_sequences(input_sequences, padding='post')
target_sequences = pad_sequences(target_sequences, padding='post')

vocab_size = len(tokenizer.word_index) + 1
max_input_len = input_sequences.shape[1]
max_target_len = target_sequences.shape[1]

# Hyperparameters
embedding_dim = 64
lstm_units = 128

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(enc_emb)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm_outputs, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(
    dec_emb, initial_state=[state_h, state_c])

# Attention and context
attention_layer = Attention()
context_vector = attention_layer([decoder_lstm_outputs, encoder_outputs])
combined = Concatenate(axis=-1)([context_vector, decoder_lstm_outputs])
decoder_outputs = Dense(vocab_size, activation='softmax')(combined)

# Final model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare target output
decoder_target_data = np.expand_dims(target_sequences, -1)

# Train the model
model.fit([input_sequences, target_sequences], decoder_target_data, batch_size=2, epochs=200, verbose=0)

# ============ Inference Setup ============ #
# Encoder model for inference
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder setup
dec_state_input_h = Input(shape=(lstm_units,))
dec_state_input_c = Input(shape=(lstm_units,))
enc_output_input = Input(shape=(max_input_len, lstm_units))

dec_emb2 = Embedding(vocab_size, embedding_dim)(decoder_inputs)
dec_lstm_outputs2, state_h2, state_c2 = LSTM(lstm_units, return_sequences=True, return_state=True)(
    dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c])
context_vector2 = Attention()([dec_lstm_outputs2, enc_output_input])
concat2 = Concatenate(axis=-1)([context_vector2, dec_lstm_outputs2])
dec_outputs2 = Dense(vocab_size, activation='softmax')(concat2)

decoder_model = Model(
    [decoder_inputs, enc_output_input, dec_state_input_h, dec_state_input_c],
    [dec_outputs2, state_h2, state_c2]
)

# ============ Response Generator ============ #
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    enc_out, state_h, state_c = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.array([[tokenizer.word_index['<start>']]])
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, enc_out, state_h, state_c], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or sampled_word == '' or len(decoded_sentence.split()) > max_target_len:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + ' '
            target_seq = np.array([[sampled_token_index]])
            state_h, state_c = h, c

    return decoded_sentence.strip()

# ============ Test ============ #
test_input = "how are you"
print("Input :", test_input)
print("Bot   :", generate_response(test_input))
