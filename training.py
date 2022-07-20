from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, LSTM,Dense, Dropout, SpatialDropout1D, Embedding
from keras import regularizers
from tensorflow.keras import optimizers
import numpy as np

class Training:
    def __init__(self):
        super(Training, self).__init__()

    def tokenize_text(self, X):
        self.token = Tokenizer()
        self.token.fit_on_texts(X)
        tokenized_text = self.token.texts_to_sequences(X)
        input_dim = max(len(txt) for txt in tokenized_text) 
        self.padded_tokenized_text = pad_sequences(tokenized_text, maxlen=input_dim).tolist()
        return self.token

    def build_bilstm_model(self, app_config):
        """
        Building BiLSTM model
        """
        model = Sequential()
        model.add(Embedding(input_dim = len(self.token.word_index) + 1, output_dim = app_config.output_dim, input_length = app_config.input_len))
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
        model.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(32,kernel_regularizer=regularizers.l2(app_config.regularizers),activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=app_config.learning_rate), metrics = ['accuracy', 'AUC'])
        return model

    @staticmethod
    def train_model(model, X, Y, epochs=10, batch_size=64, validation_split=0.0):
        """
        Training model on given data
        """
        history = model.fit(np.array(X.values.tolist()),Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history