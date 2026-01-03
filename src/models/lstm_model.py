import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import numpy as np

class LSTMEmotionClassifier:
    def __init__(self, vocab_size=10000, embedding_dim=100, max_length=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
    
    def build_model(self, num_classes, lstm_units=128, dropout_rate=0.5, bidirectional=True):
        """Build LSTM model with configurable architecture"""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
        ])
        
        if bidirectional:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
            model.add(Bidirectional(LSTM(lstm_units // 2)))
        else:
            model.add(LSTM(lstm_units, return_sequences=True))
            model.add(LSTM(lstm_units // 2))
        
        model.add(Dropout(dropout_rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))
        
        return model
    
    def prepare_data(self, texts):
        """Tokenize and pad sequences"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded
    
    def train(self, X_train, y_train, X_val, y_val, 
              lstm_units=128, dropout_rate=0.5, bidirectional=True,
              learning_rate=0.001, epochs=20, batch_size=64):
        """Train LSTM model with early stopping"""
        
        # Prepare data
        X_train_pad = self.prepare_data(X_train)
        X_val_pad = self.prepare_data(X_val)
        
        # Convert labels to categorical
        num_classes = len(np.unique(y_train))
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
        
        # Build model
        self.model = self.build_model(num_classes, lstm_units, dropout_rate, bidirectional)
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='macro')]
        )
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        checkpoint = ModelCheckpoint('models/lstm_best.h5', save_best_only=True, monitor='val_f1_score', mode='max')
        
        # Train
        history = self.model.fit(
            X_train_pad, y_train_cat,
            validation_data=(X_val_pad, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        X_pad = self.prepare_data(X)
        predictions = self.model.predict(X_pad)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        X_pad = self.prepare_data(X)
        return self.model.predict(X_pad)
    
    def save(self, model_path, tokenizer_path):
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)