import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.logistic_model import LogisticEmotionClassifier
from src.models.lstm_model import LSTMEmotionClassifier
from src.models.bert_model import BERTEmotionClassifier


def load_data():
    """Load preprocessed data"""
    train_df = pd.read_csv('data/train_processed.csv')
    val_df = pd.read_csv('data/val_processed.csv')
    test_df = pd.read_csv('data/test_processed.csv')
    
    return train_df, val_df, test_df


def train_logistic_regression(train_df, val_df, test_df):
    """Train and evaluate Logistic Regression model"""
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*50)
    
    X_train = train_df['text'].values
    y_train = train_df['primary_emotion'].values
    X_test = test_df['text'].values
    y_test = test_df['primary_emotion'].values
    
    clf = LogisticEmotionClassifier()
    clf.train(X_train, y_train, tune_hyperparams=True)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest F1 Score (macro): {f1:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save model
    clf.save('models/logistic_regression_model.pkl')
    print("Model saved to models/logistic_regression_model.pkl")
    
    return clf, f1, acc


def train_lstm(train_df, val_df, test_df):
    """Train and evaluate LSTM model"""
    print("\n" + "="*50)
    print("TRAINING LSTM")
    print("="*50)
    
    X_train = train_df['text'].values
    y_train = train_df['primary_emotion'].values
    X_val = val_df['text'].values
    y_val = val_df['primary_emotion'].values
    X_test = test_df['text'].values
    y_test = test_df['primary_emotion'].values
    
    # Try different configurations
    configs = [
        {'lstm_units': 128, 'dropout_rate': 0.3, 'bidirectional': True, 'learning_rate': 0.001},
        {'lstm_units': 256, 'dropout_rate': 0.5, 'bidirectional': True, 'learning_rate': 0.0005},
    ]
    
    best_f1 = 0
    best_model = None
    for config in configs:
        print(f"\nTrying config: {config}")
        clf = LSTMEmotionClassifier()
        clf.train(X_train, y_train, X_val, y_val, **config, epochs=10)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Test F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf

    # Save best model
    best_model.save('models/lstm_model.h5', 'models/lstm_tokenizer.pkl')
    print(f"\nBest LSTM model saved. F1 Score: {best_f1:.4f}")

    return best_model, best_f1


def train_bert(train_df, val_df, test_df, quick_mode=True):
    """Train and evaluate BERT model"""
    print("\n" + "="*50)
    print("TRAINING BERT")
    print("="*50)
    
    # Use subset for quick training (BERT is slow)
    if quick_mode:
        train_df = train_df.sample(n=5000, random_state=42)
        val_df = val_df.sample(n=1000, random_state=42)

    X_train = train_df['text'].values
    y_train = train_df['primary_emotion'].values
    X_val = val_df['text'].values
    y_val = val_df['primary_emotion'].values
    X_test = test_df['text'].values
    y_test = test_df['primary_emotion'].values

    clf = BERTEmotionClassifier()
    clf.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=16)

    # Evaluate
    y_pred = clf.predict(X_test.tolist())
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest F1 Score (macro): {f1:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # Save model
    clf.save('models/bert_model.pth')
    print("Model saved to models/bert_model.pth")

    return clf, f1, acc


def main():
    parser = argparse.ArgumentParser(description='Train emotion detection models')
    parser.add_argument('--model', type=str, choices=['logistic', 'lstm', 'bert', 'all'],
                        default='all', help='Model to train')
    parser.add_argument('--quick', action='store_true', help='Quick training mode for BERT')
    args = parser.parse_args()

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Load data
    train_df, val_df, test_df = load_data()
    print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    results = {}

    if args.model in ['logistic', 'all']:
        _, f1, acc = train_logistic_regression(train_df, val_df, test_df)
        results['Logistic Regression'] = {'F1': f1, 'Accuracy': acc}

    if args.model in ['lstm', 'all']:
        _, f1 = train_lstm(train_df, val_df, test_df)
        results['LSTM'] = {'F1': f1}

    if args.model in ['bert', 'all']:
        _, f1, acc = train_bert(train_df, val_df, test_df, quick_mode=args.quick)
        results['BERT'] = {'F1': f1, 'Accuracy': acc}

    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()