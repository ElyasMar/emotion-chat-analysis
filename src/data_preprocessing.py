import pandas as pd
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self):
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def load_goemotions(self, cache_dir='./data/raw'):
        """Load GoEmotions dataset from HuggingFace"""
        print("Loading GoEmotions dataset...")
        ds = load_dataset("mrm8488/goemotions", cache_dir=cache_dir)
        
        # Convert to pandas
        train_df = pd.DataFrame(ds['train'])
        val_df = pd.DataFrame(ds['validation'])
        test_df = pd.DataFrame(ds['test'])
        
        return train_df, val_df, test_df
    
    def preprocess_data(self, df):
        """Clean and preprocess text data"""
        df = df.copy()
        
        # Remove nulls
        df = df.dropna(subset=['text'])
        
        # Convert labels to list format
        df['emotions'] = df['labels'].apply(lambda x: x if isinstance(x, list) else [x])
        
        # Get primary emotion (most confident)
        df['primary_emotion'] = df['emotions'].apply(lambda x: x[0] if len(x) > 0 else 27)
        df['primary_emotion_name'] = df['primary_emotion'].apply(
            lambda x: self.emotion_labels[x] if x < len(self.emotion_labels) else 'neutral'
        )
        
        # Text length
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        return df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir='./data'):
        """Save processed datasets"""
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(f'{output_dir}/train_processed.csv', index=False)
        val_df.to_csv(f'{output_dir}/val_processed.csv', index=False)
        test_df.to_csv(f'{output_dir}/test_processed.csv', index=False)
        print(f"Saved processed data to {output_dir}")

if __name__ == "__main__":
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_goemotions()
    
    train_df = loader.preprocess_data(train_df)
    val_df = loader.preprocess_data(val_df)
    test_df = loader.preprocess_data(test_df)
    
    loader.save_processed_data(train_df, val_df, test_df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")