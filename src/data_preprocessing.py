import pandas as pd
from datasets import load_dataset
import numpy as np
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
        print("This may take a few minutes on first run...")
        
        try:
            ds = load_dataset("google-research-datasets/go_emotions", "simplified")
            
            # Convert to pandas
            train_df = pd.DataFrame(ds['train'])
            val_df = pd.DataFrame(ds['validation'])
            test_df = pd.DataFrame(ds['test'])
            
            print(f"✓ Dataset loaded successfully")
            print(f"  Train: {len(train_df)} samples")
            print(f"  Validation: {len(val_df)} samples")
            print(f"  Test: {len(test_df)} samples")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTrying alternative dataset source...")
            
            # Alternative: try the mrm8488 version
            ds = load_dataset("mrm8488/goemotions", cache_dir=cache_dir)
            train_df = pd.DataFrame(ds['train'])
            val_df = pd.DataFrame(ds['validation'])
            test_df = pd.DataFrame(ds['test'])
            
            print(f"✓ Dataset loaded from alternative source")
            return train_df, val_df, test_df
    
    def preprocess_data(self, df):
        """Clean and preprocess text data"""
        print(f"Preprocessing {len(df)} samples...")
        df = df.copy()
        
        # Check available columns
        print(f"Available columns: {df.columns.tolist()}")
        
        # Remove nulls from text
        df = df.dropna(subset=['text'])
        
        # Handle labels - they might be in different formats
        if 'labels' in df.columns:
            # If labels is already a column
            df['emotions'] = df['labels'].apply(lambda x: x if isinstance(x, list) else [x])
        elif any(col.startswith('label') for col in df.columns):
            # If we have label_0, label_1, etc columns
            label_cols = [col for col in df.columns if col.startswith('label')]
            df['emotions'] = df[label_cols].apply(lambda row: [i for i, val in enumerate(row) if val == 1], axis=1)
        else:
            print("Warning: Could not find label columns")
            # Create a default label
            df['emotions'] = [[27]] * len(df)  # neutral
        
        # Get primary emotion (first one or most common)
        df['primary_emotion'] = df['emotions'].apply(lambda x: x[0] if len(x) > 0 else 27)
        df['primary_emotion_name'] = df['primary_emotion'].apply(
            lambda x: self.emotion_labels[x] if x < len(self.emotion_labels) else 'neutral'
        )
        
        # Text features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Clean text
        df['text'] = df['text'].astype(str).str.strip()
        
        print(f"✓ Preprocessing complete: {len(df)} samples")
        print(f"  Emotion distribution (top 5):")
        print(df['primary_emotion_name'].value_counts().head())
        
        return df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir='./data'):
        """Save processed datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}/...")
        
        # Select relevant columns
        columns_to_save = ['text', 'primary_emotion', 'primary_emotion_name', 
                          'text_length', 'word_count', 'emotions']
        
        # Make sure all columns exist
        for col in columns_to_save.copy():
            if col not in train_df.columns:
                columns_to_save.remove(col)
        
        train_df[columns_to_save].to_csv(f'{output_dir}/train_processed.csv', index=False)
        val_df[columns_to_save].to_csv(f'{output_dir}/val_processed.csv', index=False)
        test_df[columns_to_save].to_csv(f'{output_dir}/test_processed.csv', index=False)
        
        print(f"✓ Saved processed data:")
        print(f"  - {output_dir}/train_processed.csv")
        print(f"  - {output_dir}/val_processed.csv")
        print(f"  - {output_dir}/test_processed.csv")

def main():
    print("="*60)
    print("EMOTION DATASET PREPROCESSING")
    print("="*60)
    
    loader = DataLoader()
    
    # Load dataset
    try:
        train_df, val_df, test_df = loader.load_goemotions()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have internet connection for first-time download.")
        return
    
    # Preprocess
    train_df = loader.preprocess_data(train_df)
    val_df = loader.preprocess_data(val_df)
    test_df = loader.preprocess_data(test_df)
    
    # Save
    loader.save_processed_data(train_df, val_df, test_df)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nYou can now run: python src/train.py --model all")

if __name__ == "__main__":
    main()