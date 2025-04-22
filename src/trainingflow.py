from metaflow import FlowSpec, step, Parameter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add type hints to help with linting
from typing import List, Tuple, Dict, Any

class TransformerTrainFlow(FlowSpec):
    """Flow for training a transformer model for sentiment analysis."""
    
    # Parameters for model training
    batch_size = Parameter('batch_size', default=32, type=int)
    num_epochs = Parameter('epochs', default=3, type=int)
    learning_rate = Parameter('lr', default=2e-5, type=float)
    model_name = Parameter('model_name', default='distilbert-base-uncased', type=str)
    max_length = Parameter('max_length', default=128, type=int)
    seed = Parameter('seed', default=42, type=int)

    @step
    def start(self) -> None:
        """Load and preprocess the data."""
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Load MrBeast data
        df = pd.read_csv('../MrBeast_youtube_stats.csv')
        
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
        
        # Simple sentiment labeling based on view count (for demonstration)
        df['sentiment'] = (df['viewCount'] > df['viewCount'].median()).astype(int)
        
        # Split data
        texts = df['title'].tolist()
        labels = df['sentiment'].tolist()
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=self.seed
        )
        
        self.train_data = list(zip(train_texts, train_labels))
        self.val_data = list(zip(val_texts, val_labels))
        
        print("Data loaded successfully")
        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}")
        
        self.next(self.prepare_model)

    @step
    def prepare_model(self) -> None:
        """Initialize model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"Model loaded: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.next(self.train_model)

    @step
    def train_model(self) -> None:
        """Train the transformer model."""
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for i in tqdm(range(0, len(self.train_data), self.batch_size)):
                batch_texts, batch_labels = zip(*self.train_data[i:i + self.batch_size])
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                labels = torch.tensor(batch_labels).to(self.device)
                
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / (len(self.train_data) / self.batch_size)
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i in range(0, len(self.val_data), self.batch_size):
                    batch_texts, batch_labels = zip(*self.val_data[i:i + self.batch_size])
                    
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    labels = torch.tensor(batch_labels).to(self.device)
                    
                    outputs = self.model(**inputs, labels=labels)
                    val_loss += outputs.loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
            
            avg_val_loss = val_loss / (len(self.val_data) / self.batch_size)
            accuracy = 100 * correct / total
            
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')
            print(f'Validation accuracy: {accuracy:.2f}%')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_model = self.model.state_dict()
                self.best_accuracy = accuracy
        
        self.next(self.save_model)

    @step
    def save_model(self) -> None:
        """Save the model and log metrics with MLFlow."""
        # Set up MLflow tracking
        mlflow_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db'))
        os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{mlflow_db_path}'
        mlflow.set_tracking_uri(f'sqlite:///{mlflow_db_path}')
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name('youtube-sentiment')
            if experiment is None:
                experiment_id = mlflow.create_experiment('youtube-sentiment')
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Error creating/getting experiment: {e}")
            experiment_id = mlflow.create_experiment('youtube-sentiment')
        
        mlflow.set_experiment('youtube-sentiment')
        
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters
            mlflow.log_params({
                'model_name': self.model_name,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'max_length': self.max_length
            })
            
            # Log metrics
            mlflow.log_metric('best_validation_accuracy', self.best_accuracy)
            
            # Load best model weights
            self.model.load_state_dict(self.best_model)
            
            # Save model with MLFlow
            mlflow.pytorch.log_model(
                self.model,
                'transformer_model',
                registered_model_name='youtube-sentiment-transformer'
            )
            
            # Save tokenizer
            tokenizer_path = 'tokenizer'
            self.tokenizer.save_pretrained(tokenizer_path)
            mlflow.log_artifact(tokenizer_path)
            
            self.run_id = run.info.run_id
        
        self.next(self.end)

    @step
    def end(self) -> None:
        """Final step."""
        print(f"Training completed. Model saved with run ID: {self.run_id}")

if __name__ == '__main__':
    TransformerTrainFlow() 