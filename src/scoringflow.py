from metaflow import FlowSpec, step, Parameter
import mlflow
import torch
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, Any
import os

class TransformerScoringFlow(FlowSpec):
    """Flow for scoring using a trained transformer model."""
    
    text = Parameter('text', type=str, help='Text to analyze')
    model_stage = Parameter('stage', default='None', type=str, help='Model stage to use')
    
    @step
    def start(self) -> None:
        """Load the model and tokenizer."""
        # Set up MLflow tracking
        mlflow_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db'))
        os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{mlflow_db_path}'
        mlflow.set_tracking_uri(f'sqlite:///{mlflow_db_path}')
        
        # Load the model from MLFlow model registry
        model_name = 'youtube-sentiment-transformer'
        try:
            self.model = mlflow.pytorch.load_model(
                model_uri=f"models:/{model_name}/1"  # Use version 1 since we just created it
            )
            
            # Load tokenizer from the same run
            run = mlflow.get_run(mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{model_name}'")[0].run_id)
            tokenizer_path = [f for f in run.data.params if 'tokenizer' in f.lower()][0]
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # Use same base model as training
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.next(self.predict)
    
    @step
    def predict(self) -> None:
        """Make prediction on the input text."""
        with torch.no_grad():
            inputs = self.tokenizer(
                self.text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            self.result = {
                'text': self.text,
                'sentiment': 'positive' if prediction == 1 else 'negative',
                'confidence': confidence
            }
        
        self.next(self.end)
    
    @step
    def end(self) -> None:
        """Print the results."""
        print("\nPrediction Results:")
        print(f"Text: {self.result['text']}")
        print(f"Sentiment: {self.result['sentiment']}")
        print(f"Confidence: {self.result['confidence']:.2%}")

if __name__ == '__main__':
    TransformerScoringFlow() 