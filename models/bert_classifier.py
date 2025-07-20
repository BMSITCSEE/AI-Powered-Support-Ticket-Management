"""
BERT-based ticket classifier
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import numpy as np
from app.config import Config
import os
import json
from typing import List, Tuple, Dict

class BertTicketClassifier:
    """BERT-based classifier for support tickets"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.category_labels = Config.TICKET_CATEGORIES
        self.urgency_labels = Config.URGENCY_LEVELS
        
    def load_model(self):
        """Load pre-trained model"""
        model_path = Config.MODEL_PATH
        
        if os.path.exists(model_path):
            self.model = TicketClassificationModel(
                num_categories=len(self.category_labels),
                num_urgency_levels=len(self.urgency_labels)
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            # Initialize with default model for demo
            print("Warning: No pre-trained model found. Initializing with random weights.")
            self.model = TicketClassificationModel(
                num_categories=len(self.category_labels),
                num_urgency_levels=len(self.urgency_labels)
            )
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, text: str) -> Tuple[str, str, float]:
        """Predict category and urgency for given text"""
        if self.model is None:
            self.load_model()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=Config.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            category_logits, urgency_logits = self.model(input_ids, attention_mask)
            
            # Get predictions
            category_probs = torch.softmax(category_logits, dim=-1)
            urgency_probs = torch.softmax(urgency_logits, dim=-1)
            
            category_idx = torch.argmax(category_probs, dim=-1).item()
            urgency_idx = torch.argmax(urgency_probs, dim=-1).item()
            
            category = self.category_labels[category_idx]
            urgency = self.urgency_labels[urgency_idx]
            confidence = category_probs[0][category_idx].item()
        
        return category, urgency, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict categories and urgencies for batch of texts"""
        if self.model is None:
            self.load_model()
        
        results = []
        
        # Process in batches
        batch_size = Config.BATCH_SIZE
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=Config.MAX_SEQUENCE_LENGTH,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                category_logits, urgency_logits = self.model(input_ids, attention_mask)
                
                category_probs = torch.softmax(category_logits, dim=-1)
                urgency_probs = torch.softmax(urgency_logits, dim=-1)
                
                category_indices = torch.argmax(category_probs, dim=-1)
                urgency_indices = torch.argmax(urgency_probs, dim=-1)
                
                for j in range(len(batch_texts)):
                    results.append({
                        'category': self.category_labels[category_indices[j].item()],
                        'urgency': self.urgency_labels[urgency_indices[j].item()],
                        'category_confidence': category_probs[j][category_indices[j]].item(),
                        'urgency_confidence': urgency_probs[j][urgency_indices[j]].item()
                    })
        
        return results
    
    def get_prediction_explanations(self, text: str) -> Dict:
        """Get detailed explanations for predictions"""
        if self.model is None:
            self.load_model()
        
        # Get predictions
        category, urgency, confidence = self.predict(text)
        
        # Tokenize for attention weights
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=Config.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Average attention weights across all layers and heads
            attentions = outputs.attentions
            avg_attention = torch.mean(torch.stack(attentions), dim=0)
            avg_attention = torch.mean(avg_attention, dim=1)[0]
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Create token-attention pairs
            token_attention_pairs = [
                (token, attention.item()) 
                for token, attention in zip(tokens, avg_attention[0])
                if token not in ['[CLS]', '[SEP]', '[PAD]']
            ]
            
            # Sort by attention weight
            token_attention_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'category': category,
            'urgency': urgency,
            'confidence': confidence,
            'important_tokens': token_attention_pairs[:10]  # Top 10 important tokens
        }


class TicketClassificationModel(nn.Module):
    """Neural network model for ticket classification"""
    
    def __init__(self, num_categories, num_urgency_levels, dropout_rate=0.3):
        super(TicketClassificationModel, self).__init__()
        
        # BERT base model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers (optional, for faster training)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # Classification heads
        hidden_size = self.bert.config.hidden_size
        
        # Category classification head
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_categories)
        )
        
        # Urgency classification head
        self.urgency_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_urgency_levels)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Get predictions
        category_logits = self.category_classifier(pooled_output)
        urgency_logits = self.urgency_classifier(pooled_output)
        
        return category_logits, urgency_logits
    
    def get_embeddings(self, input_ids, attention_mask):
        """Get BERT embeddings for visualization"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.pooler_output