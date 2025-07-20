"""
Training script for BERT ticket classifier
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json

from models.bert_classifier import TicketClassificationModel
from app.config import Config

class TicketDataset(Dataset):
    """Dataset class for ticket data"""
    
    def __init__(self, texts, categories, urgencies, tokenizer, max_length=128):
        self.texts = texts
        self.categories = categories
        self.urgencies = urgencies
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        category = self.categories[idx]
        urgency = self.urgencies[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category_label': torch.tensor(category, dtype=torch.long),
            'urgency_label': torch.tensor(urgency, dtype=torch.long)
        }

def train_model(train_data_path, model_save_path, epochs=10, batch_size=16):
    """Train the BERT classifier"""
    
    # Load data
    print("Loading training data...")
    df = pd.read_csv(train_data_path)
    
    # Prepare labels
    category_labels = Config.TICKET_CATEGORIES
    urgency_labels = Config.URGENCY_LEVELS
    
    category_to_idx = {cat: idx for idx, cat in enumerate(category_labels)}
    urgency_to_idx = {urg: idx for idx, urg in enumerate(urgency_labels)}
    
    # Convert labels to indices
    df['category_idx'] = df['category'].map(category_to_idx)
    df['urgency_idx'] = df['urgency'].map(urgency_to_idx)
    
    # Combine title and description
    texts = (df['title'] + " " + df['description']).tolist()
    categories = df['category_idx'].tolist()
    urgencies = df['urgency_idx'].tolist()
    
    # Split data
    X_train, X_val, y_cat_train, y_cat_val, y_urg_train, y_urg_val = train_test_split(
        texts, categories, urgencies, test_size=0.2, random_state=42, stratify=categories
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = TicketDataset(X_train, y_cat_train, y_urg_train, tokenizer)
    val_dataset = TicketDataset(X_val, y_cat_val, y_urg_val, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TicketClassificationModel(
        num_categories=len(category_labels),
        num_urgency_levels=len(urgency_labels)
    ).to(device)
    
    # Loss functions
    category_criterion = nn.CrossEntropyLoss()
    urgency_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    print(f"Training on {device}...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_cat_correct = 0
        train_urg_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            category_labels = batch['category_label'].to(device)
            urgency_labels = batch['urgency_label'].to(device)
            
            optimizer.zero_grad()
            
            category_logits, urgency_logits = model(input_ids, attention_mask)
            
            cat_loss = category_criterion(category_logits, category_labels)
            urg_loss = urgency_criterion(urgency_logits, urgency_labels)
            
            total_loss = cat_loss + urg_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate accuracy
            _, cat_predicted = torch.max(category_logits, 1)
            _, urg_predicted = torch.max(urgency_logits, 1)
            
            train_cat_correct += (cat_predicted == category_labels).sum().item()
            train_urg_correct += (urg_predicted == urgency_labels).sum().item()
            train_total += category_labels.size(0)
            
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'cat_acc': train_cat_correct / train_total,
                'urg_acc': train_urg_correct / train_total
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_cat_preds = []
        val_cat_true = []
        val_urg_preds = []
        val_urg_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                category_labels = batch['category_label'].to(device)
                urgency_labels = batch['urgency_label'].to(device)
                
                category_logits, urgency_logits = model(input_ids, attention_mask)
                
                cat_loss = category_criterion(category_logits, category_labels)
                urg_loss = urgency_criterion(urgency_logits, urgency_labels)
                
                val_loss += (cat_loss + urg_loss).item()
                
                _, cat_predicted = torch.max(category_logits, 1)
                _, urg_predicted = torch.max(urgency_logits, 1)
                
                val_cat_preds.extend(cat_predicted.cpu().numpy())
                val_cat_true.extend(category_labels.cpu().numpy())
                val_urg_preds.extend(urg_predicted.cpu().numpy())
                val_urg_true.extend(urgency_labels.cpu().numpy())
        
        # Calculate metrics
        cat_accuracy = accuracy_score(val_cat_true, val_cat_preds)
        cat_f1 = f1_score(val_cat_true, val_cat_preds, average='weighted')
        urg_accuracy = accuracy_score(val_urg_true, val_urg_preds)
        urg_f1 = f1_score(val_urg_true, val_urg_preds, average='weighted')
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Category - Accuracy: {cat_accuracy:.4f}, F1: {cat_f1:.4f}")
        print(f"Urgency - Accuracy: {urg_accuracy:.4f}, F1: {urg_f1:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
            # Save metrics
            metrics = {
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'category_accuracy': cat_accuracy,
                'category_f1': cat_f1,
                'urgency_accuracy': urg_accuracy,
                'urgency_f1': urg_f1
            }
            
            with open(model_save_path.replace('.pt', '_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
    
    print("\nTraining completed!")
    return model

if __name__ == "__main__":
    # Example usage
    train_data_path = "./data/training_tickets.csv"
    model_save_path = "./models/saved/bert_classifier.pt"
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train model
    train_model(train_data_path, model_save_path, epochs=10, batch_size=16)