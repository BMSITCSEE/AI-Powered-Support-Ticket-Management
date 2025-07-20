"""
Metrics calculation utilities
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def calculate_metrics(y_true, y_pred, labels=None):
    """Calculate classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }
    
    # Per-class metrics
    if labels:
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels)
        per_class_precision = precision_score(y_true, y_pred, average=None, labels=labels)
        per_class_recall = recall_score(y_true, y_pred, average=None, labels=labels)
        
        metrics['per_class'] = {}
        for i, label in enumerate(labels):
            metrics['per_class'][label] = {
                'f1_score': per_class_f1[i],
                'precision': per_class_precision[i],
                'recall': per_class_recall[i]
            }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels):
    """Generate confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Convert plot to base64 string for display in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return plot_data

def calculate_response_time_metrics(tickets_df):
    """Calculate response time metrics"""
    if 'created_at' not in tickets_df.columns or 'resolved_at' not in tickets_df.columns:
        return {}
    
    # Calculate resolution time
    tickets_df['resolution_time'] = (
        tickets_df['resolved_at'] - tickets_df['created_at']
    ).dt.total_seconds() / 3600  # Convert to hours
    
    metrics = {
        'avg_resolution_time': tickets_df['resolution_time'].mean(),
        'median_resolution_time': tickets_df['resolution_time'].median(),
        'by_urgency': tickets_df.groupby('urgency')['resolution_time'].agg(['mean', 'median']).to_dict(),
        'by_category': tickets_df.groupby('category')['resolution_time'].agg(['mean', 'median']).to_dict()
    }
    
    return metrics

def calculate_sla_compliance(tickets_df):
    """Calculate SLA compliance rates"""
    sla_limits = {
        'Critical': 4,    # hours
        'High': 24,
        'Medium': 48,
        'Low': 72
    }
    
    compliance_data = []
    
    for urgency, limit in sla_limits.items():
        urgency_tickets = tickets_df[tickets_df['urgency'] == urgency]
        if not urgency_tickets.empty:
            within_sla = urgency_tickets['resolution_time'] <= limit
            compliance_rate = within_sla.sum() / len(urgency_tickets)
            compliance_data.append({
                'urgency': urgency,
                'sla_limit': limit,
                'compliance_rate': compliance_rate,
                'total_tickets': len(urgency_tickets)
            })
    
    return compliance_data