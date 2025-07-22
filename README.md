# 🎫 AI-Powered Support Ticket Management System

An intelligent support ticket management system that uses BERT for automatic classification and prioritization of customer support tickets.

## 🚀 Features

- **AI-Powered Classification**: Automatically categorizes tickets using fine-tuned BERT
- **Smart Prioritization**: Assigns urgency levels based on ticket content
- **Real-time Processing**: Asynchronous ticket processing with Celery
- **Intuitive Dashboard**: Streamlit-based UI for ticket submission and management
- **Batch Processing**: Upload multiple tickets via CSV
- **Automated Notifications**: Email alerts based on ticket priority
- **Performance Metrics**: Track classification accuracy and response times

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: PyTorch + Hugging Face Transformers
- **NLP Model**: BERT (fine-tuned)
- **Database**: MySQL
- **Message Queue**: Redis
- **Task Queue**: Celery
- **Containerization**: Docker & Docker Compose

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- 4GB+ RAM recommended
- NVIDIA GPU (optional, for faster inference)

## 🔧 Installation

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-support-ticket-manager.git
cd ai-support-ticket-manager