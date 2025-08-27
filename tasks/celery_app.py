"""
Celery application configuration and tasks
"""
from celery import Celery
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Create Celery app with explicit configuration
app = Celery('tasks')

# Load configuration
app.config_from_object('app.celery_config')

# Auto-discover tasks
app.autodiscover_tasks(['tasks'])

# Import after app creation to avoid circular imports
from app.database import get_connection, save_ticket
from models.bert_classifier import BertTicketClassifier
from utils.notifications import NotificationManager
import logging

logger = logging.getLogger(__name__)

# Initialize classifier globally
classifier = None

def get_classifier():
    """Get or initialize the classifier"""
    global classifier
    if classifier is None:
        try:
            classifier = BertTicketClassifier()
            classifier.load_model()
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            classifier = None
    return classifier

@app.task(name='tasks.classify_and_notify')
def classify_and_notify(ticket_id):
    """Classify ticket and send notifications"""
    try:
        logger.info(f"Processing ticket {ticket_id}")
        
        # Get ticket from database
        connection = get_connection()
        if not connection:
            return False
            
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
        SELECT id, title, description, customer_email, category, urgency
        FROM tickets
        WHERE id = %s
        """, (ticket_id,))
        
        ticket = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not ticket:
            logger.error(f"Ticket {ticket_id} not found")
            return False
        
        # Send notification
        notification_manager = NotificationManager()
        notification_manager.send_customer_notification(
            ticket_id=ticket['id'],
            email=ticket['customer_email'],
            category=ticket['category'],
            urgency=ticket['urgency']
        )
        
        logger.info(f"Notification sent for ticket {ticket_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing ticket {ticket_id}: {str(e)}")
        return False

@app.task(name='tasks.process_batch_tickets')
def process_batch_tickets(ticket_ids):
    """Process multiple tickets in batch"""
    try:
        logger.info(f"Processing batch of {len(ticket_ids)} tickets")
        
        # Get tickets and classify them
        connection = get_connection()
        if not connection:
            return False
            
        cursor = connection.cursor(dictionary=True)
        
        for ticket_id in ticket_ids:
            # Get ticket
            cursor.execute("""
            SELECT id, title, description, customer_email
            FROM tickets
            WHERE id = %s
            """, (ticket_id,))
            
            ticket = cursor.fetchone()
            if ticket:
                # Classify if needed
                classifier = get_classifier()
                if classifier and ticket.get('category') == 'Pending':
                    text = f"{ticket['title']} {ticket['description']}"
                    category, urgency, _ = classifier.predict(text)
                    
                    # Update ticket
                    cursor.execute("""
                    UPDATE tickets
                    SET category = %s, urgency = %s
                    WHERE id = %s
                    """, (category, urgency, ticket_id))
                
                # Send notification
                classify_and_notify.delay(ticket_id)
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Batch processing completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return False

@app.task(name='tasks.send_notification')
def send_notification(ticket_id, category, urgency, customer_email=None):
    """Send notifications based on ticket urgency"""
    try:
        logger.info(f"Sending notification for ticket {ticket_id}")
        
        notification_manager = NotificationManager()
        
        # Send customer notification
        if customer_email:
            notification_manager.send_customer_notification(
                ticket_id=ticket_id,
                email=customer_email,
                category=category,
                urgency=urgency
            )
        
        # Send urgent alerts for high priority
        if urgency in ['Critical', 'High']:
            notification_manager.send_urgent_alert(
                ticket_id=ticket_id,
                category=category,
                urgency=urgency
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        return False

@app.task(name='tasks.route_to_team')
def route_to_team(ticket_id, category, urgency):
    """Route ticket to appropriate team based on category"""
    try:
        logger.info(f"Routing ticket {ticket_id} to {category} team")
        
        # Team routing logic here
        team_mapping = {
            'Technical': 'tech-support@company.com',
            'Billing': 'billing@company.com',
            'Feedback': 'feedback@company.com',
            'General': 'support@company.com',
            'Account': 'account-support@company.com'
        }
        
        team_email = team_mapping.get(category, 'support@company.com')
        
        # Update assignment
        connection = get_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("""
            UPDATE tickets
            SET assigned_to = %s
            WHERE id = %s
            """, (team_email, ticket_id))
            connection.commit()
            cursor.close()
            connection.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error routing ticket: {str(e)}")
        return False

@app.task(name='tasks.cleanup_old_tickets')
def cleanup_old_tickets():
    """Periodic task to archive old resolved tickets"""
    try:
        logger.info("Running ticket cleanup task")
        # Implementation here
        return 0
    except Exception as e:
        logger.error(f"Error in cleanup task: {str(e)}")
        return 0