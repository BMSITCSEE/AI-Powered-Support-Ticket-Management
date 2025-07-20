"""
Celery application configuration and tasks
"""
from celery import Celery
from app.config import Config
from app.database import get_connection, save_ticket
from models.bert_classifier import BertTicketClassifier
from utils.notifications import NotificationManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
app = Celery('tasks', broker=Config.CELERY_BROKER_URL)

# Celery configuration
app.conf.update(
    result_backend=Config.CELERY_RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'tasks.classify_and_notify': {'queue': 'high_priority'},
        'tasks.process_batch_tickets': {'queue': 'batch_processing'},
        'tasks.send_notification': {'queue': 'notifications'},
    }
)

# Initialize classifier globally to avoid reloading
classifier = None

def get_classifier():
    """Get or initialize the classifier"""
    global classifier
    if classifier is None:
        classifier = BertTicketClassifier()
        classifier.load_model()
    return classifier

@app.task(name='tasks.classify_and_notify')
def classify_and_notify(ticket_id):
    """Classify ticket and send notifications"""
    try:
        logger.info(f"Processing ticket {ticket_id}")
        
        # Get ticket from database
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
        SELECT id, title, description, customer_email, category, urgency
        FROM tickets
        WHERE id = %s
        """, (ticket_id,))
        
        ticket = cursor.fetchone()
        cursor.close()
        
        if not ticket:
            logger.error(f"Ticket {ticket_id} not found")
            return False
        
        # If not already classified
        if ticket['category'] == 'Pending' or ticket['urgency'] == 'Pending':
            # Classify ticket
            classifier = get_classifier()
            text = f"{ticket['title']} {ticket['description']}"
            category, urgency, confidence = classifier.predict(text)
            
            # Update ticket with classification
            cursor = connection.cursor()
            cursor.execute("""
            UPDATE tickets
            SET category = %s, urgency = %s
            WHERE id = %s
            """, (category, urgency, ticket_id))
            connection.commit()
            cursor.close()
            
            logger.info(f"Ticket {ticket_id} classified - Category: {category}, Urgency: {urgency}")
        else:
            category = ticket['category']
            urgency = ticket['urgency']
        
        connection.close()
        
        # Send notifications based on urgency
        send_notification.delay(
            ticket_id=ticket_id,
            category=category,
            urgency=urgency,
            customer_email=ticket['customer_email']
        )
        
        # Route to appropriate team
        route_to_team.delay(ticket_id, category, urgency)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing ticket {ticket_id}: {str(e)}")
        return False

@app.task(name='tasks.process_batch_tickets')
def process_batch_tickets(ticket_ids):
    """Process multiple tickets in batch"""
    try:
        logger.info(f"Processing batch of {len(ticket_ids)} tickets")
        
        # Get tickets from database
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        
        placeholders = ','.join(['%s'] * len(ticket_ids))
        cursor.execute(f"""
        SELECT id, title, description
        FROM tickets
        WHERE id IN ({placeholders})
        """, ticket_ids)
        
        tickets = cursor.fetchall()
        cursor.close()
        
        # Prepare texts for batch classification
        texts = [f"{t['title']} {t['description']}" for t in tickets]
        
        # Classify in batch
        classifier = get_classifier()
        results = classifier.predict_batch(texts)
        
        # Update tickets with classifications
        cursor = connection.cursor()
        for ticket, result in zip(tickets, results):
            cursor.execute("""
            UPDATE tickets
            SET category = %s, urgency = %s
            WHERE id = %s
            """, (result['category'], result['urgency'], ticket['id']))
            
            # Queue notification task
            send_notification.delay(
                ticket_id=ticket['id'],
                category=result['category'],
                urgency=result['urgency']
            )
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Batch processing completed for {len(ticket_ids)} tickets")
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
        
        # Send internal notifications for high priority tickets
        if urgency in ['Critical', 'High']:
            notification_manager.send_urgent_alert(
                ticket_id=ticket_id,
                category=category,
                urgency=urgency
            )
        
        # Log notification in database
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
        INSERT INTO notifications (ticket_id, type, recipient, message, status)
        VALUES (%s, %s, %s, %s, 'Sent')
        """, (
            ticket_id,
            f"{urgency} Alert",
            customer_email or 'Internal',
            f"Notification sent for {category} ticket with {urgency} priority"
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        return False

@app.task(name='tasks.route_to_team')
def route_to_team(ticket_id, category, urgency):
    """Route ticket to appropriate team based on category"""
    try:
        logger.info(f"Routing ticket {ticket_id} to {category} team")
        
        # Define team routing rules
        team_mapping = {
            'Technical': 'tech-support@company.com',
            'Billing': 'billing@company.com',
            'Feedback': 'feedback@company.com',
            'General': 'support@company.com',
            'Account': 'account-support@company.com'
        }
        
        team_email = team_mapping.get(category, 'support@company.com')
        
        # Update ticket assignment (simplified for demo)
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
        UPDATE tickets
        SET assigned_to = %s
        WHERE id = %s
        """, (team_email, ticket_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        # Send notification to team
        notification_manager = NotificationManager()
        notification_manager.send_team_assignment(
            ticket_id=ticket_id,
            team_email=team_email,
            category=category,
            urgency=urgency
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error routing ticket: {str(e)}")
        return False

@app.task(name='tasks.cleanup_old_tickets')
def cleanup_old_tickets():
    """Periodic task to archive old resolved tickets"""
    try:
        logger.info("Running ticket cleanup task")
        
        connection = get_connection()
        cursor = connection.cursor()
        
        # Archive tickets older than 90 days
        cursor.execute("""
        UPDATE tickets
        SET status = 'Archived'
        WHERE status = 'Closed'
        AND updated_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
        """)
        
        affected_rows = cursor.rowcount
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Archived {affected_rows} old tickets")
        return affected_rows
        
    except Exception as e:
        logger.error(f"Error in cleanup task: {str(e)}")
        return 0

# Celery beat schedule for periodic tasks
app.conf.beat_schedule = {
    'cleanup-old-tickets': {
        'task': 'tasks.cleanup_old_tickets',
        'schedule': 86400.0,  # Run daily
    },
}