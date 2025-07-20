"""
Additional notification-related tasks
"""
from tasks.celery_app import app
from utils.notifications import NotificationManager
from app.database import get_connection
import logging

logger = logging.getLogger(__name__)

@app.task(name='tasks.send_daily_summary')
def send_daily_summary():
    """Send daily summary of tickets to admins"""
    try:
        logger.info("Generating daily summary")
        
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get today's statistics
        cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN urgency = 'Critical' THEN 1 ELSE 0 END) as critical,
            SUM(CASE WHEN urgency = 'High' THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved
        FROM tickets
        WHERE DATE(created_at) = CURDATE()
        """)
        
        stats = cursor.fetchone()
        
        # Get category breakdown
        cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM tickets
        WHERE DATE(created_at) = CURDATE()
        GROUP BY category
        """)
        
        category_stats = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        # Send summary
        notification_manager = NotificationManager()
        notification_manager.send_daily_summary(stats, category_stats)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating daily summary: {str(e)}")
        return False

@app.task(name='tasks.escalate_unresolved_tickets')
def escalate_unresolved_tickets():
    """Escalate tickets that haven't been resolved within SLA"""
    try:
        logger.info("Checking for tickets to escalate")
        
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Find tickets exceeding SLA
        cursor.execute("""
        SELECT id, title, category, urgency, created_at, assigned_to
        FROM tickets
        WHERE status NOT IN ('Resolved', 'Closed', 'Archived')
        AND (
            (urgency = 'Critical' AND created_at < DATE_SUB(NOW(), INTERVAL 4 HOUR)) OR
            (urgency = 'High' AND created_at < DATE_SUB(NOW(), INTERVAL 24 HOUR)) OR
            (urgency = 'Medium' AND created_at < DATE_SUB(NOW(), INTERVAL 48 HOUR))
        )
        """)
        
        tickets_to_escalate = cursor.fetchall()
        
        notification_manager = NotificationManager()
        
        for ticket in tickets_to_escalate:
            # Send escalation notification
            notification_manager.send_escalation_alert(ticket)
            
            # Update ticket status
            cursor.execute("""
            UPDATE tickets
            SET status = 'Escalated'
            WHERE id = %s
            """, (ticket['id'],))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Escalated {len(tickets_to_escalate)} tickets")
        return len(tickets_to_escalate)
        
    except Exception as e:
        logger.error(f"Error in escalation task: {str(e)}")
        return 0

@app.task(name='tasks.send_resolution_feedback_request')
def send_resolution_feedback_request(ticket_id):
    """Send feedback request after ticket resolution"""
    try:
        logger.info(f"Sending feedback request for ticket {ticket_id}")
        
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
        SELECT id, title, customer_email, resolved_at
        FROM tickets
        WHERE id = %s AND status = 'Resolved'
        """, (ticket_id,))
        
        ticket = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if ticket:
            notification_manager = NotificationManager()
            notification_manager.send_feedback_request(
                ticket_id=ticket['id'],
                email=ticket['customer_email'],
                title=ticket['title']
            )
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error sending feedback request: {str(e)}")
        return False

@app.task(name='tasks.send_weekly_performance_report')
def send_weekly_performance_report():
    """Generate and send weekly performance metrics"""
    try:
        logger.info("Generating weekly performance report")
        
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get weekly metrics
        cursor.execute("""
        SELECT 
            COUNT(*) as total_tickets,
            AVG(TIMESTAMPDIFF(HOUR, created_at, updated_at)) as avg_resolution_time,
            SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved_tickets,
            SUM(CASE WHEN status = 'Escalated' THEN 1 ELSE 0 END) as escalated_tickets
        FROM tickets
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """)
        
        weekly_stats = cursor.fetchone()
        
        # Get agent performance
        cursor.execute("""
        SELECT 
            assigned_to,
            COUNT(*) as tickets_handled,
            AVG(TIMESTAMPDIFF(HOUR, created_at, updated_at)) as avg_handle_time
        FROM tickets
        WHERE assigned_to IS NOT NULL
        AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY assigned_to
        ORDER BY tickets_handled DESC
        """)
        
        agent_stats = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        # Send report
        notification_manager = NotificationManager()
        notification_manager.send_weekly_report(weekly_stats, agent_stats)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating weekly report: {str(e)}")
        return False

# Additional periodic tasks for Celery beat
app.conf.beat_schedule.update({
    'send-daily-summary': {
        'task': 'tasks.send_daily_summary',
        'schedule': 86400.0,  # Daily
    },
    'escalate-tickets': {
        'task': 'tasks.escalate_unresolved_tickets',
        'schedule': 3600.0,  # Every hour
    },
    'weekly-performance-report': {
        'task': 'tasks.send_weekly_performance_report',
        'schedule': 604800.0,  # Weekly
    }
})