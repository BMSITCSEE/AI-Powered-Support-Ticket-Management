"""
Notification management utilities
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import Config
import logging

logger = logging.getLogger(__name__)

class NotificationManager:
    """Manage various types of notifications"""
    
    def __init__(self):
        self.smtp_host = Config.EMAIL_HOST
        self.smtp_port = Config.EMAIL_PORT
        self.smtp_user = Config.EMAIL_USER
        self.smtp_password = Config.EMAIL_PASSWORD
    
    def send_email(self, to_email, subject, body, is_html=False):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    def send_customer_notification(self, ticket_id, email, category, urgency):
        """Send notification to customer about ticket submission"""
        subject = f"Ticket #{ticket_id} - Received"
        
        body = f"""
        <html>
            <body>
                <h2>Thank you for contacting support!</h2>
                <p>We have received your ticket with the following details:</p>
                <ul>
                    <li><strong>Ticket ID:</strong> #{ticket_id}</li>
                    <li><strong>Category:</strong> {category}</li>
                    <li><strong>Priority:</strong> {urgency}</li>
                </ul>
                <p>Expected response time based on priority:</p>
                <ul>
                    <li>Critical: Within 4 hours</li>
                    <li>High: Within 24 hours</li>
                    <li>Medium: Within 48 hours</li>
                    <li>Low: Within 72 hours</li>
                </ul>
                <p>You will receive updates as we work on resolving your issue.</p>
                <br>
                <p>Best regards,<br>Support Team</p>
            </body>
        </html>
        """
        
        return self.send_email(email, subject, body, is_html=True)
    
    def send_urgent_alert(self, ticket_id, category, urgency):
        """Send alert for urgent tickets"""
        # In production, this would send to a distribution list or Slack
        admin_email = "admin@company.com"
        
        subject = f"🚨 URGENT: {urgency} Priority Ticket #{ticket_id}"
        
        body = f"""
        <html>
            <body>
                <h2 style="color: red;">Urgent Ticket Alert</h2>
                <p>A new {urgency} priority ticket requires immediate attention:</p>
                <ul>
                    <li><strong>Ticket ID:</strong> #{ticket_id}</li>
                    <li><strong>Category:</strong> {category}</li>
                    <li><strong>Priority:</strong> {urgency}</li>
                </ul>
                <p>Please assign this ticket to the appropriate team member immediately.</p>
                <br>
                <p><a href="http://localhost:8501/admin_dashboard">View in Dashboard</a></p>
            </body>
        </html>
        """
        
        return self.send_email(admin_email, subject, body, is_html=True)
    
    def send_team_assignment(self, ticket_id, team_email, category, urgency):
        """Send notification when ticket is assigned to team"""
        subject = f"New Ticket Assignment - #{ticket_id}"
        
        body = f"""
        <html>
            <body>
                <h2>New Ticket Assignment</h2>
                <p>A new ticket has been assigned to your team:</p>
                <ul>
                    <li><strong>Ticket ID:</strong> #{ticket_id}</li>
                    <li><strong>Category:</strong> {category}</li>
                    <li><strong>Priority:</strong> {urgency}</li>
                </ul>
                <p>Please review and respond within the SLA timeframe.</p>
                <br>
                <p><a href="http://localhost:8501/admin_dashboard">View Ticket</a></p>
            </body>
        </html>
        """
        
        return self.send_email(team_email, subject, body, is_html=True)
    
    def send_escalation_alert(self, ticket):
        """Send escalation alert for overdue tickets"""
        manager_email = "manager@company.com"
        
        subject = f"⚠️ ESCALATION: Ticket #{ticket['id']} Overdue"
        
        body = f"""
        <html>
            <body>
                <h2 style="color: orange;">Ticket Escalation Alert</h2>
                <p>The following ticket has exceeded its SLA and requires escalation:</p>
                <ul>
                    <li><strong>Ticket ID:</strong> #{ticket['id']}</li>
                    <li><strong>Title:</strong> {ticket['title']}</li>
                    <li><strong>Category:</strong> {ticket['category']}</li>
                    <li><strong>Priority:</strong> {ticket['urgency']}</li>
                    <li><strong>Created:</strong> {ticket['created_at']}</li>
                    <li><strong>Assigned To:</strong> {ticket.get('assigned_to', 'Unassigned')}</li>
                </ul>
                <p>Immediate action required to meet customer expectations.</p>
            </body>
        </html>
        """
        
        return self.send_email(manager_email, subject, body, is_html=True)
    
    def send_daily_summary(self, stats, category_stats):
        """Send daily summary to admins"""
        admin_email = "admin@company.com"
        
        subject = f"Daily Ticket Summary - {stats['total']} New Tickets"
        
        category_rows = "".join([
            f"<tr><td>{cat['category']}</td><td>{cat['count']}</td></tr>"
            for cat in category_stats
        ])
        
        body = f"""
        <html>
            <body>
                <h2>Daily Ticket Summary</h2>
                <h3>Overall Statistics</h3>
                <ul>
                    <li>Total New Tickets: {stats['total']}</li>
                    <li>Critical Tickets: {stats['critical']}</li>
                    <li>High Priority Tickets: {stats['high']}</li>
                    <li>Resolved Today: {stats['resolved']}</li>
                </ul>
                
                <h3>By Category</h3>
                <table border="1" cellpadding="5">
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                    </tr>
                    {category_rows}
                </table>
                
                <br>
                <p><a href="http://localhost:8501/admin_dashboard">View Full Dashboard</a></p>
            </body>
        </html>
        """
        
        return self.send_email(admin_email, subject, body, is_html=True)