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
        self.from_name = Config.EMAIL_FROM_NAME
        self.enabled = Config.EMAIL_ENABLED
    
    def send_email(self, to_email, subject, body, is_html=False):
        """Send email notification"""
        if not self.enabled:
            logger.info(f"Email disabled. Would send to {to_email}: {subject}")
            return True
            
        try:
            msg = MIMEMultipart()
            msg['From'] = f"{self.from_name} <{self.smtp_user}>"
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
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2c3e50;">Thank you for contacting support!</h2>
                    <p>We have received your ticket with the following details:</p>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <p><strong>Ticket ID:</strong> #{ticket_id}</p>
                        <p><strong>Category:</strong> {category}</p>
                        <p><strong>Priority:</strong> {urgency}</p>
                    </div>
                    <h3>Expected Response Time:</h3>
                    <ul>
                        <li>Critical: Within 4 hours</li>
                        <li>High: Within 24 hours</li>
                        <li>Medium: Within 48 hours</li>
                        <li>Low: Within 72 hours</li>
                    </ul>
                    <p>You will receive updates as we work on resolving your issue.</p>
                    <hr style="border: 1px solid #e0e0e0; margin: 30px 0;">
                    <p style="color: #666; font-size: 14px;">
                        Best regards,<br>
                        AI Support Team
                    </p>
                </div>
            </body>
        </html>
        """
        
        return self.send_email(email, subject, body, is_html=True)
    
    def send_urgent_alert(self, ticket_id, category, urgency):
        """Send alert for urgent tickets"""
        admin_email = "admin@company.com"  # Configure this in production
        
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
            </body>
        </html>
        """
        
        return self.send_email(admin_email, subject, body, is_html=True)