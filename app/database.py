"""
Database models and operations
"""
import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime
from app.config import Config
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Connection pool for better performance
connection_pool = None

def init_connection_pool():
    """Initialize connection pool"""
    global connection_pool
    try:
        connection_pool = pooling.MySQLConnectionPool(
            pool_name="ticket_pool",
            pool_size=5,
            pool_reset_session=True,
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE
        )
        logger.info("Connection pool created successfully")
    except Error as e:
        logger.error(f"Error creating connection pool: {e}")

def get_connection():
    """Get connection from pool"""
    global connection_pool
    if connection_pool is None:
        init_connection_pool()
    
    try:
        return connection_pool.get_connection()
    except Error as e:
        logger.error(f"Error getting connection: {e}")
        # Fallback to direct connection
        try:
            return mysql.connector.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DATABASE
            )
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return None

def init_db():
    """Initialize database tables"""
    connection = get_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    
    try:
        # Create tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            category VARCHAR(50),
            urgency VARCHAR(20),
            status VARCHAR(20) DEFAULT 'Open',
            customer_email VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP NULL,
            assigned_to VARCHAR(255),
            resolution_notes TEXT,
            satisfaction_rating INT,
            INDEX idx_category (category),
            INDEX idx_urgency (urgency),
            INDEX idx_status (status),
            INDEX idx_created (created_at),
            INDEX idx_email (customer_email)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            role VARCHAR(20) DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL,
            is_active BOOLEAN DEFAULT TRUE,
            reset_token VARCHAR(255),
            reset_token_expiry TIMESTAMP NULL,
            INDEX idx_username (username),
            INDEX idx_email (email)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # Create notifications table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticket_id INT,
            type VARCHAR(50),
            recipient VARCHAR(255),
            subject VARCHAR(255),
            message TEXT,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'Pending',
            error_message TEXT,
            retry_count INT DEFAULT 0,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE,
            INDEX idx_ticket (ticket_id),
            INDEX idx_status (status),
            INDEX idx_sent (sent_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # Create ticket_history table for audit trail
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticket_id INT,
            user_id INT,
            action VARCHAR(50),
            old_value TEXT,
            new_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id),
            INDEX idx_ticket_history (ticket_id),
            INDEX idx_created_history (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # Create user_permissions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_permissions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            permission VARCHAR(50),
            granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            granted_by INT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (granted_by) REFERENCES users(id),
            UNIQUE KEY unique_user_permission (user_id, permission)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # Create default admin user if not exists
        cursor.execute("""
        INSERT IGNORE INTO users (username, password_hash, email, role)
        VALUES ('admin', '$2b$12$YourHashedPasswordHere', 'admin@example.com', 'admin')
        """)
        
        connection.commit()
        logger.info("Database initialized successfully")
        return True
        
    except Error as e:
        logger.error(f"Error initializing database: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def save_ticket(title, description, category, urgency, customer_email):
    """Save a new ticket to database"""
    connection = get_connection()
    if not connection:
        return None
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("""
        INSERT INTO tickets (title, description, category, urgency, customer_email)
        VALUES (%s, %s, %s, %s, %s)
        """, (title, description, category, urgency, customer_email))
        
        connection.commit()
        ticket_id = cursor.lastrowid
        
        # Log the action
        log_ticket_action(ticket_id, None, 'created', None, f'Category: {category}, Urgency: {urgency}')
        
        return ticket_id
    except Error as e:
        logger.error(f"Error saving ticket: {e}")
        connection.rollback()
        return None
    finally:
        cursor.close()
        connection.close()

def get_all_tickets(filters=None):
    """Retrieve all tickets with optional filters"""
    connection = get_connection()
    if not connection:
        return pd.DataFrame()
    
    query = """
    SELECT id, title, description, category, urgency, status, 
           customer_email, created_at, updated_at, assigned_to,
           TIMESTAMPDIFF(HOUR, created_at, IFNULL(resolved_at, NOW())) as resolution_hours
    FROM tickets
    WHERE 1=1
    """
    
    params = []
    
    if filters:
        if filters.get('status'):
            query += " AND status = %s"
            params.append(filters['status'])
        if filters.get('category'):
            query += " AND category = %s"
            params.append(filters['category'])
        if filters.get('urgency'):
            query += " AND urgency = %s"
            params.append(filters['urgency'])
        if filters.get('date_from'):
            query += " AND created_at >= %s"
            params.append(filters['date_from'])
        if filters.get('date_to'):
            query += " AND created_at <= %s"
            params.append(filters['date_to'])
    
    query += " ORDER BY created_at DESC"
    
    try:
        df = pd.read_sql(query, connection, params=params)
        return df
    except Error as e:
        logger.error(f"Error fetching tickets: {e}")
        return pd.DataFrame()
    finally:
        connection.close()

def get_ticket_by_id(ticket_id):
    """Get single ticket by ID"""
    connection = get_connection()
    if not connection:
        return None
    
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("""
    SELECT * FROM tickets WHERE id = %s
    """, (ticket_id,))
    
    ticket = cursor.fetchone()
    cursor.close()
    connection.close()
    
    return ticket

def get_ticket_stats():
    """Get ticket statistics for dashboard"""
    connection = get_connection()
    if not connection:
        return {}
    
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Total tickets
        cursor.execute("SELECT COUNT(*) as total FROM tickets")
        total = cursor.fetchone()['total']
        
        # By category
        cursor.execute("""
        SELECT category, COUNT(*) as count 
        FROM tickets 
        GROUP BY category
        ORDER BY count DESC
        """)
        by_category = cursor.fetchall()
        
        # By urgency
        cursor.execute("""
        SELECT urgency, COUNT(*) as count 
        FROM tickets 
        GROUP BY urgency
        ORDER BY FIELD(urgency, 'Critical', 'High', 'Medium', 'Low')
        """)
        by_urgency = cursor.fetchall()
        
        # By status
        cursor.execute("""
        SELECT status, COUNT(*) as count 
        FROM tickets 
        GROUP BY status
        ORDER BY count DESC
        """)
        by_status = cursor.fetchall()
        
        # Average resolution time
        cursor.execute("""
        SELECT 
            AVG(TIMESTAMPDIFF(HOUR, created_at, resolved_at)) as avg_resolution_hours,
            urgency
        FROM tickets 
        WHERE resolved_at IS NOT NULL
        GROUP BY urgency
        """)
        resolution_times = cursor.fetchall()
        
        return {
            'total': total,
            'by_category': by_category,
            'by_urgency': by_urgency,
            'by_status': by_status,
            'resolution_times': resolution_times
        }
    except Error as e:
        logger.error(f"Error getting stats: {e}")
        return {}
    finally:
        cursor.close()
        connection.close()

def update_ticket_status(ticket_id, status, assigned_to=None, resolution_notes=None):
    """Update ticket status and assignment"""
    connection = get_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()

        try:
        # Get current values for history
        cursor.execute("SELECT status, assigned_to FROM tickets WHERE id = %s", (ticket_id,))
        current = cursor.fetchone()
        
        # Build update query
        update_fields = ["status = %s"]
        params = [status]
        
        if assigned_to is not None:
            update_fields.append("assigned_to = %s")
            params.append(assigned_to)
        
        if resolution_notes is not None:
            update_fields.append("resolution_notes = %s")
            params.append(resolution_notes)
        
        if status == 'Resolved':
            update_fields.append("resolved_at = NOW()")
        
        params.append(ticket_id)
        
        query = f"UPDATE tickets SET {', '.join(update_fields)} WHERE id = %s"
        cursor.execute(query, params)
        
        # Log the changes
        if current:
            old_status, old_assigned = current
            if old_status != status:
                log_ticket_action(ticket_id, None, 'status_changed', old_status, status)
            if assigned_to and old_assigned != assigned_to:
                log_ticket_action(ticket_id, None, 'assigned', old_assigned, assigned_to)
        
        connection.commit()
        return True
    except Error as e:
        logger.error(f"Error updating ticket: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def log_ticket_action(ticket_id, user_id, action, old_value, new_value):
    """Log ticket history for audit trail"""
    connection = get_connection()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("""
        INSERT INTO ticket_history (ticket_id, user_id, action, old_value, new_value)
        VALUES (%s, %s, %s, %s, %s)
        """, (ticket_id, user_id, action, str(old_value), str(new_value)))
        connection.commit()
    except Error as e:
        logger.error(f"Error logging action: {e}")
    finally:
        cursor.close()
        connection.close()

def get_ticket_history(ticket_id):
    """Get ticket history"""
    connection = get_connection()
    if not connection:
        return []
    
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("""
    SELECT th.*, u.username 
    FROM ticket_history th
    LEFT JOIN users u ON th.user_id = u.id
    WHERE th.ticket_id = %s
    ORDER BY th.created_at DESC
    """, (ticket_id,))
    
    history = cursor.fetchall()
    cursor.close()
    connection.close()
    
    return history

def save_notification(ticket_id, type, recipient, subject, message):
    """Save notification record"""
    connection = get_connection()
    if not connection:
        return None
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("""
        INSERT INTO notifications (ticket_id, type, recipient, subject, message, status)
        VALUES (%s, %s, %s, %s, %s, 'Sent')
        """, (ticket_id, type, recipient, subject, message))
        
        connection.commit()
        return cursor.lastrowid
    except Error as e:
        logger.error(f"Error saving notification: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def get_user_tickets(email):
    """Get tickets for a specific user"""
    connection = get_connection()
    if not connection:
        return pd.DataFrame()
    
    query = """
    SELECT id, title, category, urgency, status, created_at, updated_at
    FROM tickets
    WHERE customer_email = %s
    ORDER BY created_at DESC
    """
    
    try:
        df = pd.read_sql(query, connection, params=[email])
        return df
    except Error as e:
        logger.error(f"Error fetching user tickets: {e}")
        return pd.DataFrame()
    finally:
        connection.close()

def get_agent_performance(date_from=None, date_to=None):
    """Get agent performance metrics"""
    connection = get_connection()
    if not connection:
        return pd.DataFrame()
    
    query = """
    SELECT 
        assigned_to as agent,
        COUNT(*) as total_tickets,
        SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved_tickets,
        AVG(CASE WHEN resolved_at IS NOT NULL 
            THEN TIMESTAMPDIFF(HOUR, created_at, resolved_at) 
            ELSE NULL END) as avg_resolution_hours,
        AVG(satisfaction_rating) as avg_satisfaction
    FROM tickets
    WHERE assigned_to IS NOT NULL
    """
    
    params = []
    if date_from:
        query += " AND created_at >= %s"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= %s"
        params.append(date_to)
    
    query += " GROUP BY assigned_to ORDER BY resolved_tickets DESC"
    
    try:
        df = pd.read_sql(query, connection, params=params)
        return df
    except Error as e:
        logger.error(f"Error fetching agent performance: {e}")
        return pd.DataFrame()
    finally:
        connection.close()

def update_satisfaction_rating(ticket_id, rating):
    """Update ticket satisfaction rating"""
    connection = get_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("""
        UPDATE tickets 
        SET satisfaction_rating = %s 
        WHERE id = %s
        """, (rating, ticket_id))
        
        connection.commit()
        return True
    except Error as e:
        logger.error(f"Error updating rating: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def cleanup_old_data(days=90):
    """Archive or delete old data"""
    connection = get_connection()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        # Archive old tickets
        cursor.execute("""
        UPDATE tickets 
        SET status = 'Archived' 
        WHERE status = 'Closed' 
        AND updated_at < DATE_SUB(NOW(), INTERVAL %s DAY)
        """, (days,))
        
        archived_count = cursor.rowcount
        
        # Clean old notifications
        cursor.execute("""
        DELETE FROM notifications 
        WHERE sent_at < DATE_SUB(NOW(), INTERVAL %s DAY)
        """, (days,))
        
        deleted_notifications = cursor.rowcount
        
        connection.commit()
        logger.info(f"Archived {archived_count} tickets, deleted {deleted_notifications} old notifications")
        
    except Error as e:
        logger.error(f"Error in cleanup: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()
    