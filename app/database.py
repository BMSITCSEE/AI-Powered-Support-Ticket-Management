"""
Database module for PostgreSQL/Neon support
Migrated from MySQL to PostgreSQL with Neon cloud database support
"""

import os
import logging
from typing import Optional, Dict, Any, List, Generator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Float, Boolean, ForeignKey, Enum, Index, text
)
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Enums for ticket system
class TicketStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    PROCESSED = "processed"
    FAILED = "failed"

class TicketCategory(str, enum.Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    FEEDBACK = "feedback"
    OTHER = "other"

class UrgencyLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DatabaseConfig:
    """Database configuration handler for PostgreSQL/Neon"""
    
    @staticmethod
    def get_database_url() -> str:
        """
        Get PostgreSQL database URL with Neon support
        Supports both standard PostgreSQL and Neon connection strings
        """
        # Check for direct DATABASE_URL (common in cloud deployments)
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            # Handle Neon's postgres:// format
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
            
            # Add SSL mode for Neon if not present
            if 'neon.tech' in database_url and 'sslmode' not in database_url:
                separator = '&' if '?' in database_url else '?'
                database_url += f'{separator}sslmode=require'
            
            return database_url
        
        # Fallback to component-based configuration
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        database = os.getenv('POSTGRES_DB', 'ticket_system')
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', '')
        
        if password:
            return f'postgresql://{user}:{password}@{host}:{port}/{database}'
        return f'postgresql://{user}@{host}:{port}/{database}'


class Database:
    """PostgreSQL/Neon database manager with connection pooling"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or DatabaseConfig.get_database_url()
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self):
        """Lazy-load database engine with appropriate pooling"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    def _create_engine(self):
        """Create SQLAlchemy engine optimized for PostgreSQL/Neon"""
        
        # Detect if using Neon
        is_neon = 'neon.tech' in self.database_url
        
        # Configure engine based on environment
        if is_neon:
            # Neon optimized settings
            engine_config = {
                'pool_pre_ping': True,  # Verify connections
                'pool_size': 5,         # Smaller pool for serverless
                'max_overflow': 10,
                'pool_recycle': 300,    # Recycle connections every 5 min
                'connect_args': {
                    'connect_timeout': 30,
                    'options': '-c statement_timeout=30000',  # 30s timeout
                    'sslmode': 'require'
                }
            }
        else:
            # Standard PostgreSQL settings
            engine_config = {
                'pool_pre_ping': True,
                'pool_size': 10,
                'max_overflow': 20,
                'pool_recycle': 3600,
                'connect_args': {
                    'connect_timeout': 10,
                    'options': '-c statement_timeout=60000'  # 60s timeout
                }
            }
        
        # Add echo for debugging in development
        if os.getenv('DEBUG', 'false').lower() == 'true':
            engine_config['echo'] = True
        
        try:
            engine = create_engine(self.database_url, **engine_config)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text('SELECT 1'))
                logger.info(f"Database connected successfully. PostgreSQL version: {result.scalar()}")
            
            return engine
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    @property
    def session_factory(self):
        """Get session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                autoflush=False
            )
        return self._session_factory
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations
        
        Usage:
            with db.session_scope() as session:
                session.add(ticket)
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health and connection"""
        try:
            with self.engine.connect() as conn:
                # PostgreSQL health check query
                result = conn.execute(text("""
                    SELECT 
                        version() as postgres_version,
                        current_database() as database_name,
                        pg_database_size(current_database()) as database_size,
                        count(*) as active_connections
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """))
                
                row = result.first()
                return {
                    'status': 'healthy',
                    'postgres_version': row[0],
                    'database_name': row[1],
                    'database_size_bytes': row[2],
                    'active_connections': row[3],
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def close(self):
        """Close all database connections"""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tickets = relationship("Ticket", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username}>"


class Ticket(Base):
    __tablename__ = 'tickets'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    
    # Classification fields
    category = Column(Enum(TicketCategory), default=TicketCategory.OTHER, index=True)
    urgency = Column(Enum(UrgencyLevel), default=UrgencyLevel.MEDIUM, index=True)
    status = Column(Enum(TicketStatus), default=TicketStatus.PENDING, index=True)
    
    # ML model results
    confidence_score = Column(Float, default=0.0)
    model_version = Column(String(50))
    processing_time_ms = Column(Integer)
    
    # User relationship
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    user_email = Column(String(255))  # Denormalized for quick access
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="tickets")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_status_created', 'status', 'created_at'),
        Index('idx_category_urgency', 'category', 'urgency'),
    )
    
    def __repr__(self):
        return f"<Ticket {self.id}: {self.title[:50]}...>"


# Singleton database instance
_db_instance = None

def get_database() -> Database:
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


# Convenience functions for backward compatibility
def get_db_session() -> Session:
    """Get a new database session"""
    return get_database().get_session()


@contextmanager
def get_db():
    """
    Dependency injection for FastAPI/Streamlit
    
    Usage:
        with get_db() as db:
            tickets = db.query(Ticket).all()
    """
    db = get_database()
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()


def init_database():
    """Initialize database and create tables"""
    db = get_database()
    db.create_tables()
    logger.info("Database initialized successfully")


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute raw SQL query (use sparingly)"""
    db = get_database()
    with db.session_scope() as session:
        result = session.execute(text(query), params or {})
        if result.returns_rows:
            return [dict(row._mapping) for row in result]
        return []


# Helper functions for common operations
def create_ticket(
    title: str, 
    description: str, 
    user_id: Optional[int] = None,
    user_email: Optional[str] = None
) -> Ticket:
    """Create a new ticket"""
    db = get_database()
    with db.session_scope() as session:
        ticket = Ticket(
            title=title,
            description=description,
            user_id=user_id,
            user_email=user_email
        )
        session.add(ticket)
        session.flush()
        ticket_id = ticket.id
    
    return ticket

def save_ticket(
    title: str,
    description: str,
    category: Optional[TicketCategory] = None,
    urgency: Optional[UrgencyLevel] = None,
    status: Optional[TicketStatus] = None,
    user_id: Optional[int] = None,
    user_email: Optional[str] = None,
    confidence_score: Optional[float] = None,
    model_version: Optional[str] = None,
    ticket_id: Optional[int] = None
) -> Ticket:
    """
    Save a ticket (create new or update existing)
    
    Args:
        title: Ticket title
        description: Ticket description
        category: Ticket category (optional)
        urgency: Urgency level (optional)
        status: Ticket status (optional)
        user_id: User ID (optional)
        user_email: User email (optional)
        confidence_score: ML model confidence (optional)
        model_version: ML model version (optional)
        ticket_id: If provided, updates existing ticket
    
    Returns:
        Saved Ticket object
    """
    db = get_database()
    
    with db.session_scope() as session:
        if ticket_id:
            # Update existing ticket
            ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()
            if not ticket:
                raise ValueError(f"Ticket with ID {ticket_id} not found")
            
            # Update fields
            ticket.title = title
            ticket.description = description
            if category is not None:
                ticket.category = category
            if urgency is not None:
                ticket.urgency = urgency
            if status is not None:
                ticket.status = status
            if confidence_score is not None:
                ticket.confidence_score = confidence_score
            if model_version is not None:
                ticket.model_version = model_version
            ticket.updated_at = datetime.utcnow()
            
            logger.info(f"Updated ticket ID: {ticket_id}")
        else:
            # Create new ticket
            ticket = Ticket(
                title=title,
                description=description,
                category=category or TicketCategory.OTHER,
                urgency=urgency or UrgencyLevel.MEDIUM,
                status=status or TicketStatus.PENDING,
                user_id=user_id,
                user_email=user_email,
                confidence_score=confidence_score,
                model_version=model_version
            )
            session.add(ticket)
            session.flush()  # Get the ID before commit
            logger.info(f"Created new ticket ID: {ticket.id}")
        
        # Refresh to get all fields
        session.refresh(ticket)
        
        # Create a detached copy to return
        session.expunge(ticket)
        make_transient(ticket)
        
        return ticket


def get_ticket_by_id(ticket_id: int) -> Optional[Ticket]:
    """Get ticket by ID"""
    db = get_database()
    with db.session_scope() as session:
        return session.query(Ticket).filter(Ticket.id == ticket_id).first()


def update_ticket_classification(
    ticket_id: int,
    category: TicketCategory,
    urgency: UrgencyLevel,
    confidence_score: float,
    processing_time_ms: int
) -> bool:
    """Update ticket classification results"""
    db = get_database()
    with db.session_scope() as session:
        ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket:
            ticket.category = category
            ticket.urgency = urgency
            ticket.confidence_score = confidence_score
            ticket.processing_time_ms = processing_time_ms
            ticket.status = TicketStatus.PROCESSED
            ticket.processed_at = datetime.utcnow()
            return True
        return False


def get_pending_tickets(limit: int = 100) -> List[Ticket]:
    """Get pending tickets for processing"""
    db = get_database()
    with db.session_scope() as session:
        return session.query(Ticket)\
            .filter(Ticket.status == TicketStatus.PENDING)\
            .order_by(Ticket.created_at)\
            .limit(limit)\
            .all()


def get_tickets_by_status(status: TicketStatus, limit: int = 100) -> List[Ticket]:
    """Get tickets by status"""
    db = get_database()
    with db.session_scope() as session:
        return session.query(Ticket)\
            .filter(Ticket.status == status)\
            .order_by(Ticket.created_at.desc())\
            .limit(limit)\
            .all()


def get_ticket_statistics() -> Dict[str, Any]:
    """Get ticket statistics for dashboard"""
    db = get_database()
    with db.session_scope() as session:
        # Total tickets
        total_tickets = session.query(Ticket).count()
        
        # Tickets by status
        status_stats = {}
        for status in TicketStatus:
            count = session.query(Ticket).filter(Ticket.status == status).count()
            status_stats[status.value] = count
        
        # Tickets by category
        category_stats = {}
        for category in TicketCategory:
            count = session.query(Ticket).filter(Ticket.category == category).count()
            category_stats[category.value] = count
        
        # Tickets by urgency
        urgency_stats = {}
        for urgency in UrgencyLevel:
            count = session.query(Ticket).filter(Ticket.urgency == urgency).count()
            urgency_stats[urgency.value] = count
        
        # Average processing time
        avg_processing_time = session.query(
            text('AVG(processing_time_ms)')
        ).select_from(Ticket).filter(
            Ticket.processing_time_ms.isnot(None)
        ).scalar() or 0
        
        # Average confidence score
        avg_confidence = session.query(
            text('AVG(confidence_score)')
        ).select_from(Ticket).filter(
            Ticket.confidence_score.isnot(None)
        ).scalar() or 0
        
        return {
            'total_tickets': total_tickets,
            'by_status': status_stats,
            'by_category': category_stats,
            'by_urgency': urgency_stats,
            'avg_processing_time_ms': float(avg_processing_time),
            'avg_confidence_score': float(avg_confidence),
            'timestamp': datetime.utcnow().isoformat()
        }


def bulk_create_tickets(tickets_data: List[Dict[str, Any]]) -> List[int]:
    """Bulk create tickets from list of dictionaries"""
    db = get_database()
    created_ids = []
    
    with db.session_scope() as session:
        for data in tickets_data:
            ticket = Ticket(
                title=data.get('title', ''),
                description=data.get('description', ''),
                user_email=data.get('user_email'),
                category=data.get('category', TicketCategory.OTHER),
                urgency=data.get('urgency', UrgencyLevel.MEDIUM),
                status=TicketStatus.PENDING
            )
            session.add(ticket)
            session.flush()
            created_ids.append(ticket.id)
    
    return created_ids


def search_tickets(
    query: str,
    limit: int = 50,
    category: Optional[TicketCategory] = None,
    urgency: Optional[UrgencyLevel] = None
) -> List[Ticket]:
    """Search tickets by text query with optional filters"""
    db = get_database()
    with db.session_scope() as session:
        # Build query
        search = session.query(Ticket)
        
        # Text search (PostgreSQL full-text search)
        if query:
            search = search.filter(
                text("""
                    to_tsvector('english', title || ' ' || description) 
                    @@ plainto_tsquery('english', :query)
                """)
            ).params(query=query)
        
        # Apply filters
        if category:
            search = search.filter(Ticket.category == category)
        if urgency:
            search = search.filter(Ticket.urgency == urgency)
        
        return search.order_by(Ticket.created_at.desc()).limit(limit).all()


def get_user_tickets(user_id: int, limit: int = 50) -> List[Ticket]:
    """Get tickets for a specific user"""
    db = get_database()
    with db.session_scope() as session:
        return session.query(Ticket)\
            .filter(Ticket.user_id == user_id)\
            .order_by(Ticket.created_at.desc())\
            .limit(limit)\
            .all()


def cleanup_old_tickets(days: int = 90) -> int:
    """Clean up old processed tickets"""
    db = get_database()
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with db.session_scope() as session:
        deleted_count = session.query(Ticket)\
            .filter(
                Ticket.status == TicketStatus.PROCESSED,
                Ticket.processed_at < cutoff_date
            )\
            .delete()
    
    logger.info(f"Cleaned up {deleted_count} old tickets")
    return deleted_count


# User management functions
def create_user(email: str, username: str, full_name: Optional[str] = None) -> User:
    """Create a new user"""
    db = get_database()
    with db.session_scope() as session:
        user = User(
            email=email,
            username=username,
            full_name=full_name
        )
        session.add(user)
        session.flush()
        return user


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email"""
    db = get_database()
    with db.session_scope() as session:
        return session.query(User).filter(User.email == email).first()


def get_or_create_user(email: str, username: Optional[str] = None) -> User:
    """Get existing user or create new one"""
    existing_user = get_user_by_email(email)
    if existing_user:
        return existing_user
    
    # Create new user
    if not username:
        username = email.split('@')[0]
    
    return create_user(email, username)


# Migration helpers for MySQL to PostgreSQL
def migrate_from_mysql(mysql_connection_string: str):
    """
    Helper function to migrate data from MySQL to PostgreSQL
    This is a basic example - customize based on your needs
    """
    try:
        from sqlalchemy import create_engine as create_mysql_engine
        
        # Connect to MySQL
        mysql_engine = create_mysql_engine(mysql_connection_string)
        
        # Get PostgreSQL database
        pg_db = get_database()
        
        # Create tables in PostgreSQL
        pg_db.create_tables()
        
        with mysql_engine.connect() as mysql_conn:
            # Migrate users
            users_result = mysql_conn.execute(text("SELECT * FROM users"))
            users_data = [dict(row._mapping) for row in users_result]
            
            with pg_db.session_scope() as pg_session:
                for user_data in users_data:
                    user = User(**user_data)
                    pg_session.add(user)
            
            # Migrate tickets
            tickets_result = mysql_conn.execute(text("SELECT * FROM tickets"))
            tickets_data = [dict(row._mapping) for row in tickets_result]
            
            with pg_db.session_scope() as pg_session:
                for ticket_data in tickets_data:
                    # Convert MySQL enum values if needed
                    if 'category' in ticket_data:
                        ticket_data['category'] = TicketCategory(ticket_data['category'])
                    if 'urgency' in ticket_data:
                        ticket_data['urgency'] = UrgencyLevel(ticket_data['urgency'])
                    if 'status' in ticket_data:
                        ticket_data['status'] = TicketStatus(ticket_data['status'])
                    
                    ticket = Ticket(**ticket_data)
                    pg_session.add(ticket)
        
        logger.info("Migration from MySQL completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


# Performance optimization for PostgreSQL
def create_indexes():
    """Create additional indexes for performance optimization"""
    db = get_database()
    
    with db.engine.connect() as conn:
        # Full-text search index
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tickets_fulltext 
            ON tickets USING gin(to_tsvector('english', title || ' ' || description))
        """))
        
        # Composite indexes for common queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tickets_user_status 
            ON tickets(user_id, status, created_at DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tickets_processing 
            ON tickets(status, created_at) 
            WHERE status = 'pending'
        """))
        
        conn.commit()
        
    logger.info("Performance indexes created")


# Database maintenance utilities
def vacuum_analyze():
    """Run VACUUM ANALYZE for PostgreSQL optimization"""
    db = get_database()
    with db.engine.connect() as conn:
        conn.execute(text("VACUUM ANALYZE"))
        logger.info("VACUUM ANALYZE completed")


def get_database_size() -> Dict[str, Any]:
    """Get database size information"""
    db = get_database()
    with db.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                pg_database_size(current_database()) as total_size,
                pg_size_pretty(pg_database_size(current_database())) as total_size_pretty,
                (SELECT count(*) FROM tickets) as ticket_count,
                (SELECT count(*) FROM users) as user_count,
                (SELECT pg_size_pretty(pg_total_relation_size('tickets'))) as tickets_table_size,
                (SELECT pg_size_pretty(pg_total_relation_size('users'))) as users_table_size
        """))
        
        row = result.first()
        return {
            'total_size_bytes': row[0],
            'total_size_pretty': row[1],
            'ticket_count': row[2],
            'user_count': row[3],
            'tickets_table_size': row[4],
            'users_table_size': row[5],
            'timestamp': datetime.utcnow().isoformat()
        }


def get_connection_stats() -> Dict[str, Any]:
    """Get database connection statistics"""
    db = get_database()
    with db.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active_connections,
                count(*) FILTER (WHERE state = 'idle') as idle_connections,
                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
                max(EXTRACT(epoch FROM (now() - query_start))) as longest_query_seconds
            FROM pg_stat_activity
            WHERE datname = current_database()
        """))
        
        row = result.first()
        return {
            'total_connections': row[0],
            'active_connections': row[1],
            'idle_connections': row[2],
            'idle_in_transaction': row[3],
            'longest_query_seconds': row[4] or 0,
            'timestamp': datetime.utcnow().isoformat()
        }


# Environment-specific configurations
class DatabaseEnvironment:
    """Database configuration for different environments"""
    
    @staticmethod
    def get_test_config() -> Dict[str, Any]:
        """Test environment configuration"""
        return {
            'database_url': os.getenv('TEST_DATABASE_URL', 'postgresql://postgres:password@localhost:5432/ticket_system_test'),
            'echo': True,
            'pool_size': 1,
            'max_overflow': 0
        }
    
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Development environment configuration"""
        return {
            'database_url': DatabaseConfig.get_database_url(),
            'echo': os.getenv('SQL_ECHO', 'false').lower() == 'true',
            'pool_size': 5,
            'max_overflow': 10
        }
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Production environment configuration"""
        return {
            'database_url': DatabaseConfig.get_database_url(),
            'echo': False,
            'pool_size': 20,
            'max_overflow': 40,
            'pool_pre_ping': True,
            'pool_recycle': 3600
        }


# Testing utilities
def create_test_database():
    """Create a test database with sample data"""
    db = get_database()
    
    # Create tables
    db.create_tables()
    
    # Create test users
    test_users = [
        {'email': 'admin@example.com', 'username': 'admin', 'full_name': 'Admin User', 'is_admin': True},
        {'email': 'user1@example.com', 'username': 'user1', 'full_name': 'Test User 1'},
        {'email': 'user2@example.com', 'username': 'user2', 'full_name': 'Test User 2'},
    ]
    
    with db.session_scope() as session:
        for user_data in test_users:
            user = User(**user_data)
            session.add(user)
        session.flush()
        
        # Create test tickets
        test_tickets = [
            {
                'title': 'Cannot login to account',
                'description': 'I forgot my password and cannot reset it. The reset email is not arriving.',
                'category': TicketCategory.TECHNICAL,
                'urgency': UrgencyLevel.HIGH,
                'user_id': 1,
                'user_email': 'user1@example.com'
            },
            {
                'title': 'Billing issue with subscription',
                'description': 'I was charged twice for my monthly subscription. Please refund the duplicate charge.',
                'category': TicketCategory.BILLING,
                'urgency': UrgencyLevel.HIGH,
                'user_id': 2,
                'user_email': 'user2@example.com'
            },
            {
                'title': 'Feature request',
                'description': 'It would be great if you could add dark mode to the application.',
                'category': TicketCategory.FEEDBACK,
                'urgency': UrgencyLevel.LOW,
                'user_id': 1,
                'user_email': 'user1@example.com'
            },
            {
                'title': 'General inquiry about pricing',
                'description': 'What are the differences between the pro and enterprise plans?',
                'category': TicketCategory.GENERAL,
                'urgency': UrgencyLevel.MEDIUM,
                'user_email': 'anonymous@example.com'
            }
        ]
        
        for ticket_data in test_tickets:
            ticket = Ticket(**ticket_data)
            session.add(ticket)
    
    logger.info("Test database created with sample data")


# Backup and restore utilities
def backup_database(backup_path: str):
    """Create a backup of the database (PostgreSQL specific)"""
    import subprocess
    
    db_config = DatabaseConfig()
    db_url = db_config.get_database_url()
    
    # Parse database URL
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    
    # Build pg_dump command
    cmd = [
        'pg_dump',
        '-h', parsed.hostname or 'localhost',
        '-p', str(parsed.port or 5432),
        '-U', parsed.username or 'postgres',
        '-d', parsed.path.lstrip('/'),
        '-f', backup_path,
        '--verbose',
        '--format=custom',
        '--no-owner',
        '--no-privileges'
    ]
    
    # Set password through environment
    env = os.environ.copy()
    if parsed.password:
        env['PGPASSWORD'] = parsed.password
    
    try:
        subprocess.run(cmd, check=True, env=env)
        logger.info(f"Database backed up to {backup_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Backup failed: {e}")
        raise


def restore_database(backup_path: str):
    """Restore database from backup"""
    import subprocess
    
    db_config = DatabaseConfig()
    db_url = db_config.get_database_url()
    
    # Parse database URL
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    
    # Build pg_restore command
    cmd = [
        'pg_restore',
        '-h', parsed.hostname or 'localhost',
        '-p', str(parsed.port or 5432),
        '-U', parsed.username or 'postgres',
        '-d', parsed.path.lstrip('/'),
        '--verbose',
        '--clean',
        '--if-exists',
        '--no-owner',
        '--no-privileges',
        backup_path
    ]
    
    # Set password through environment
    env = os.environ.copy()
    if parsed.password:
        env['PGPASSWORD'] = parsed.password
    
    try:
        subprocess.run(cmd, check=True, env=env)
        logger.info(f"Database restored from {backup_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Restore failed: {e}")
        raise


# Connection pool monitoring
def get_pool_status() -> Dict[str, Any]:
    """Get connection pool status"""
    db = get_database()
    pool = db.engine.pool
    
    return {
        'size': getattr(pool, 'size', 0),
        'checked_in_connections': getattr(pool, 'checkedin', 0),
        'checked_out_connections': getattr(pool, 'checkedout', 0),
        'overflow': getattr(pool, 'overflow', 0),
        'total': getattr(pool, 'checkedin', 0) + getattr(pool, 'checkedout', 0)
    }


# Neon-specific utilities
def get_neon_branch_info() -> Optional[Dict[str, Any]]:
    """Get Neon branch information if using Neon"""
    db = get_database()
    
    if 'neon.tech' not in db.database_url:
        return None
    
    try:
        with db.engine.connect() as conn:
            # Neon-specific system functions
            result = conn.execute(text("""
                SELECT 
                    current_setting('neon.branch_name', true) as branch_name,
                    current_setting('neon.project_id', true) as project_id,
                    pg_is_in_recovery() as is_replica
            """))
            
            row = result.first()
            return {
                'branch_name': row[0],
                'project_id': row[1],
                'is_replica': row[2],
                'timestamp': datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get Neon info: {e}")
        return None


# Export all important functions and classes
__all__ = [
    # Core classes
    'Database',
    'DatabaseConfig',
    'Base',
    
    # Models
    'User',
    'Ticket',
    'TicketStatus',
    'TicketCategory', 
    'UrgencyLevel',
    
    # Database functions
    'get_database',
    'get_db_session',
    'get_db',
    'init_database',
    
    # Ticket operations
    'create_ticket',
    'get_ticket_by_id',
    'update_ticket_classification',
    'get_pending_tickets',
    'get_tickets_by_status',
    'get_ticket_statistics',
    'bulk_create_tickets',
    'search_tickets',
    'get_user_tickets',
    'cleanup_old_tickets',
    
    # User operations
    'create_user',
    'get_user_by_email',
    'get_or_create_user',
    
    # Utilities
    'execute_query',
    'migrate_from_mysql',
    'create_indexes',
    'vacuum_analyze',
    'get_database_size',
    'get_connection_stats',
    'create_test_database',
    'backup_database',
    'restore_database',
    'get_pool_status',
    'get_neon_branch_info'
]

# Initialize logger for module
logger.info("PostgreSQL/Neon database module loaded")

# --- Streamlit compatibility helper ---

def get_connection():
    """
    Compatibility wrapper for legacy code.
    Returns a SQLAlchemy session.
    """
    return get_database().get_session()

