"""
Authentication and authorization utilities
"""
import streamlit as st
import hashlib
import jwt
from datetime import datetime, timedelta
from app.config import Config
from app.database import get_connection_stats
import bcrypt
import secrets
import psycopg2.extras


def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def verify_user(username, password):
    """Verify user credentials"""
    connection = get_connection()
    if not connection:
        return False
    
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    cursor.execute("""
    SELECT id, username, password_hash, role, email
    FROM users 
    WHERE username = %s OR email = %s
    """, (username, username))
    
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if user and verify_password(password, user['password_hash']):
        return user
    return None

def create_user(username, password, email, role='user'):
    """Create a new user"""
    connection = get_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute("""
        INSERT INTO users (username, password_hash, email, role)
        VALUES (%s, %s, %s, %s)
        """, (username, password_hash, email, role))
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

def create_jwt_token(user_id, username, role):
    """Create JWT token for user session"""
    payload = {
        'user_id': user_id,
        'username': username,
        'role': role,
        'exp': datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    
    token = jwt.encode(payload, Config.SECRET_KEY, algorithm='HS256')
    return token

def verify_jwt_token(token):
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def check_authentication():
    """Check if user is authenticated for admin access"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
    
    if not st.session_state.authenticated:
        # Show login form
        st.markdown("### 🔐 Admin Login Required")
        
        with st.form("login_form"):
            username = st.text_input(
                "Username or Email",
                placeholder="Enter your username or email"
            )
            password = st.text_input(
                "Password", 
                type="password",
                placeholder="Enter your password"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("Login", use_container_width=True, type="primary")
            with col2:
                if st.form_submit_button("Forgot Password?", use_container_width=True):
                    st.info("Password reset functionality coming soon!")
            
            if login_button:
                if username and password:
                    # For demo purposes, accept admin/admin
                    if username == "admin" and password == "admin":
                        st.session_state.authenticated = True
                        st.session_state.user = {
                            'id': 1,
                            'username': 'admin',
                            'role': 'admin',
                            'email': 'admin@example.com'
                        }
                        st.success("Welcome back, admin!")
                        st.rerun()
                    else:
                        user = verify_user(username, password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.session_state.token = create_jwt_token(
                                user['id'], 
                                user['username'], 
                                user['role']
                            )
                            st.success(f"Welcome back, {user['username']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
        
        return False
    
    return True

def logout():
    """Logout current user"""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.token = None
    st.info("You have been logged out successfully")

def require_role(required_role):
    """Decorator to require specific role for access"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.authenticated:
                st.error("Authentication required")
                return None
            
            user_role = st.session_state.user.get('role', 'user')
            if user_role != required_role and user_role != 'admin':
                st.error(f"Access denied. {required_role} role required.")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def create_password_reset_token(email):
    """Create password reset token"""
    token = secrets.token_urlsafe(32)
    expiry = datetime.utcnow() + timedelta(hours=1)
    
    # Store token in database (simplified)
    connection = get_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("""
        UPDATE users 
        SET reset_token = %s, reset_token_expiry = %s 
        WHERE email = %s
        """, (token, expiry, email))
        connection.commit()
        cursor.close()
        connection.close()
    
    return token

def verify_reset_token(token):
    """Verify password reset token"""
    connection = get_connection()
    if not connection:
        return None
    
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
    SELECT id, username, email 
    FROM users 
    WHERE reset_token = %s AND reset_token_expiry > NOW()
    """, (token,))
    
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    
    return user

def update_password(user_id, new_password):
    """Update user password"""
    connection = get_connection()
    if not connection:
        return False
    
    cursor = connection.cursor()
    password_hash = hash_password(new_password)
    
    try:
        cursor.execute("""
        UPDATE users 
        SET password_hash = %s, reset_token = NULL, reset_token_expiry = NULL 
        WHERE id = %s
        """, (password_hash, user_id))
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        print(f"Error updating password: {e}")
        return False

def get_user_permissions(user_id):
    """Get user permissions"""
    connection = get_connection()
    if not connection:
        return []
    
    cursor = connection.cursor()
    cursor.execute("""
    SELECT permission 
    FROM user_permissions 
    WHERE user_id = %s
    """, (user_id,))
    
    permissions = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    
    return permissions

def has_permission(permission):
    """Check if current user has specific permission"""
    if not st.session_state.authenticated:
        return False
    
    user_role = st.session_state.user.get('role', 'user')
    
    # Admin has all permissions
    if user_role == 'admin':
        return True
    
    # Check specific permissions
    user_id = st.session_state.user.get('id')
    permissions = get_user_permissions(user_id)
    
    return permission in permissions
