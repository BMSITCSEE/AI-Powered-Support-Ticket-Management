"""
Main Streamlit application entry point
"""
import streamlit as st
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import Config
from app.database import init_db
from app.auth import check_authentication, logout
from app.pages import submit_ticket, batch_upload, admin_dashboard


#
# Configure logging (safe for local + Streamlit Cloud)
import os

handlers = [logging.StreamHandler()]  # always log to console

# Try enabling file logging only if filesystem allows
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
log_dir = os.path.abspath(log_dir)

try:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")
    handlers.append(logging.FileHandler(log_file))
except Exception as e:
    print(f"⚠️ File logging disabled: {e}")

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=handlers
)
logger = logging.getLogger(__name__)



# Page configuration
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-support-ticket-manager',
        'Report a bug': "https://github.com/yourusername/ai-support-ticket-manager/issues",
        'About': f"{Config.APP_NAME} v{Config.APP_VERSION}"
    }
)

def init_session_state():
    """Initialize session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.page = "Submit Ticket"
        logger.info("Session state initialized")

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding-top: 2rem;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Secondary button */
    .stButton > button:not([kind="primary"]) {
        background-color: #f0f2f6;
        color: #262730;
        border: 1px solid #e0e2e6;
    }
    
    /* Success alert */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    
    /* Error alert */
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        # Logo/Header
        st.markdown(f"# 🎫 {Config.APP_NAME}")
        st.markdown(f"**Version:** {Config.APP_VERSION}")
        st.markdown("---")
        
        # Navigation
        pages = ["Submit Ticket", "Batch Upload", "Admin Dashboard"]
        
        # Add user info if authenticated
        if st.session_state.authenticated:
            user = st.session_state.user
            st.markdown(f"**Logged in as:** {user['username']}")
            st.markdown(f"**Role:** {user['role'].title()}")
            st.markdown("---")
        
        # Page selection
        selected_page = st.selectbox(
            "Navigate to:",
            pages,
            key="navigation",
            help="Select a page to navigate"
        )
        
        # Admin authentication for dashboard
        if selected_page == "Admin Dashboard" and not st.session_state.authenticated:
            st.warning("🔒 Admin login required")
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.authenticated:
            st.markdown("### 📊 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Open", "24", "+3")
            with col2:
                st.metric("Critical", "2", "-1")
        
        # Support info
        st.markdown("---")
        st.markdown("### 💡 Need Help?")
        st.markdown("""
        - 📧 support@example.com
        - 📞 1-800-SUPPORT
        - 💬 [Live Chat](https://example.com/chat)
        """)
        
        # Environment indicator
        if Config.ENVIRONMENT != "production":
            st.markdown("---")
            st.warning(f"🔧 Environment: {Config.ENVIRONMENT.upper()}")
        
        # Logout button
        if st.session_state.authenticated:
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                logout()
                st.rerun()
        
        # Footer
        st.markdown("---")
        st.caption(f"© 2024 {Config.APP_NAME}")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

def main():
    """Main application function"""
    # Initialize
    init_session_state()
    init_db()
    apply_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Get selected page
    page = st.session_state.get("navigation", "Submit Ticket")
    
    # Route to appropriate page
    try:
        if page == "Submit Ticket":
            submit_ticket.show()
        elif page == "Batch Upload":
            batch_upload.show()
        elif page == "Admin Dashboard":
            if check_authentication():
                admin_dashboard.show()
            else:
                st.error("🔒 Please login to access the admin dashboard")
    except Exception as e:
        logger.error(f"Error rendering page {page}: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        if Config.DEBUG:
            st.exception(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        st.error("A critical error occurred. Please contact support.")
        if Config.DEBUG:
            st.exception(e)
