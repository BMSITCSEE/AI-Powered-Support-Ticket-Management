"""
Single ticket submission page
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.database import save_ticket
from models.bert_classifier import BertTicketClassifier
from tasks.celery_app import classify_and_notify
import torch

def show():
    """Display ticket submission form"""
    st.title("🎫 Submit Support Ticket")
    st.markdown("---")
    
    # Initialize model in session state
    if "model" not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.model = BertTicketClassifier()
            st.session_state.model.load_model()
    
    # Ticket submission form
    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input(
                "Ticket Title*", 
                placeholder="Brief summary of your issue",
                help="Provide a concise title for your ticket"
            )
            customer_email = st.text_input(
                "Your Email*", 
                placeholder="email@example.com",
                help="We'll use this to send updates about your ticket"
            )
        
        with col2:
            st.markdown("&nbsp;")  # Spacing
            st.info("💡 **Tip:** Our AI will automatically categorize and prioritize your ticket based on the information you provide.")
        
        description = st.text_area(
            "Description*", 
            placeholder="Please describe your issue in detail. Include:\n- What you were trying to do\n- What happened instead\n- Any error messages\n- Steps to reproduce the issue",
            height=200,
            help="The more details you provide, the better we can help!"
        )
        
        # Additional optional fields
        with st.expander("Additional Information (Optional)"):
            col1, col2 = st.columns(2)
            with col1:
                account_id = st.text_input("Account ID", placeholder="Your account number")
                phone = st.text_input("Phone Number", placeholder="+1 (555) 123-4567")
            with col2:
                product = st.selectbox("Product/Service", ["", "Web App", "Mobile App", "API", "Other"])
                attachment = st.file_uploader("Attach Screenshot", type=['png', 'jpg', 'jpeg', 'pdf'])
        
        # Submit button
        submitted = st.form_submit_button("Submit Ticket", use_container_width=True, type="primary")
        
        if submitted:
            # Validation
            errors = []
            if not title:
                errors.append("Title is required")
            if not description:
                errors.append("Description is required")
            if not customer_email:
                errors.append("Email is required")
            elif "@" not in customer_email or "." not in customer_email.split("@")[1]:
                errors.append("Please enter a valid email address")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                with st.spinner("Processing your ticket..."):
                    # Combine title and description for classification
                    full_text = f"{title} {description}"
                    
                    # Classify ticket
                    category, urgency, confidence = st.session_state.model.predict(full_text)
                    
                    # Save to database
                    ticket_id = save_ticket(
                        title=title,
                        description=description,
                        category=category,
                        urgency=urgency,
                        customer_email=customer_email
                    )
                    
                    if ticket_id:
                        # Trigger async notification
                        classify_and_notify.delay(ticket_id)
                        
                        # Success message with details
                        st.success(f"✅ Ticket #{ticket_id} submitted successfully!")
                        
                        # Display ticket details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Category", category)
                        with col2:
                            urgency_color = {
                                'Critical': '🔴',
                                'High': '🟠',
                                'Medium': '🟡',
                                'Low': '🟢'
                            }.get(urgency, '⚪')
                            st.metric("Priority", f"{urgency_color} {urgency}")
                        with col3:
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Expected response time
                        response_times = {
                            'Critical': '4 hours',
                            'High': '24 hours',
                            'Medium': '48 hours',
                            'Low': '72 hours'
                        }
                        
                        st.info(f"""
                        📧 **Confirmation email sent to:** {customer_email}
                        
                        ⏱️ **Expected response time:** Within {response_times.get(urgency, '72 hours')}
                        
                        📌 **Your ticket reference:** #{ticket_id}
                        """)
                        
                        # Display next steps
                        with st.expander("📋 What happens next?"):
                            st.markdown(f"""
                            1. **Immediate:** You'll receive a confirmation email with your ticket details
                            2. **Within 1 hour:** Your ticket will be assigned to the {category} team
                            3. **Within {response_times.get(urgency)}:** You'll receive an initial response
                            4. **Ongoing:** Track updates via email or check status using ticket #{ticket_id}
                            
                            **Need urgent help?**
                            - For critical issues, call our hotline: 1-800-SUPPORT
                            - Live chat available for High priority issues
                            """)
                        
                        # Clear form
                        st.session_state.clear()
                        if st.button("Submit Another Ticket"):
                            st.rerun()
                    else:
                        st.error("❌ Error submitting ticket. Please try again or contact support.")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 📞 Need Help?")
        st.markdown("""
        **Before submitting a ticket:**
        - Check our [FAQ](https://example.com/faq)
        - View [System Status](https://status.example.com)
        - Browse [Knowledge Base](https://kb.example.com)
        
        **Contact Options:**
        - 📧 Email: support@example.com
        - 📞 Phone: 1-800-SUPPORT
        - 💬 Live Chat: Available 24/7
        """)
        
        st.markdown("---")
        st.markdown("### 🕐 Support Hours")
        st.markdown("""
        **Regular Support:** Mon-Fri 9AM-6PM EST
        **Premium Support:** 24/7
        **Emergency Hotline:** Always available
        """)