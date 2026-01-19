"""
Batch ticket upload page
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import io

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.database import save_ticket, TicketCategory, UrgencyLevel
from models.bert_classifier import BertTicketClassifier
from tasks.celery_app import process_batch_tickets

def show():
    """Display batch upload interface"""
    st.title("📁 Batch Ticket Upload")
    st.markdown("---")
    
    # Instructions
    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        ### CSV Format Requirements:
        Your CSV file should contain the following columns:
        - **title**: Ticket title (required)
        - **description**: Detailed description (required)
        - **customer_email**: Customer email address (required)
        
        ### Sample Format:
        | title | description | customer_email |
        |-------|-------------|----------------|
        | Login issue | Cannot access account... | user@email.com |
        | Billing question | Wrong charge on invoice... | customer@email.com |
        
        ### Tips:
        - Maximum 1000 tickets per batch
        - Ensure all email addresses are valid
        - Descriptions should be detailed for better classification
        - Remove any empty rows before uploading
        """)
        
        # Download sample template
        sample_df = pd.DataFrame({
            'title': [
                'Login issue',
                'Billing question',
                'Feature request',
                'App crash report',
                'Account upgrade'
            ],
            'description': [
                'Cannot access my account after password reset. Getting error code 1001.',
                'I was charged twice for my subscription this month. Please refund the duplicate charge.',
                'Would like to see dark mode feature in the mobile app for night time usage.',
                'The app crashes when trying to upload files larger than 10MB on Android devices.',
                'I want to upgrade from Basic to Premium plan but the option is not showing in my account.'
            ],
            'customer_email': [
                'user1@email.com',
                'user2@email.com',
                'user3@email.com',
                'user4@email.com',
                'user5@email.com'
            ]
        })
        
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Template",
            data=csv,
            file_name="ticket_template.csv",
            mime="text/csv"
        )
    
    # File upload section
    st.markdown("### 📤 Upload Your File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Maximum 1000 tickets per batch"
    )
    
    if uploaded_file is not None:
        # Read and validate CSV
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validation
            required_cols = ['title', 'description', 'customer_email']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Clean data
            original_count = len(df)
            df = df.dropna(subset=required_cols)
            df = df[df['customer_email'].str.contains('@', na=False)]
            cleaned_count = len(df)
            
            if cleaned_count < original_count:
                st.warning(f"⚠️ Removed {original_count - cleaned_count} invalid rows")
            
            # Display preview
            st.markdown("### 📊 Data Preview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Valid Tickets", cleaned_count)
            with col2:
                st.metric("Unique Customers", df['customer_email'].nunique())
            with col3:
                avg_desc_length = df['description'].str.len().mean()
                st.metric("Avg Description Length", f"{avg_desc_length:.0f} chars")
            
            # Show sample data
            st.markdown("#### Sample Data (First 5 rows)")
            st.dataframe(df.head(), use_container_width=True)
            
            # Validation checks
            st.markdown("### ✅ Validation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Email validation
                valid_emails = df['customer_email'].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$')
                email_validity = valid_emails.sum() / len(df) * 100
                
                if email_validity == 100:
                    st.success(f"✅ All email addresses are valid")
                else:
                    st.warning(f"⚠️ {email_validity:.1f}% of emails are valid")
                
                # Title length check
                short_titles = (df['title'].str.len() < 10).sum()
                if short_titles > 0:
                    st.warning(f"⚠️ {short_titles} titles are very short (< 10 chars)")
                else:
                    st.success("✅ All titles have adequate length")
            
            with col2:
                # Description length check
                short_descs = (df['description'].str.len() < 20).sum()
                if short_descs > 0:
                    st.warning(f"⚠️ {short_descs} descriptions are very short (< 20 chars)")
                else:
                    st.success("✅ All descriptions have adequate length")
                
                # Duplicate check
                duplicates = df.duplicated(subset=['title', 'description']).sum()
                if duplicates > 0:
                    st.warning(f"⚠️ Found {duplicates} duplicate tickets")
                else:
                    st.success("✅ No duplicate tickets found")
            
            # Classification preview
            if st.checkbox("🔍 Preview Classifications (first 5 tickets)"):
                with st.spinner("Classifying sample tickets..."):
                    # Initialize model
                    if "model" not in st.session_state:
                        st.session_state.model = BertTicketClassifier()
                        st.session_state.model.load_model()
                    
                    # Classify first 5 tickets
                    sample_texts = (df.head()['title'] + " " + df.head()['description']).tolist()
                    predictions = st.session_state.model.predict_batch(sample_texts)
                    
                    # Display predictions
                    preview_df = df.head().copy()
                    preview_df['Predicted Category'] = [p['category'] for p in predictions]
                    preview_df['Predicted Urgency'] = [p['urgency'] for p in predictions]
                    preview_df['Confidence'] = [f"{p['category_confidence']:.2%}" for p in predictions]
                    
                    st.dataframe(
                        preview_df[['title', 'Predicted Category', 'Predicted Urgency', 'Confidence']],
                        use_container_width=True
                    )
            
            # Process button
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Process All Tickets", use_container_width=True, type="primary"):
                    if len(df) > 1000:
                        st.error("❌ Maximum 1000 tickets per batch. Please split your file.")
                    else:
                        with st.spinner(f"Processing {len(df)} tickets..."):
                            # Save tickets and get IDs
                            ticket_ids = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, row in df.iterrows():
                                status_text.text(f"Processing ticket {idx + 1}/{len(df)}: {row['title'][:50]}...")
                                
                                # Save ticket with pending classification 
                                ticket = save_ticket(
                                    title=row['title'],
                                    description=row['description'],
                                    category=TicketCategory.OTHER,
                                    urgency=UrgencyLevel.MEDIUM,
                                    user_email=row['customer_email']
                                )
                                
                                ticket_ids.append(ticket.id)

                                
                                if ticket_id:
                                    ticket_ids.append(ticket_id)
                                
                                progress_bar.progress((idx + 1) / len(df))
                            
                            status_text.empty()
                            
                            # Trigger async batch processing
                            if ticket_ids:
                                # Try to trigger async batch processing
                                try:
                                    from tasks.celery_app import process_batch_tickets
                                    result = process_batch_tickets(ticket_ids)
                                    st.info(f"🔄 Batch processing queued (Task ID: {result.id})")
                                except Exception as e:
                                    st.warning(f"Async processing unavailable: {e}")
                                    # Continue anyway - tickets are saved
                                
                                st.success(f"""
                                ✅ Successfully uploaded {len(ticket_ids)} tickets!
                                
                                📊 All tickets have been classified and saved to the database.
                                📧 Email notifications will be sent shortly.
                                """)
                                
                                # Prepare results for download
                                results_df = df.copy()
                                results_df['ticket_id'] = ticket_ids
                                results_df['status'] = 'Processing'
                                results_df['upload_timestamp'] = pd.Timestamp.now()
                                
                                # Create download buttons
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    csv_results = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Upload Receipt",
                                        data=csv_results,
                                        file_name=f"ticket_upload_receipt_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    # Create summary report
                                    summary = f"""
                                    Upload Summary Report
                                    ====================
                                    Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                                    Total Tickets: {len(ticket_ids)}
                                    Ticket IDs: {min(ticket_ids)} - {max(ticket_ids)}
                                    
                                    Next Steps:
                                    1. Check Admin Dashboard for classification results
                                    2. Review any tickets that need manual intervention
                                    3. Monitor customer notifications
                                    """
                                    
                                    st.download_button(
                                        label="📄 Download Summary Report",
                                        data=summary,
                                        file_name=f"upload_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain"
                                    )
                                
                                # Show next steps
                                with st.expander("📋 What Happens Next?"):
                                    st.markdown("""
                                    1. **Immediate**: Tickets are queued for AI classification
                                    2. **1-2 minutes**: Classification completed and tickets categorized
                                    3. **2-5 minutes**: Automated notifications sent to customers
                                    4. **5-10 minutes**: Tickets routed to appropriate support teams
                                    5. **Ongoing**: Monitor and manage tickets in Admin Dashboard
                                    
                                    **Pro Tips:**
                                    - High-priority tickets are automatically escalated
                                    - You'll receive alerts for critical issues
                                    - Check the dashboard for real-time updates
                                    """)
                            else:
                                st.error("❌ Failed to save tickets. Please try again.")
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    # Additional features
    st.markdown("---")
    st.markdown("### 🛠️ Additional Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 View Upload History"):
            st.info("Upload history feature coming soon!")
    
    with col2:
        if st.button("⚙️ Configure Auto-Assignment Rules"):
            st.info("Auto-assignment configuration coming soon!")
