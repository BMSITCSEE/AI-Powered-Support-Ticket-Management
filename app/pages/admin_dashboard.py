"""
Admin dashboard page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.database import get_all_tickets, get_ticket_stats, update_ticket_status
from utils.metrics import calculate_metrics, plot_confusion_matrix, calculate_sla_compliance

def show():
    """Display admin dashboard"""
    st.title("📊 Admin Dashboard")
    st.markdown("---")
    
    # Fetch statistics
    stats = get_ticket_stats()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tickets", 
            stats['total'],
            delta="+12% from last week"
        )
    
    with col2:
        open_tickets = sum(s['count'] for s in stats['by_status'] if s['status'] == 'Open')
        st.metric(
            "Open Tickets", 
            open_tickets,
            delta="-5% from yesterday"
        )
        with col3:
        critical_tickets = sum(s['count'] for s in stats['by_urgency'] if s['urgency'] == 'Critical')
        st.metric(
            "Critical Issues", 
            critical_tickets,
            delta="2 new today"
        )
    
    with col4:
        avg_resolution = "24h 32m"  # This would be calculated from DB
        st.metric(
            "Avg Resolution Time", 
            avg_resolution,
            delta="-2h from last month"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        st.subheader("Tickets by Category")
        if stats['by_category']:
            df_category = pd.DataFrame(stats['by_category'])
            fig_category = px.pie(
                df_category, 
                values='count', 
                names='category',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Urgency distribution
        st.subheader("Tickets by Urgency")
        if stats['by_urgency']:
            df_urgency = pd.DataFrame(stats['by_urgency'])
            # Order by urgency level
            urgency_order = ['Critical', 'High', 'Medium', 'Low']
            df_urgency['urgency'] = pd.Categorical(
                df_urgency['urgency'], 
                categories=urgency_order, 
                ordered=True
            )
            df_urgency = df_urgency.sort_values('urgency')
            
            fig_urgency = px.bar(
                df_urgency, 
                x='urgency', 
                y='count',
                color='urgency',
                color_discrete_map={
                    'Critical': '#ff4444',
                    'High': '#ff8800',
                    'Medium': '#ffbb33',
                    'Low': '#00C851'
                }
            )
            st.plotly_chart(fig_urgency, use_container_width=True)
    
    # Time series chart
    st.subheader("Ticket Volume Trend")
    # Generate sample time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    ticket_counts = [stats['total'] // 30 + (i % 5) for i in range(30)]
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=dates,
        y=ticket_counts,
        mode='lines+markers',
        name='Daily Tickets',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=8)
    ))
    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Tickets",
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # Ticket management table
    st.subheader("🎫 Ticket Management")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_filter = st.selectbox(
            "Status", 
            ["All", "Open", "In Progress", "Resolved", "Closed", "Escalated"]
        )
    with col2:
        category_filter = st.selectbox(
            "Category",
            ["All", "Technical", "Billing", "Feedback", "General", "Account"]
        )
    with col3:
        urgency_filter = st.selectbox(
            "Urgency",
            ["All", "Critical", "High", "Medium", "Low"]
        )
    with col4:
        date_filter = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now())
        )
    
    # Fetch and display tickets
    tickets_df = get_all_tickets()
    
    # Apply filters
    if status_filter != "All":
        tickets_df = tickets_df[tickets_df['status'] == status_filter]
    if category_filter != "All":
        tickets_df = tickets_df[tickets_df['category'] == category_filter]
    if urgency_filter != "All":
        tickets_df = tickets_df[tickets_df['urgency'] == urgency_filter]
    
    # Display tickets
    if not tickets_df.empty:
        # Add action buttons
        for idx, row in tickets_df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
            
            with col1:
                st.write(f"**#{row['id']}** - {row['title'][:50]}...")
            with col2:
                urgency_color = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(row['urgency'], '⚪')
                st.write(f"{urgency_color} {row['urgency']}")
            with col3:
                st.write(f"📁 {row['category']}")
            with col4:
                status_emoji = {
                    'Open': '📬',
                    'In Progress': '⏳',
                    'Resolved': '✅',
                    'Closed': '📪',
                    'Escalated': '🚨'
                }.get(row['status'], '❓')
                st.write(f"{status_emoji} {row['status']}")
            with col5:
                if st.button("View", key=f"view_{row['id']}"):
                    show_ticket_details(row)
        
        # Export functionality
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            csv = tickets_df.to_csv(index=False)
            st.download_button(
                label="📥 Export to CSV",
                data=csv,
                file_name=f"tickets_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()
    else:
        st.info("No tickets found matching the selected filters.")
    
    # Model Performance Metrics
    st.markdown("---")
    st.subheader("🤖 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classification Accuracy", "94.2%", "+1.2%")
    with col2:
        st.metric("F1 Score", "0.92", "+0.03")
    with col3:
        st.metric("Avg Inference Time", "120ms", "-15ms")
    
    # SLA Compliance
    st.markdown("---")
    st.subheader("📈 SLA Compliance")
    
    # Calculate SLA compliance (mock data for demo)
    sla_data = pd.DataFrame({
        'Urgency': ['Critical', 'High', 'Medium', 'Low'],
        'SLA Target (hours)': [4, 24, 48, 72],
        'Compliance Rate': [92, 87, 95, 98],
        'Tickets': [15, 45, 120, 80]
    })
    
    fig_sla = px.bar(
        sla_data,
        x='Urgency',
        y='Compliance Rate',
        color='Compliance Rate',
        color_continuous_scale=['red', 'yellow', 'green'],
        text='Compliance Rate'
    )
    fig_sla.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig_sla, use_container_width=True)
    
    # Agent Performance
    st.markdown("---")
    st.subheader("👥 Agent Performance")
    
    # Mock agent data
    agent_data = pd.DataFrame({
        'Agent': ['agent1@company.com', 'agent2@company.com', 'agent3@company.com'],
        'Tickets Resolved': [45, 38, 52],
        'Avg Resolution Time (hours)': [18, 22, 16],
        'Customer Satisfaction': [4.8, 4.5, 4.9]
    })
    
    st.dataframe(
        agent_data.style.highlight_max(subset=['Tickets Resolved', 'Customer Satisfaction'])
                       .highlight_min(subset=['Avg Resolution Time (hours)']),
        use_container_width=True
    )

def show_ticket_details(ticket):
    """Show detailed ticket information in a modal"""
    with st.expander(f"Ticket #{ticket['id']} Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Title:** {ticket['title']}")
            st.markdown(f"**Category:** {ticket['category']}")
            st.markdown(f"**Urgency:** {ticket['urgency']}")
            st.markdown(f"**Status:** {ticket['status']}")
        
        with col2:
            st.markdown(f"**Customer:** {ticket['customer_email']}")
            st.markdown(f"**Created:** {ticket['created_at']}")
            st.markdown(f"**Assigned To:** {ticket['assigned_to'] or 'Unassigned'}")
        
        st.markdown("**Description:**")
        st.text_area("", value=ticket['description'], height=150, disabled=True, key=f"desc_{ticket['id']}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            new_status = st.selectbox(
                "Update Status",
                ["Open", "In Progress", "Resolved", "Closed", "Escalated"],
                index=["Open", "In Progress", "Resolved", "Closed", "Escalated"].index(ticket['status']),
                key=f"status_{ticket['id']}"
            )
        with col2:
            assignee = st.text_input(
                "Assign To",
                value=ticket['assigned_to'] or "",
                key=f"assign_{ticket['id']}"
            )
        with col3:
            if st.button("Update", key=f"update_{ticket['id']}"):
                if update_ticket_status(ticket['id'], new_status, assignee):
                    st.success("Ticket updated successfully!")
                    st.rerun()
        
        # Resolution notes
        st.markdown("**Resolution Notes:**")
        resolution_notes = st.text_area(
            "",
            placeholder="Add resolution notes here...",
            key=f"resolution_{ticket['id']}"
        )
        
        if st.button("Save Notes", key=f"save_notes_{ticket['id']}"):
            # Save resolution notes logic here
            st.success("Notes saved!")