"""
DFS Meta-Optimizer v8.0.0 - Performance Dashboard
Real-time performance monitoring UI for Streamlit

NEW IN v8.0.0:
- Live performance metrics
- Bottleneck visualization
- Historical trends
- Alert system
- One-click optimization suggestions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
from performance_monitor import monitor, PerformanceReport

def render_performance_dashboard():
    """
    Render comprehensive performance dashboard in Streamlit.
    
    Displays:
    - Real-time metrics
    - Function timing breakdown
    - Bottleneck identification
    - Performance alerts
    - Optimization suggestions
    """
    st.header("⚡ Performance Monitor")
    
    # Get current metrics
    dashboard_metrics = PerformanceReport.get_dashboard_metrics()
    summary = monitor.get_summary()
    
    if dashboard_metrics['status'] == 'No data':
        st.info("🔍 No performance data collected yet. Run some optimizations to see metrics!")
        return
    
    # Status indicator
    status = dashboard_metrics['status']
    if status == 'Healthy':
        st.success(f"✅ System Status: {status}")
    else:
        st.warning(f"⚠️ System Status: {status}")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Time",
            f"{dashboard_metrics['total_time_ms']:.0f}ms",
            help="Total time spent in monitored functions"
        )
    
    with col2:
        st.metric(
            "Avg Time/Call",
            f"{dashboard_metrics['avg_time_ms']:.1f}ms",
            help="Average time per function call"
        )
    
    with col3:
        st.metric(
            "Function Calls",
            dashboard_metrics['call_count'],
            help="Total number of function calls monitored"
        )
    
    with col4:
        cache_hit_rate = (monitor.cache_hits / 
                         (monitor.cache_hits + monitor.cache_misses) * 100
                         if (monitor.cache_hits + monitor.cache_misses) > 0 else 0)
        st.metric(
            "Cache Hit Rate",
            f"{cache_hit_rate:.1f}%",
            help="Percentage of cached responses"
        )
    
    st.markdown("---")
    
    # Two columns: Bottlenecks and Alerts
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("🐌 Top Bottlenecks")
        
        if dashboard_metrics['bottlenecks']:
            # Create DataFrame for visualization
            bottleneck_df = pd.DataFrame(dashboard_metrics['bottlenecks'])
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=bottleneck_df['avg_time_ms'],
                    y=bottleneck_df['name'],
                    orientation='h',
                    marker=dict(
                        color=bottleneck_df['avg_time_ms'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Avg Time (ms)")
                    ),
                    text=bottleneck_df['avg_time_ms'].round(1),
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Average Execution Time by Function",
                xaxis_title="Time (ms)",
                yaxis_title="",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            with st.expander("📊 Detailed Breakdown"):
                display_df = bottleneck_df[['name', 'avg_time_ms', 'max_time_ms', 'call_count', 'total_time_ms']]
                display_df.columns = ['Function', 'Avg (ms)', 'Max (ms)', 'Calls', 'Total (ms)']
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No bottlenecks detected")
    
    with col_right:
        st.subheader("🚨 Recent Alerts")
        
        if dashboard_metrics['alerts']:
            for alert in dashboard_metrics['alerts']:
                st.warning(alert)
        else:
            st.success("✅ No performance issues detected")
        
        # Performance tips
        st.subheader("💡 Optimization Tips")
        
        if summary and summary.get('slowest_functions'):
            slowest = summary['slowest_functions'][0]
            
            tips = []
            
            if slowest['avg_time_ms'] > 1000:
                tips.append("🔹 Consider enabling parallel processing for lineup generation")
            
            if cache_hit_rate < 50:
                tips.append("🔹 Enable response caching to speed up repeated queries")
            
            if slowest['name'] in ['_generate_genetic', 'optimize_lineups']:
                tips.append("🔹 Reduce population size or generations for faster results")
            
            if not tips:
                tips.append("✅ Performance is optimized!")
            
            for tip in tips:
                st.markdown(tip)
    
    st.markdown("---")
    
    # Function-level breakdown
    st.subheader("🔍 Function Performance Details")
    
    if summary and summary.get('slowest_functions'):
        func_names = [f['name'] for f in summary['slowest_functions']]
        selected_func = st.selectbox(
            "Select function to analyze:",
            func_names,
            help="View detailed stats for specific function"
        )
        
        if selected_func:
            func_stats = monitor.get_function_stats(selected_func)
            
            if func_stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Calls", func_stats.call_count)
                
                with col2:
                    st.metric("Avg Time", f"{func_stats.avg_time_ms:.2f}ms")
                
                with col3:
                    st.metric("Min Time", f"{func_stats.min_time_ms:.2f}ms")
                
                with col4:
                    st.metric("Max Time", f"{func_stats.max_time_ms:.2f}ms")
                
                # Last 10 calls chart
                if func_stats.last_10_times:
                    st.markdown("**Recent Call Times:**")
                    times_df = pd.DataFrame({
                        'Call': range(1, len(func_stats.last_10_times) + 1),
                        'Time (ms)': list(func_stats.last_10_times)
                    })
                    
                    fig = px.line(
                        times_df,
                        x='Call',
                        y='Time (ms)',
                        markers=True,
                        title=f"Last {len(func_stats.last_10_times)} Calls"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Metrics"):
            monitor.reset()
            st.success("Metrics reset!")
            st.rerun()
    
    with col2:
        if st.button("📥 Export Data"):
            metrics_data = monitor.export_metrics()
            df = pd.DataFrame(metrics_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="performance_metrics.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📊 Full Report"):
            report = PerformanceReport.generate_text_report()
            st.text_area("Performance Report", report, height=400)


def show_performance_summary():
    """
    Show compact performance summary in sidebar.
    """
    dashboard_metrics = PerformanceReport.get_dashboard_metrics()
    
    if dashboard_metrics['status'] != 'No data':
        st.sidebar.markdown("### ⚡ Performance")
        
        status_emoji = "✅" if dashboard_metrics['status'] == 'Healthy' else "⚠️"
        st.sidebar.markdown(f"{status_emoji} **Status:** {dashboard_metrics['status']}")
        
        st.sidebar.metric(
            "Total Time",
            f"{dashboard_metrics['total_time_ms']:.0f}ms"
        )
        
        st.sidebar.metric(
            "Function Calls",
            dashboard_metrics['call_count']
        )
        
        if dashboard_metrics['alerts']:
            st.sidebar.warning(f"⚠️ {len(dashboard_metrics['alerts'])} alerts")


__all__ = ['render_performance_dashboard', 'show_performance_summary']
