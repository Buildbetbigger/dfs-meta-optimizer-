"""
DFS Meta-Optimizer - Streamlit Application

A revolutionary DFS optimizer that maximizes competitive advantage
rather than just projected points.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(__file__))

from config.settings import (
    SALARY_CAP,
    ROSTER_SIZE,
    CAPTAIN_MULTIPLIER,
    OPTIMIZATION_MODES,
    ENABLE_CLAUDE_AI
)
from modules.opponent_modeling import OpponentModel
from modules.optimization_engine import LineupOptimizer

# Check if Claude assistant is available
try:
    from modules.claude_assistant import ClaudeAssistant
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


# Page config
st.set_page_config(
    page_title="DFS Meta-Optimizer",
    page_icon="🎯",
    layout="wide"
)


def initialize_session_state():
    """Initialize session state variables"""
    if 'players_df' not in st.session_state:
        st.session_state.players_df = None
    if 'opponent_model' not in st.session_state:
        st.session_state.opponent_model = None
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'generated_lineups' not in st.session_state:
        st.session_state.generated_lineups = []
    if 'claude_assistant' not in st.session_state and CLAUDE_AVAILABLE:
        try:
            st.session_state.claude_assistant = ClaudeAssistant()
        except Exception as e:
            st.session_state.claude_assistant = None


def load_csv_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and validate CSV data with robust handling of different formats
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Handle first_name/last_name format
        if 'first_name' in df.columns and 'last_name' in df.columns:
            st.info("✓ Detected first_name/last_name format - combining names...")
            df['name'] = df['first_name'].fillna('') + ' ' + df['last_name'].fillna('')
            df['name'] = df['name'].str.strip()
        
        # Clean all names
        df['name'] = df['name'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        
        # Validate required columns
        required_cols = ['name', 'team', 'position', 'salary', 'projection']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return None
        
        # Add ownership if not present
        if 'ownership' not in df.columns:
            df['ownership'] = 15.0
            st.info("✓ No ownership column - will use AI prediction or defaults")
        
        # Calculate ceiling/floor if not present
        if 'ceiling' not in df.columns:
            df['ceiling'] = df['projection'] * 1.3
            st.info("✓ Calculated ceiling (projection × 1.3)")
        
        if 'floor' not in df.columns:
            df['floor'] = df['projection'] * 0.7
            st.info("✓ Calculated floor (projection × 0.7)")
        
        # Validate data types
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        df['projection'] = pd.to_numeric(df['projection'], errors='coerce')
        df['ownership'] = pd.to_numeric(df['ownership'], errors='coerce')
        df['ceiling'] = pd.to_numeric(df['ceiling'], errors='coerce')
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['name', 'salary', 'projection'])
        
        if len(df) == 0:
            st.error("No valid player data after processing")
            return None
        
        st.success(f"✅ CSV processed successfully! {len(df)} players ready")
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None


def player_input_section():
    """Section for uploading or manually entering player data"""
    st.header("📊 Player Data Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload CSV", "Manual Entry"],
        horizontal=True
    )
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload player pool CSV",
            type=['csv'],
            help="CSV must include: name, team, position, salary, projection"
        )
        
        if uploaded_file is not None:
            df = load_csv_data(uploaded_file)
            if df is not None:
                st.session_state.players_df = df
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df, use_container_width=True)
                
                # Show summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Players", len(df))
                with col2:
                    st.metric("Avg Salary", f"${df['salary'].mean():,.0f}")
                with col3:
                    st.metric("Avg Projection", f"{df['projection'].mean():.1f}")
                
                # AI Ownership Prediction
                if CLAUDE_AVAILABLE and st.session_state.claude_assistant:
                    if st.button("🤖 Predict Ownership with AI", type="primary"):
                        with st.spinner("🤖 AI is predicting ownership percentages..."):
                            try:
                                # Call Claude to predict ownership (batch method for all players)
                                updated_df = st.session_state.claude_assistant.batch_predict_ownership(df.copy())
                                
                                if updated_df is not None and 'ownership' in updated_df.columns:
                                    # Ensure ownership is numeric
                                    updated_df['ownership'] = pd.to_numeric(
                                        updated_df['ownership'], 
                                        errors='coerce'
                                    ).fillna(15.0)
                                    
                                    st.session_state.players_df = updated_df
                                    st.success("✅ AI ownership prediction complete!")
                                    
                                    # Show ownership distribution
                                    st.subheader("Predicted Ownership Distribution")
                                    
                                    # Create summary with explicit formatting
                                    top_owned = updated_df.nlargest(10, 'ownership').copy()
                                    summary_data = []
                                    for _, row in top_owned.iterrows():
                                        summary_data.append({
                                            'Player': str(row['name']),
                                            'Position': str(row['position']),
                                            'Salary': f"${int(row['salary']):,}",
                                            'Projection': f"{float(row['projection']):.1f}",
                                            'Ownership': f"{float(row['ownership']):.1f}%"
                                        })
                                    
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                else:
                                    st.warning("AI prediction returned no data, using defaults")
                                    
                            except Exception as e:
                                st.error(f"AI prediction error: {str(e)}")
                                st.info("Continuing with default 15% ownership for all players")
    
    else:  # Manual Entry
        st.info("Manual entry feature - Coming soon")
        st.write("For now, please use CSV upload")


def opponent_modeling_section():
    """Section for opponent modeling analysis"""
    st.header("🎯 Opponent Modeling")
    
    if st.session_state.players_df is None:
        st.warning("⚠️ Please load player data first")
        return
    
    if st.button("🔍 Run Opponent Modeling Analysis", type="primary"):
        with st.spinner("Analyzing player leverage and field behavior..."):
            try:
                # Initialize opponent model
                opponent_model = OpponentModel(st.session_state.players_df)
                st.session_state.opponent_model = opponent_model
                
                # Calculate metrics
                analysis = opponent_model.analyze_field_dynamics()
                
                # Display results
                st.success("✅ Analysis complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔥 High Leverage Plays")
                    leverage_data = analysis['leverage_scores'].nlargest(10, 'leverage_score')
                    
                    # Format for display
                    display_data = []
                    for _, row in leverage_data.iterrows():
                        display_data.append({
                            'Player': str(row['name']),
                            'Pos': str(row['position']),
                            'Proj': f"{float(row['projection']):.1f}",
                            'Own%': f"{float(row['ownership']):.1f}",
                            'Leverage': f"{float(row['leverage_score']):.1f}"
                        })
                    
                    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("📈 Chalk Plays")
                    chalk_data = analysis['chalk_players'].head(10)
                    
                    # Format for display
                    display_data = []
                    for _, row in chalk_data.iterrows():
                        display_data.append({
                            'Player': str(row['name']),
                            'Pos': str(row['position']),
                            'Proj': f"{float(row['projection']):.1f}",
                            'Own%': f"{float(row['ownership']):.1f}"
                        })
                    
                    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
                
                # Ownership distribution
                st.subheader("📊 Ownership Distribution")
                bins = [0, 10, 20, 30, 50, 100]
                labels = ['0-10%', '10-20%', '20-30%', '30-50%', '50%+']
                ownership_dist = pd.cut(
                    st.session_state.players_df['ownership'],
                    bins=bins,
                    labels=labels
                ).value_counts().sort_index()
                
                st.bar_chart(ownership_dist)
                
            except Exception as e:
                st.error(f"Error in opponent modeling: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def lineup_optimization_section():
    """Section for generating optimized lineups"""
    st.header("⚡ Lineup Optimization")
    
    if st.session_state.players_df is None:
        st.warning("⚠️ Please load player data first")
        return
    
    if st.session_state.opponent_model is None:
        st.warning("⚠️ Please run opponent modeling analysis first")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_lineups = st.number_input(
            "Number of lineups",
            min_value=1,
            max_value=150,
            value=20,
            step=1
        )
    
    with col2:
        mode = st.selectbox(
            "Optimization mode",
            options=list(OPTIMIZATION_MODES.keys()),
            index=1  # Default to "balanced"
        )
    
    with col3:
        max_exposure = st.slider(
            "Max player exposure %",
            min_value=10,
            max_value=100,
            value=40,
            step=5
        )
    
    if st.button("🚀 Generate Lineups", type="primary"):
        with st.spinner(f"Generating {num_lineups} optimized lineups..."):
            try:
                # Initialize optimizer if needed
                if st.session_state.optimizer is None:
                    st.session_state.optimizer = LineupOptimizer(
                        st.session_state.players_df,
                        st.session_state.opponent_model
                    )
                
                # Generate lineups
                lineups = st.session_state.optimizer.generate_lineups(
                    num_lineups=num_lineups,
                    mode=mode,
                    max_exposure=max_exposure / 100
                )
                
                st.session_state.generated_lineups = lineups
                
                st.success(f"✅ Generated {len(lineups)} unique lineups!")
                
                # Show portfolio summary
                if lineups:
                    portfolio = st.session_state.optimizer.get_portfolio_analysis()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Projection", f"{portfolio['avg_projection']:.1f}")
                    with col2:
                        st.metric("Avg Ownership", f"{portfolio['avg_ownership']:.1f}%")
                    with col3:
                        st.metric("Avg Leverage", f"{portfolio['avg_leverage']:.2f}")
                    with col4:
                        st.metric("Unique Lineups", f"{portfolio['unique_lineups']}/{len(lineups)}")
                
            except Exception as e:
                st.error(f"Error generating lineups: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def lineup_display_section():
    """Section for displaying and exporting generated lineups"""
    st.header("📋 Generated Lineups")
    
    if not st.session_state.generated_lineups:
        st.info("No lineups generated yet")
        return
    
    lineups = st.session_state.generated_lineups
    
    # Display options
    col1, col2 = st.columns([3, 1])
    with col1:
        display_mode = st.radio(
            "Display mode:",
            ["Summary", "Detailed View"],
            horizontal=True
        )
    with col2:
        if st.button("📥 Export to CSV"):
            try:
                st.session_state.optimizer.export_to_csv('lineups.csv')
                st.success("✅ Exported to lineups.csv")
            except Exception as e:
                st.error(f"Export error: {str(e)}")
    
    if display_mode == "Summary":
        # Create summary dataframe with explicit formatting
        summary_data = []
        for i, lineup in enumerate(lineups, 1):
            metrics = lineup['metrics']
            summary_data.append({
                'Lineup': i,
                'Captain': str(lineup['captain']),
                'Projection': f"{float(metrics.get('total_projection', 0)):.1f}",
                'Ceiling': f"{float(metrics.get('total_ceiling', 0)):.1f}",
                'Salary': f"${int(metrics.get('total_salary', 0)):,}",
                'Ownership': f"{float(metrics.get('avg_ownership', 0)):.1f}%",
                'Leverage': f"{float(metrics.get('lineup_leverage', 0)):.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    else:  # Detailed View
        lineup_num = st.selectbox(
            "Select lineup to view:",
            range(1, len(lineups) + 1)
        )
        
        lineup = lineups[lineup_num - 1]
        
        st.subheader(f"Lineup {lineup_num} Details")
        
        # Lineup metrics
        metrics = lineup['metrics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Projection", f"{float(metrics.get('total_projection', 0)):.1f}")
        with col2:
            st.metric("Salary", f"${int(metrics.get('total_salary', 0)):,}")
        with col3:
            st.metric("Ownership", f"{float(metrics.get('avg_ownership', 0)):.1f}%")
        with col4:
            st.metric("Leverage", f"{float(metrics.get('lineup_leverage', 0)):.2f}")
        
        # Player details
        st.subheader("Players")
        
        # Captain
        st.markdown(f"**👑 Captain (CPT):** {lineup['captain']}")
        
        # Flex players
        st.markdown("**FLEX:**")
        for i, player in enumerate(lineup['flex'], 1):
            st.markdown(f"{i}. {player}")


def sidebar_info():
    """Sidebar with app information and settings"""
    with st.sidebar:
        st.title("🎯 DFS Meta-Optimizer")
        st.markdown("---")
        
        st.subheader("📖 About")
        st.write("""
        This optimizer uses **opponent modeling** to find lineups
        that maximize your competitive advantage, not just points.
        """)
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        st.write(f"**Salary Cap:** ${SALARY_CAP:,}")
        st.write(f"**Roster Size:** {ROSTER_SIZE} ({ROSTER_SIZE-1} flex + 1 captain)")
        st.write(f"**Captain Multiplier:** {CAPTAIN_MULTIPLIER}x")
        
        if CLAUDE_AVAILABLE and st.session_state.get('claude_assistant'):
            st.success("✅ Claude AI: Active")
        else:
            st.warning("⚠️ Claude AI: Inactive")
        
        st.markdown("---")
        st.subheader("📊 Optimization Modes")
        
        for mode, config in OPTIMIZATION_MODES.items():
            st.write(f"**{mode.title()}:**")
            st.write(f"- Projection: {config.get('projection_weight', 0)}")
            st.write(f"- Leverage: {config.get('leverage_weight', 0)}")
            st.write(f"- Ceiling: {config.get('ceiling_weight', 0)}")
            st.write("")


def main():
    """Main application function"""
    initialize_session_state()
    sidebar_info()
    
    st.title("🎯 DFS Meta-Optimizer")
    st.markdown("*Optimize for competitive advantage, not just points*")
    st.markdown("---")
    
    # Main sections
    player_input_section()
    st.markdown("---")
    opponent_modeling_section()
    st.markdown("---")
    lineup_optimization_section()
    st.markdown("---")
    lineup_display_section()


if __name__ == "__main__":
    main()
