"""
DFS Meta-Optimizer - Main Streamlit Application
Version 5.0.0

Revolutionary DFS optimizer that maximizes competitive advantage through:
- Opponent modeling and leverage scoring
- AI-powered ownership prediction
- Advanced optimization algorithms
- Real-time strategic analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.opponent_modeling import OpponentModeler
    from modules.optimization_engine import LineupOptimizer
    from modules.ai_assistant import AIAssistant
    from config.settings import (
        SALARY_CAP, ROSTER_SIZE, OPTIMIZATION_MODES,
        ENABLE_CLAUDE_AI, AI_OWNERSHIP_PREDICTION,
        HIGH_OWNERSHIP_THRESHOLD, LOW_OWNERSHIP_THRESHOLD
    )
except ImportError as e:
    st.error(f"❌ Module Import Error: {e}")
    st.info("Ensure all module files exist in the correct directories")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="DFS Meta-Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .leverage-high {
        background-color: #d4edda;
        padding: 0.3rem;
        border-radius: 0.3rem;
    }
    .leverage-low {
        background-color: #f8d7da;
        padding: 0.3rem;
        border-radius: 0.3rem;
    }
    .chalk-player {
        background-color: #fff3cd;
        padding: 0.3rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'players_df' not in st.session_state:
        st.session_state.players_df = None
    if 'opponent_model' not in st.session_state:
        st.session_state.opponent_model = None
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'ai_assistant' not in st.session_state:
        if ENABLE_CLAUDE_AI:
            st.session_state.ai_assistant = AIAssistant()
        else:
            st.session_state.ai_assistant = None
    if 'generated_lineups' not in st.session_state:
        st.session_state.generated_lineups = []
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []


def clean_player_name(name):
    """Clean player name for consistent matching"""
    if pd.isna(name):
        return ""
    # Strip whitespace and convert to lowercase
    cleaned = str(name).strip().lower()
    # Remove extra whitespace between names
    cleaned = ' '.join(cleaned.split())
    return cleaned


def process_csv(df):
    """
    Process uploaded CSV into standardized format
    Handles both single 'name' column and separate first_name/last_name columns
    """
    try:
        # Create a clean copy
        processed = df.copy()
        
        # Standardize column names
        processed.columns = processed.columns.str.strip().str.lower()
        
        # Handle name columns
        if 'name' in processed.columns:
            # Single name column - split it
            name_parts = processed['name'].str.strip().str.split(n=1, expand=True)
            processed['first_name'] = name_parts[0] if len(name_parts.columns) > 0 else ''
            processed['last_name'] = name_parts[1] if len(name_parts.columns) > 1 else ''
        elif 'first_name' in processed.columns and 'last_name' in processed.columns:
            # Already has separate columns
            processed['first_name'] = processed['first_name'].fillna('').str.strip()
            processed['last_name'] = processed['last_name'].fillna('').str.strip()
        else:
            st.error("❌ CSV must have either 'name' column OR 'first_name' and 'last_name' columns")
            return None
        
        # Create full name for display
        processed['player_name'] = (
            processed['first_name'].str.strip() + ' ' + 
            processed['last_name'].str.strip()
        ).str.strip()
        
        # Create clean name for matching
        processed['clean_name'] = processed['player_name'].apply(clean_player_name)
        
        # Required columns
        required_cols = ['position', 'team', 'salary', 'projection']
        
        # Check for required columns
        missing = [col for col in required_cols if col not in processed.columns]
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.info("Required: position, team, salary, projection")
            return None
        
        # Convert data types
        processed['salary'] = pd.to_numeric(processed['salary'], errors='coerce')
        processed['projection'] = pd.to_numeric(processed['projection'], errors='coerce')
        
        # Add derived columns if missing
        if 'ceiling' not in processed.columns:
            processed['ceiling'] = processed['projection'] * 1.3
        else:
            processed['ceiling'] = pd.to_numeric(processed['ceiling'], errors='coerce')
            
        if 'floor' not in processed.columns:
            processed['floor'] = processed['projection'] * 0.7
        else:
            processed['floor'] = pd.to_numeric(processed['floor'], errors='coerce')
            
        if 'ownership' not in processed.columns:
            processed['ownership'] = 15.0  # Default 15%
        else:
            processed['ownership'] = pd.to_numeric(processed['ownership'], errors='coerce')
        
        # Clean data
        processed = processed.dropna(subset=['player_name', 'salary', 'projection'])
        processed = processed[processed['salary'] > 0]
        processed = processed[processed['projection'] > 0]
        
        # Calculate value
        processed['value'] = processed['projection'] / (processed['salary'] / 1000)
        
        return processed
        
    except Exception as e:
        st.error(f"❌ Error processing CSV: {str(e)}")
        return None


def predict_ownership_ai(df):
    """Use Claude AI to predict player ownership percentages"""
    if not ENABLE_CLAUDE_AI or not st.session_state.ai_assistant:
        st.warning("⚠️ Claude AI not available - using default 15% ownership")
        return df
    
    try:
        with st.spinner("🤖 AI analyzing field ownership..."):
            # Prepare player list for AI
            player_data = df[['player_name', 'position', 'salary', 'projection']].to_dict('records')
            
            # Get predictions from AI
            ownership_dict = st.session_state.ai_assistant.predict_ownership(player_data)
            
            if ownership_dict and len(ownership_dict) > 0:
                # Update ownership in dataframe using clean names
                for clean_name, ownership in ownership_dict.items():
                    mask = df['clean_name'] == clean_name
                    if mask.any():
                        df.loc[mask, 'ownership'] = ownership
                
                st.success(f"✅ AI predicted ownership for {len(ownership_dict)} players")
                
                # Show distribution
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Ownership", f"{df['ownership'].mean():.1f}%")
                with col2:
                    st.metric("Min Ownership", f"{df['ownership'].min():.1f}%")
                with col3:
                    st.metric("Max Ownership", f"{df['ownership'].max():.1f}%")
            else:
                st.warning("⚠️ AI prediction returned no results - using defaults")
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ AI prediction failed: {str(e)}")
        st.info("Using default 15% ownership for all players")
        return df


def display_player_pool(df):
    """Display the processed player pool with statistics"""
    st.subheader("📊 Player Pool")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("Avg Salary", f"${df['salary'].mean():,.0f}")
    with col3:
        st.metric("Avg Projection", f"{df['projection'].mean():.1f}")
    with col4:
        st.metric("Avg Value", f"{df['value'].mean():.2f}")
    
    # Filters
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        position_filter = st.multiselect(
            "Filter by Position",
            options=sorted(df['position'].unique()),
            default=sorted(df['position'].unique())
        )
    
    with col2:
        team_filter = st.multiselect(
            "Filter by Team",
            options=sorted(df['team'].unique()),
            default=sorted(df['team'].unique())
        )
    
    # Apply filters
    filtered_df = df[
        (df['position'].isin(position_filter)) &
        (df['team'].isin(team_filter))
    ]
    
    # Display dataframe
    display_cols = ['player_name', 'position', 'team', 'salary', 'projection', 
                    'ceiling', 'floor', 'ownership', 'value']
    
    st.dataframe(
        filtered_df[display_cols].sort_values('projection', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    return filtered_df


def run_opponent_modeling(df):
    """Run opponent modeling analysis"""
    st.subheader("🎯 Opponent Modeling Analysis")
    
    try:
        with st.spinner("Analyzing field construction patterns..."):
            # Create opponent modeler
            modeler = OpponentModeler(df)
            st.session_state.opponent_model = modeler
            
            # Calculate metrics
            chalk_players = modeler.identify_chalk_plays()
            leverage_plays = modeler.identify_leverage_opportunities()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔥 Chalk Plays (High Ownership)")
                if len(chalk_players) > 0:
                    for _, player in chalk_players.head(10).iterrows():
                        st.markdown(
                            f"<div class='chalk-player'>"
                            f"**{player['player_name']}** ({player['position']}) - "
                            f"${player['salary']:,} | {player['ownership']:.1f}% owned"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No high-ownership players identified")
            
            with col2:
                st.markdown("#### 💎 Leverage Opportunities")
                if len(leverage_plays) > 0:
                    for _, player in leverage_plays.head(10).iterrows():
                        leverage_score = player.get('leverage_score', 0)
                        st.markdown(
                            f"<div class='leverage-high'>"
                            f"**{player['player_name']}** ({player['position']}) - "
                            f"${player['salary']:,} | Leverage: {leverage_score:.2f}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No leverage opportunities identified")
            
            # Visualization
            st.markdown("---")
            st.markdown("#### 📈 Ownership vs Value")
            
            fig = px.scatter(
                df,
                x='value',
                y='ownership',
                size='projection',
                color='position',
                hover_data=['player_name', 'salary', 'projection'],
                title="Player Value vs Expected Ownership",
                labels={'value': 'Value (Proj/1K)', 'ownership': 'Ownership %'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Opponent modeling complete!")
            return True
            
    except Exception as e:
        st.error(f"❌ Error in opponent modeling: {str(e)}")
        return False


def generate_lineups_section(df):
    """Lineup generation section"""
    st.subheader("🚀 Generate Lineups")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mode = st.selectbox(
            "Optimization Mode",
            options=list(OPTIMIZATION_MODES.keys()),
            format_func=lambda x: x.title()
        )
    
    with col2:
        num_lineups = st.number_input(
            "Number of Lineups",
            min_value=1,
            max_value=20,
            value=3
        )
    
    with col3:
        uniqueness = st.slider(
            "Uniqueness",
            min_value=1,
            max_value=9,
            value=5,
            help="Higher = more different lineups"
        )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_exposure = st.slider(
                "Max Player Exposure %",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            ) / 100.0
        
        with col2:
            min_salary_pct = st.slider(
                "Min Salary Usage %",
                min_value=90,
                max_value=100,
                value=96
            ) / 100.0
    
    if st.button("🎯 Generate Lineups", type="primary", use_container_width=True):
        if df is None or len(df) == 0:
            st.error("❌ No player data available!")
            return
        
        try:
            with st.spinner(f"Generating {num_lineups} {mode} lineups..."):
                # Create optimizer
                optimizer = LineupOptimizer(
                    df,
                    mode=mode,
                    opponent_model=st.session_state.opponent_model
                )
                st.session_state.optimizer = optimizer
                
                # Generate lineups
                lineups = optimizer.generate_lineups(
                    num_lineups=num_lineups,
                    uniqueness=uniqueness,
                    max_exposure=max_exposure,
                    min_salary_pct=min_salary_pct
                )
                
                if lineups and len(lineups) > 0:
                    st.session_state.generated_lineups = lineups
                    st.success(f"✅ Generated {len(lineups)} unique lineups!")
                else:
                    st.error("❌ Failed to generate lineups")
                    
        except Exception as e:
            st.error(f"❌ Error generating lineups: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_lineups_section():
    """Display generated lineups"""
    if not st.session_state.generated_lineups:
        st.info("💡 Generate lineups to see them here")
        return
    
    st.subheader("📋 Generated Lineups")
    
    lineups = st.session_state.generated_lineups
    
    # Summary metrics
    total_proj = sum(lu['total_projection'] for lu in lineups)
    avg_proj = total_proj / len(lineups) if lineups else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lineups Generated", len(lineups))
    with col2:
        st.metric("Avg Projection", f"{avg_proj:.1f}")
    with col3:
        st.metric("Avg Salary", f"${sum(lu['total_salary'] for lu in lineups) / len(lineups):,.0f}")
    with col4:
        st.metric("Avg Ownership", f"{sum(lu.get('avg_ownership', 0) for lu in lineups) / len(lineups):.1f}%")
    
    # Display each lineup
    for i, lineup in enumerate(lineups, 1):
        with st.expander(f"Lineup {i} - Proj: {lineup['total_projection']:.1f} | Salary: ${lineup['total_salary']:,}"):
            
            # Convert to dataframe
            lineup_df = pd.DataFrame(lineup['players'])
            
            # Display lineup
            st.dataframe(
                lineup_df[['player_name', 'position', 'team', 'salary', 'projection', 'ownership']],
                use_container_width=True,
                hide_index=True
            )
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Projection", f"{lineup['total_projection']:.1f}")
            with col2:
                st.metric("Total Salary", f"${lineup['total_salary']:,}")
            with col3:
                st.metric("Avg Ownership", f"{lineup.get('avg_ownership', 0):.1f}%")


def sidebar_content():
    """Sidebar with app info and settings"""
    with st.sidebar:
        st.markdown("## 🎯 DFS Meta-Optimizer")
        st.markdown("*v5.0.0*")
        st.markdown("---")
        
        st.markdown("### 📖 About")
        st.markdown("""
        This optimizer uses **opponent modeling** to find lineups that maximize 
        your competitive advantage, not just projected points.
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        st.write(f"**Salary Cap:** ${SALARY_CAP:,}")
        st.write(f"**Roster Size:** {ROSTER_SIZE}")
        
        if ENABLE_CLAUDE_AI and st.session_state.ai_assistant:
            st.success("✅ Claude AI: Active")
        else:
            st.warning("⚠️ Claude AI: Inactive")
        
        st.markdown("---")
        st.markdown("### 📊 Optimization Modes")
        
        for mode_name, mode_config in OPTIMIZATION_MODES.items():
            with st.expander(mode_name.title()):
                st.write(f"**{mode_config['description']}**")
                st.write(f"- Projection Weight: {mode_config['projection_weight']}")
                st.write(f"- Leverage Weight: {mode_config['leverage_weight']}")
                st.write(f"- Ceiling Weight: {mode_config['ceiling_weight']}")


def main():
    """Main application function"""
    initialize_session_state()
    sidebar_content()
    
    # Header
    st.markdown("<h1 class='main-header'>🎯 DFS Meta-Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Maximize Competitive Advantage Through Opponent Modeling</p>", unsafe_allow_html=True)
    
    # Step 1: Upload Data
    st.markdown("---")
    st.header("📤 Step 1: Upload Player Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with player data",
        type=['csv'],
        help="CSV must include: name (or first_name/last_name), position, team, salary, projection"
    )
    
    if uploaded_file is not None:
        # Process CSV
        with st.spinner("Processing player data..."):
            df = process_csv(pd.read_csv(uploaded_file))
        
        if df is not None:
            st.session_state.players_df = df
            st.success(f"✅ Loaded {len(df)} players successfully!")
            
            # AI Ownership Prediction
            if ENABLE_CLAUDE_AI and AI_OWNERSHIP_PREDICTION:
                if st.button("🤖 Predict Ownership with AI", type="secondary"):
                    st.session_state.players_df = predict_ownership_ai(df)
                    df = st.session_state.players_df
            
            # Display player pool
            st.markdown("---")
            filtered_df = display_player_pool(df)
            
            # Step 2: Opponent Modeling
            st.markdown("---")
            st.header("🎯 Step 2: Opponent Modeling")
            
            if st.button("Run Opponent Analysis", type="secondary", use_container_width=True):
                run_opponent_modeling(df)
            
            # Step 3: Generate Lineups
            st.markdown("---")
            st.header("🚀 Step 3: Generate Lineups")
            generate_lineups_section(df)
            
            # Step 4: View Lineups
            st.markdown("---")
            st.header("📋 Step 4: Review Lineups")
            display_lineups_section()
    
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Show example format
        with st.expander("📄 See Example CSV Format"):
            example_df = pd.DataFrame({
                'name': ['Patrick Mahomes', 'Travis Kelce'],
                'position': ['QB', 'TE'],
                'team': ['KC', 'KC'],
                'salary': [8500, 6800],
                'projection': [24.5, 16.2]
            })
            st.dataframe(example_df)


if __name__ == "__main__":
    main()
