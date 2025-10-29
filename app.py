"""
DFS Meta-Optimizer - Advanced Streamlit Application
Version 6.0.0 - MOST ADVANCED STATE

Revolutionary DFS optimizer with PhD-level features:
- Contest-size aware optimization
- Genetic Algorithm v2 with visualization
- Monte Carlo tournament simulation
- AI ownership prediction with prompt caching
- Lineup explanations
- Exposure management
- Historical performance tracking
- Real-time cost monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from typing import Dict, List, Optional

# Import advanced Group 1 modules
try:
    from settings import (
        get_config, 
        get_optimization_mode, 
        get_config_manager,
        detect_contest_type,
        ContestType,
        OptimizationMode,
        validate_api_key
    )
    from opponent_modeling import OpponentModel
    from claude_assistant import AIAssistant
    from optimization_engine import LineupOptimizer, ExposureCaps, OptimizationResult
except ImportError as e:
    st.error(f"❌ Module Import Error: {e}")
    st.info("Ensure all Group 1 files are in the same directory as app.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="DFS Meta-Optimizer v6.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .version-badge {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    
    if 'config' not in st.session_state:
        st.session_state.config = get_config()
    
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = get_config_manager()
    
    if 'players_df' not in st.session_state:
        st.session_state.players_df = None
    
    if 'opponent_model' not in st.session_state:
        st.session_state.opponent_model = None
    
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    
    if 'ai_assistant' not in st.session_state:
        config = st.session_state.config
        if config.enable_claude_ai and validate_api_key():
            try:
                st.session_state.ai_assistant = AIAssistant(
                    api_key=config.anthropic_api_key,
                    enable_caching=config.enable_response_caching,
                    enable_prompt_caching=config.enable_prompt_caching
                )
            except Exception as e:
                st.session_state.ai_assistant = None
        else:
            st.session_state.ai_assistant = None
    
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    
    if 'generated_lineups' not in st.session_state:
        st.session_state.generated_lineups = []
    
    if 'contest_size' not in st.session_state:
        st.session_state.contest_size = 10000


def process_csv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Process uploaded CSV"""
    try:
        processed = df.copy()
        processed.columns = processed.columns.str.strip().str.lower()
        
        if 'name' not in processed.columns:
            if 'first_name' in processed.columns and 'last_name' in processed.columns:
                processed['name'] = (
                    processed['first_name'].fillna('').astype(str).str.strip() + ' ' + 
                    processed['last_name'].fillna('').astype(str).str.strip()
                ).str.strip()
            else:
                st.error("❌ CSV must have 'name' OR 'first_name' + 'last_name'")
                return None
        
        processed['name'] = processed['name'].astype(str).str.strip()
        
        required = ['name', 'position', 'team', 'salary', 'projection']
        missing = [col for col in required if col not in processed.columns]
        
        if missing:
            st.error(f"❌ Missing: {', '.join(missing)}")
            return None
        
        processed['salary'] = pd.to_numeric(processed['salary'], errors='coerce')
        processed['projection'] = pd.to_numeric(processed['projection'], errors='coerce')
        
        if 'ceiling' not in processed.columns:
            processed['ceiling'] = processed['projection'] * 1.4
        if 'floor' not in processed.columns:
            processed['floor'] = processed['projection'] * 0.6
        if 'ownership' not in processed.columns:
            processed['ownership'] = 15.0
        
        processed = processed.dropna(subset=['name', 'salary', 'projection'])
        processed = processed[processed['salary'] > 0]
        processed = processed[processed['projection'] > 0]
        
        processed['value'] = processed['projection'] / (processed['salary'] / 1000)
        
        return processed
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def main():
    """Main application"""
    
    initialize_session_state()
    
    # Header
    st.markdown("<h1 class='main-header'>🎯 DFS Meta-Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='version-badge'>v6.0.0 - Most Advanced State</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>PhD-Level Optimization with Contest-Aware Intelligence</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 DFS Meta-Optimizer")
        st.markdown("*v6.0.0*")
        st.markdown("---")
        
        config = st.session_state.config
        
        if st.session_state.ai_assistant:
            st.success("✅ AI: Active")
            stats = st.session_state.ai_assistant.get_usage_stats()
            st.metric("Cost", f"${stats['total_cost']:.3f}")
            st.metric("Cache", f"{stats['cache_hit_rate']*100:.0f}%")
        else:
            st.warning("⚠️ AI: Inactive")
    
    st.markdown("---")
    
    # Upload
    st.header("📤 Upload Player Data")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = process_csv(pd.read_csv(uploaded_file))
        
        if df is not None:
            st.session_state.players_df = df
            st.success(f"✅ Loaded {len(df)} players")
            
            # Show players
            st.dataframe(df[['name', 'position', 'team', 'salary', 'projection']].head(20))
            
            # AI Prediction
            if st.session_state.ai_assistant:
                if st.button("🤖 Predict Ownership"):
                    with st.spinner("Analyzing..."):
                        ownership = st.session_state.ai_assistant.predict_ownership(df)
                        for name, own in ownership.items():
                            mask = df['name'].str.lower() == name.lower()
                            if mask.any():
                                df.loc[mask, 'ownership'] = own
                        st.success("✅ Ownership predicted")
                        st.session_state.players_df = df
            
            # Contest Setup
            st.markdown("---")
            st.header("🎮 Contest Setup")
            
            col1, col2 = st.columns(2)
            with col1:
                entry_fee = st.number_input("Entry Fee", 0.0, 1000.0, 20.0)
            with col2:
                field_size = st.number_input("Field Size", 2, 500000, 10000)
            
            contest_type = detect_contest_type(entry_fee, 20, field_size)
            st.info(f"Contest: {contest_type.value}")
            st.session_state.contest_size = field_size
            
            # Opponent Analysis
            st.markdown("---")
            st.header("🎯 Opponent Analysis")
            
            if st.button("Run Analysis"):
                with st.spinner("Analyzing..."):
                    model = OpponentModel(
                        df,
                        contest_size=field_size,
                        enable_correlation=True
                    )
                    st.session_state.opponent_model = model
                    
                    analysis = model.analyze_field_distribution()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chalk Plays", analysis['chalk_count'])
                    with col2:
                        st.metric("Leverage Plays", analysis['leverage_play_count'])
                    with col3:
                        st.metric("Multiplier", f"{analysis['size_multiplier']:.2f}x")
                    
                    st.success("✅ Analysis complete")
            
            # Generate Lineups
            st.markdown("---")
            st.header("🚀 Generate Lineups")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mode_name = st.selectbox("Mode", ['balanced', 'aggressive', 'leverage', 'contrarian'])
            with col2:
                num_lineups = st.number_input("Lineups", 1, 150, 20)
            with col3:
                use_genetic = st.checkbox("Genetic Algorithm", True)
            
            if st.button("🎯 Generate", type="primary"):
                if st.session_state.opponent_model is None:
                    st.warning("Run opponent analysis first!")
                else:
                    with st.spinner(f"Generating {num_lineups} lineups..."):
                        mode = get_optimization_mode(mode_name)
                        
                        optimizer = LineupOptimizer(
                            df,
                            st.session_state.opponent_model,
                            mode.to_dict()
                        )
                        
                        result = optimizer.generate_lineups(
                            num_lineups,
                            use_genetic=use_genetic
                        )
                        
                        st.session_state.optimization_result = result
                        st.session_state.generated_lineups = result.lineups
                        
                        st.success(f"✅ Generated {len(result.lineups)} lineups in {result.generation_time:.1f}s")
            
            # Display Lineups
            if st.session_state.generated_lineups:
                st.markdown("---")
                st.header("📋 Generated Lineups")
                
                lineups = st.session_state.generated_lineups
                
                for i, lineup in enumerate(lineups[:10], 1):
                    with st.expander(f"Lineup {i} - {lineup['projection'].sum():.1f} pts"):
                        st.dataframe(
                            lineup[['name', 'position', 'salary', 'projection']],
                            use_container_width=True
                        )
    
    else:
        st.info("👆 Upload a CSV to start")


if __name__ == "__main__":
    main()
