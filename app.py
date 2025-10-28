"""
DFS Meta-Optimizer - Streamlit Application

A revolutionary DFS optimizer that maximizes competitive advantage
rather than just projected points.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import traceback

# Import modules
try:
    from modules.optimization_engine import LineupOptimizer
    from modules.opponent_modeling import OpponentModeler
    from modules.ai_assistant import AIAssistant
    from settings import OPTIMIZATION_MODES
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("Make sure all module files are in the 'modules/' directory")
    st.stop()

# Page config
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
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def clean_player_name(name):
    """Clean player name for consistent matching"""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()

def load_player_data(uploaded_file):
    """Load and process player data with robust CSV handling"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle first_name/last_name columns if present
        if 'first_name' in df.columns and 'last_name' in df.columns:
            df['name'] = df['first_name'].fillna('').astype(str).str.strip() + ' ' + \
                         df['last_name'].fillna('').astype(str).str.strip()
            df['name'] = df['name'].str.strip()
        elif 'name' not in df.columns:
            st.error("❌ CSV must have either 'name' column or 'first_name' and 'last_name' columns")
            return None
        
        # Filter out DST entries (Defense/Special Teams)
        if 'position' in df.columns:
            original_count = len(df)
            df = df[df['position'].str.upper() != 'DST'].copy()
            dst_filtered = original_count - len(df)
            if dst_filtered > 0:
                st.info(f"ℹ️ Filtered out {dst_filtered} DST entries")
        
        # Required columns
        required = ['name', 'team', 'position', 'salary', 'projection']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.info("Required columns: name, team, position, salary, projection")
            return None
        
        # Convert to numeric
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        df['projection'] = pd.to_numeric(df['projection'], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['name', 'salary', 'projection'])
        df = df[df['salary'] > 0]
        df = df[df['projection'] > 0]
        
        # Calculate ceiling and floor if not present
        if 'ceiling' not in df.columns:
            df['ceiling'] = df['projection'] * 1.4
        else:
            df['ceiling'] = pd.to_numeric(df['ceiling'], errors='coerce').fillna(df['projection'] * 1.4)
            
        if 'floor' not in df.columns:
            df['floor'] = df['projection'] * 0.7
        else:
            df['floor'] = pd.to_numeric(df['floor'], errors='coerce').fillna(df['projection'] * 0.7)
        
        # Add ownership if not present (will be updated by AI if enabled)
        if 'ownership' not in df.columns:
            df['ownership'] = 15.0  # Default placeholder
        else:
            df['ownership'] = pd.to_numeric(df['ownership'], errors='coerce').fillna(15.0)
        
        # Clean all text fields
        df['name'] = df['name'].astype(str).str.strip()
        df['team'] = df['team'].astype(str).str.strip()
        df['position'] = df['position'].astype(str).str.strip().str.upper()
        
        # Remove any duplicate names (keep first occurrence)
        df = df.drop_duplicates(subset=['name'], keep='first')
        
        st.success(f"✅ Loaded {len(df)} players successfully")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        st.code(traceback.format_exc())
        return None

def display_lineup(lineup_df, lineup_num):
    """Display a single lineup with formatting"""
    st.markdown(f"### Lineup #{lineup_num}")
    
    # Format the display
    display_df = lineup_df[['name', 'team', 'position', 'salary', 'projection', 'ownership']].copy()
    display_df.columns = ['Player', 'Team', 'Pos', 'Salary', 'Proj', 'Own%']
    display_df['Salary'] = display_df['Salary'].apply(lambda x: f"${x:,.0f}")
    display_df['Proj'] = display_df['Proj'].apply(lambda x: f"{x:.1f}")
    display_df['Own%'] = display_df['Own%'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_salary = lineup_df['salary'].sum()
        pct = (total_salary / 50000) * 100
        st.metric("Total Salary", f"${total_salary:,.0f}", f"{pct:.1f}%")
    with col2:
        total_proj = lineup_df['projection'].sum()
        st.metric("Total Projection", f"{total_proj:.1f}")
    with col3:
        avg_own = lineup_df['ownership'].mean()
        st.metric("Avg Ownership", f"{avg_own:.1f}%")
    with col4:
        leverage = lineup_df['ceiling'].sum() / max(avg_own, 0.1)
        st.metric("Leverage Score", f"{leverage:.1f}")

def main():
    # Header
    st.markdown('<div class="main-header">🎯 DFS Meta-Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Beat the field, not just the slate</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("1. Upload Player Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV must include: name, team, position, salary, projection"
        )
        
        if uploaded_file:
            if 'player_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
                with st.spinner("Loading player data..."):
                    df = load_player_data(uploaded_file)
                    if df is not None:
                        st.session_state.player_data = df
                        st.session_state.last_upload = uploaded_file.name
                        st.rerun()
        
        # Optimization settings
        if 'player_data' in st.session_state:
            st.subheader("2. Optimization Mode")
            
            mode = st.selectbox(
                "Select Mode",
                options=list(OPTIMIZATION_MODES.keys()),
                format_func=lambda x: f"{x.title()} - {OPTIMIZATION_MODES[x].get('description', '')[:30]}..."
            )
            
            st.subheader("3. Lineup Settings")
            
            num_lineups = st.slider("Number of Lineups", 1, 150, 20)
            
            diversity = st.slider(
                "Diversity Factor",
                0.0, 1.0, 0.3,
                help="Higher = more unique lineups"
            )
            
            min_salary_pct = st.slider(
                "Min Salary Usage %",
                90, 100, 98,
                help="Minimum percentage of salary cap to use"
            )
            
            st.subheader("4. AI Features")
            
            use_ai = st.checkbox(
                "Enable AI Ownership Prediction",
                value=False,
                help="Uses Claude API to predict ownership (costs ~$0.30 per run)"
            )
            
            if use_ai:
                api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    value=st.secrets.get("ANTHROPIC_API_KEY", ""),
                    help="Get your API key from console.anthropic.com"
                )
                if api_key:
                    st.session_state.anthropic_api_key = api_key
            
            # Generate button
            st.markdown("---")
            generate_btn = st.button("🚀 Generate Lineups", type="primary", use_container_width=True)
        else:
            generate_btn = False
            st.info("👆 Upload player data to begin")
    
    # Main content area
    if 'player_data' not in st.session_state:
        # Welcome screen
        st.info("👈 Upload your player data CSV to get started")
        
        with st.expander("📋 CSV Format Requirements"):
            st.markdown("""
            **Required columns:**
            - `name` (or `first_name` + `last_name`)
            - `team`
            - `position`
            - `salary`
            - `projection`
            
            **Optional columns:**
            - `ceiling` (auto-calculated if missing)
            - `floor` (auto-calculated if missing)
            - `ownership` (can be predicted by AI)
            
            **Example:**
            ```
            name,team,position,salary,projection
            Patrick Mahomes,KC,QB,8500,24.5
            Travis Kelce,KC,TE,7800,18.2
            ```
            """)
        
        with st.expander("🎯 How It Works"):
            st.markdown("""
            This optimizer uses **opponent modeling** and **leverage scoring** to build lineups 
            that maximize your competitive advantage:
            
            1. **Leverage Analysis**: Identifies high-ceiling plays with low ownership
            2. **Opponent Modeling**: Predicts what the field will do
            3. **Smart Diversification**: Creates unique lineups that complement each other
            4. **AI Integration**: Optional AI-powered ownership prediction
            
            **Not just another points optimizer** - this tool helps you beat the field!
            """)
        return
    
    # Show player data summary
    df = st.session_state.player_data
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("Avg Salary", f"${df['salary'].mean():,.0f}")
    with col3:
        st.metric("Avg Projection", f"{df['projection'].mean():.1f}")
    with col4:
        positions = df['position'].nunique()
        st.metric("Positions", positions)
    
    # Generate lineups
    if generate_btn:
        with st.spinner("🧬 Generating optimized lineups..."):
            try:
                # Initialize AI if enabled
                if use_ai and 'anthropic_api_key' in st.session_state:
                    st.info("🤖 Running AI ownership prediction...")
                    ai_assistant = AIAssistant(st.session_state.anthropic_api_key)
                    
                    # Get AI predictions
                    ownership_predictions = ai_assistant.predict_ownership(df)
                    
                    # Update ownership in dataframe
                    for player_name, predicted_own in ownership_predictions.items():
                        mask = df['name'].str.lower() == player_name.lower()
                        if mask.any():
                            df.loc[mask, 'ownership'] = predicted_own
                    
                    st.success(f"✅ AI updated ownership for {len(ownership_predictions)} players")
                    st.session_state.player_data = df
                
                # Initialize opponent modeler
                opponent_modeler = OpponentModeler(df)
                
                # Calculate leverage scores
                df = opponent_modeler.calculate_leverage_scores(df)
                st.session_state.player_data = df
                
                # Initialize optimizer
                optimizer = LineupOptimizer(
                    df,
                    opponent_modeler,
                    mode_config=OPTIMIZATION_MODES[mode]
                )
                
                # Generate lineups
                lineups = optimizer.generate_lineups(
                    num_lineups=num_lineups,
                    diversity_factor=diversity,
                    min_salary_pct=min_salary_pct / 100
                )
                
                if not lineups:
                    st.error("❌ Failed to generate valid lineups. Check your constraints.")
                    return
                
                st.session_state.lineups = lineups
                st.success(f"✅ Generated {len(lineups)} unique lineups!")
                
            except Exception as e:
                st.error(f"❌ Error generating lineups: {str(e)}")
                st.code(traceback.format_exc())
                return
    
    # Display lineups
    if 'lineups' in st.session_state:
        st.markdown("---")
        st.header("📊 Generated Lineups")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Individual Lineups", "📈 Portfolio Analysis", "💾 Export"])
        
        with tab1:
            # Display each lineup
            for i, lineup in enumerate(st.session_state.lineups, 1):
                with st.expander(f"Lineup #{i} - Proj: {lineup['projection'].sum():.1f}, Salary: ${lineup['salary'].sum():,.0f}"):
                    display_lineup(lineup, i)
        
        with tab2:
            st.subheader("Portfolio Metrics")
            
            try:
                lineups = st.session_state.lineups
                
                # Calculate portfolio metrics
                all_projections = [lu['projection'].sum() for lu in lineups]
                all_salaries = [lu['salary'].sum() for lu in lineups]
                all_ownerships = [lu['ownership'].mean() for lu in lineups]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Projection", f"{np.mean(all_projections):.1f}")
                    st.metric("Projection Range", f"{np.min(all_projections):.1f} - {np.max(all_projections):.1f}")
                with col2:
                    st.metric("Avg Salary", f"${np.mean(all_salaries):,.0f}")
                    st.metric("Salary Range", f"${np.min(all_salaries):,.0f} - ${np.max(all_salaries):,.0f}")
                with col3:
                    st.metric("Avg Ownership", f"{np.mean(all_ownerships):.1f}%")
                    st.metric("Ownership Range", f"{np.min(all_ownerships):.1f}% - {np.max(all_ownerships):.1f}%")
                
                # Player exposure
                st.subheader("Player Exposure")
                
                player_counts = {}
                for lineup in lineups:
                    for _, player in lineup.iterrows():
                        name = player['name']
                        player_counts[name] = player_counts.get(name, 0) + 1
                
                exposure_df = pd.DataFrame([
                    {'Player': name, 'Exposure': (count / len(lineups)) * 100}
                    for name, count in player_counts.items()
                ]).sort_values('Exposure', ascending=False)
                
                # Top 10 most exposed
                st.markdown("**Most Exposed Players:**")
                top_exposure = exposure_df.head(10).copy()
                top_exposure['Exposure'] = top_exposure['Exposure'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(top_exposure, use_container_width=True, hide_index=True)
                
                # Least exposed (>0%)
                st.markdown("**Least Exposed Players:**")
                least_exposure = exposure_df[exposure_df['Exposure'] > 0].tail(10).copy()
                least_exposure['Exposure'] = least_exposure['Exposure'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(least_exposure, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error calculating portfolio metrics: {str(e)}")
                st.code(traceback.format_exc())
        
        with tab3:
            st.subheader("Export Lineups")
            
            # Prepare export data
            export_lineups = []
            for i, lineup in enumerate(st.session_state.lineups, 1):
                lineup_dict = {'Lineup': i}
                for idx, (_, player) in enumerate(lineup.iterrows(), 1):
                    lineup_dict[f'Player_{idx}'] = player['name']
                    lineup_dict[f'Pos_{idx}'] = player['position']
                    lineup_dict[f'Salary_{idx}'] = player['salary']
                export_lineups.append(lineup_dict)
            
            export_df = pd.DataFrame(export_lineups)
            
            # Download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Lineups (CSV)",
                data=csv,
                file_name=f"dfs_lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.info("💡 Import this CSV into your DFS platform for bulk upload")

if __name__ == "__main__":
    main()
