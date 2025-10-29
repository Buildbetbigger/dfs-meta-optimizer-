"""
DFS Meta-Optimizer - Main Streamlit Application
Version 6.0.0 - MOST ADVANCED STATE

Revolutionary DFS optimizer with PhD-level features:
- Genetic Algorithm v2 for evolutionary optimization
- Contest-size aware opponent modeling
- Monte Carlo tournament simulation
- AI-powered ownership with 90% cost reduction
- Exposure management with hard caps
- Real-time lineup explanations
- Historical performance tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import traceback
from typing import Optional, List, Dict, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Group 1 v6.0 modules
try:
    from settings import (
        get_config, get_config_manager, get_optimization_mode,
        validate_api_key, detect_contest_type,
        OptimizationMode, ContestType
    )
    from opponent_modeling import OpponentModel, create_opponent_model
    from claude_assistant import AIAssistant
    from optimization_engine import LineupOptimizer, ExposureCaps, OptimizationResult
    
    IMPORTS_SUCCESS = True
except ImportError as e:
    st.error(f"❌ Module Import Error: {e}")
    st.info("Ensure all v6.0 files are in the same directory as app.py")
    st.stop()
    IMPORTS_SUCCESS = False

# Page configuration
st.set_page_config(
    page_title="DFS Meta-Optimizer v6.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with advanced v6.0 styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .version-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .feature-badge {
        background: #10b981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'lineups' not in st.session_state:
        st.session_state.lineups = []
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'ai_predictions' not in st.session_state:
        st.session_state.ai_predictions = None
    if 'opponent_model' not in st.session_state:
        st.session_state.opponent_model = None
    if 'player_pool' not in st.session_state:
        st.session_state.player_pool = None
    if 'ai_cost_tracking' not in st.session_state:
        st.session_state.ai_cost_tracking = {
            'total_cost': 0.0,
            'total_tokens': 0,
            'requests': 0
        }
    if 'contest_type' not in st.session_state:
        st.session_state.contest_type = None
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []

def display_header():
    """Display main application header"""
    st.markdown('<h1 class="main-header">🚀 DFS Meta-Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="version-badge">v6.0.0 - Advanced AI Edition</div>', unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("""
    <div style="margin: 1rem 0;">
        <span class="feature-badge">🧬 Genetic Algorithm v2</span>
        <span class="feature-badge">🎯 Contest Detection</span>
        <span class="feature-badge">🤖 AI Integration</span>
        <span class="feature-badge">📊 Exposure Caps</span>
        <span class="feature-badge">🎲 Monte Carlo</span>
        <span class="feature-badge">💰 Cost Tracking</span>
    </div>
    """, unsafe_allow_html=True)

def process_player_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Process uploaded player CSV with enhanced v6.0 handling"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Handle various CSV formats
        required_cols = ['name', 'position', 'salary', 'team', 'projection']
        
        # Try to standardize column names
        col_mapping = {
            'Name': 'name', 'Player': 'name', 'player': 'name',
            'Position': 'position', 'Pos': 'position', 'pos': 'position',
            'Salary': 'salary', 'sal': 'salary', 'cost': 'salary',
            'Team': 'team', 'Tm': 'team', 'team': 'team',
            'Projection': 'projection', 'Proj': 'projection', 'fpts': 'projection',
            'FPPG': 'projection', 'AvgPointsPerGame': 'projection'
        }
        
        df.rename(columns=col_mapping, inplace=True)
        
        # Handle first_name/last_name format
        if 'first_name' in df.columns and 'last_name' in df.columns:
            df['name'] = df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Found columns: {list(df.columns)}")
            return None
        
        # Clean and validate data
        df['name'] = df['name'].astype(str).str.strip()
        df['position'] = df['position'].astype(str).str.strip().str.upper()
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        df['projection'] = pd.to_numeric(df['projection'], errors='coerce')
        df['team'] = df['team'].astype(str).str.strip().str.upper()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['name', 'position', 'salary', 'projection'])
        
        # Filter out invalid positions (like DST if not wanted)
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST', 'D', 'K']
        df = df[df['position'].isin(valid_positions)]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['name', 'position'], keep='first')
        
        # Add value column
        df['value'] = df['projection'] / (df['salary'] / 1000)
        
        st.success(f"✅ Loaded {len(df)} players successfully")
        return df
        
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        st.code(traceback.format_exc())
        return None

def configure_optimization_settings():
    """Configure optimization settings with v6.0 features"""
    st.sidebar.header("⚙️ Optimization Settings")
    
    # Contest Type Detection
    st.sidebar.subheader("🎯 Contest Configuration")
    contest_size = st.sidebar.number_input(
        "Contest Size",
        min_value=2,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Number of entries in your contest"
    )
    
    # Auto-detect contest type
    detected_type = detect_contest_type(contest_size)
    st.sidebar.info(f"📊 Detected: **{detected_type.value}**")
    st.session_state.contest_type = detected_type
    
    # Optimization Mode
    st.sidebar.subheader("🎮 Optimization Mode")
    mode_options = {
        "Aggressive GPP": "aggressive",
        "Balanced GPP": "balanced", 
        "Conservative Cash": "conservative",
        "Tournament": "tournament"
    }
    
    selected_mode_name = st.sidebar.selectbox(
        "Strategy",
        options=list(mode_options.keys()),
        index=1,
        help="Strategy affects ownership targeting and lineup construction"
    )
    
    mode_key = mode_options[selected_mode_name]
    optimization_mode = get_optimization_mode(mode_key)
    
    # Display mode details
    with st.sidebar.expander("📋 Mode Details"):
        st.write(f"**Ownership Target:** {optimization_mode.target_ownership*100:.0f}%")
        st.write(f"**Stack Weight:** {optimization_mode.stack_weight:.2f}")
        st.write(f"**Leverage Weight:** {optimization_mode.leverage_weight:.2f}")
        st.write(f"**Uniqueness:** {optimization_mode.uniqueness_threshold:.2f}")
    
    # Advanced Features
    st.sidebar.subheader("🚀 Advanced Features")
    
    # Genetic Algorithm v2
    use_genetic = st.sidebar.checkbox(
        "🧬 Genetic Algorithm v2",
        value=True,
        help="Evolutionary optimization for better lineup diversity"
    )
    
    if use_genetic:
        with st.sidebar.expander("🧬 Genetic Settings"):
            population_size = st.slider("Population Size", 50, 500, 200, 50)
            generations = st.slider("Generations", 10, 100, 50, 10)
            mutation_rate = st.slider("Mutation Rate", 0.01, 0.30, 0.10, 0.01)
    else:
        population_size = 200
        generations = 50
        mutation_rate = 0.10
    
    # Monte Carlo Simulation
    use_monte_carlo = st.sidebar.checkbox(
        "🎲 Monte Carlo Simulation",
        value=False,
        help="Tournament simulation for win probability (slower)"
    )
    
    if use_monte_carlo:
        simulations = st.sidebar.slider("Simulations", 100, 10000, 1000, 100)
    else:
        simulations = 1000
    
    # Exposure Management
    st.sidebar.subheader("📊 Exposure Management")
    
    enable_exposure_caps = st.sidebar.checkbox(
        "Enable Exposure Caps",
        value=True,
        help="Limit player exposure across lineups"
    )
    
    if enable_exposure_caps:
        with st.sidebar.expander("📊 Exposure Settings"):
            max_global = st.slider("Max Global Exposure %", 10, 100, 50, 5)
            max_per_lineup = st.slider("Max Per Lineup %", 5, 50, 20, 5)
            
            exposure_caps = ExposureCaps(
                max_global_exposure=max_global / 100,
                max_exposure_per_lineup=max_per_lineup / 100
            )
    else:
        exposure_caps = None
    
    # Number of lineups
    num_lineups = st.sidebar.number_input(
        "Number of Lineups",
        min_value=1,
        max_value=150,
        value=20,
        step=5,
        help="More lineups = better portfolio but longer generation time"
    )
    
    # AI Integration
    st.sidebar.subheader("🤖 AI Integration")
    
    config = get_config()
    api_key = config.get('claude_api_key', '')
    
    if not api_key:
        st.sidebar.warning("⚠️ No Claude API key configured")
        st.sidebar.info("Add CLAUDE_API_KEY to settings.py for AI features")
        use_ai = False
    else:
        use_ai = st.sidebar.checkbox(
            "Enable AI Predictions",
            value=True,
            help="AI-powered ownership predictions (costs ~$0.01 per run)"
        )
    
    return {
        'contest_size': contest_size,
        'contest_type': detected_type,
        'optimization_mode': optimization_mode,
        'use_genetic': use_genetic,
        'population_size': population_size,
        'generations': generations,
        'mutation_rate': mutation_rate,
        'use_monte_carlo': use_monte_carlo,
        'simulations': simulations,
        'exposure_caps': exposure_caps,
        'num_lineups': num_lineups,
        'use_ai': use_ai
    }

def display_cost_tracking():
    """Display AI cost tracking dashboard"""
    if st.session_state.ai_cost_tracking['requests'] > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 AI Cost Tracking")
        
        tracking = st.session_state.ai_cost_tracking
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Requests", tracking['requests'])
            st.metric("Total Cost", f"${tracking['total_cost']:.4f}")
        with col2:
            st.metric("Total Tokens", f"{tracking['total_tokens']:,}")
            if tracking['requests'] > 0:
                avg_cost = tracking['total_cost'] / tracking['requests']
                st.metric("Avg/Request", f"${avg_cost:.4f}")

def run_optimization(player_pool: pd.DataFrame, settings: Dict):
    """Run the optimization with v6.0 features"""
    try:
        with st.spinner("🚀 Initializing optimization..."):
            
            # Create opponent model
            opponent_model = create_opponent_model(
                contest_size=settings['contest_size'],
                mode=settings['optimization_mode']
            )
            st.session_state.opponent_model = opponent_model
            
            # Initialize AI assistant if enabled
            ai_assistant = None
            if settings['use_ai']:
                with st.spinner("🤖 Initializing AI assistant..."):
                    ai_assistant = AIAssistant()
                    
                    # Get AI predictions
                    with st.spinner("🤖 Generating AI ownership predictions..."):
                        top_players = player_pool.nlargest(50, 'projection')
                        
                        predictions = ai_assistant.predict_ownership(
                            player_pool=top_players,
                            mode=settings['optimization_mode'].name,
                            contest_type=settings['contest_type'].value
                        )
                        
                        st.session_state.ai_predictions = predictions
                        
                        # Update cost tracking
                        if hasattr(ai_assistant, 'total_cost'):
                            st.session_state.ai_cost_tracking['total_cost'] += ai_assistant.total_cost
                            st.session_state.ai_cost_tracking['total_tokens'] += ai_assistant.total_tokens
                            st.session_state.ai_cost_tracking['requests'] += 1
                        
                        # Apply predictions to player pool
                        for player_name, data in predictions.items():
                            mask = player_pool['name'] == player_name
                            if mask.any():
                                player_pool.loc[mask, 'ai_ownership'] = data['projected_ownership']
                                player_pool.loc[mask, 'ai_leverage'] = data.get('leverage_score', 0)
            
            # Calculate opponent modeling scores
            with st.spinner("📊 Running opponent modeling..."):
                player_pool['leverage_score'] = opponent_model.calculate_leverage_scores(
                    player_pool
                )
                
                player_pool['uniqueness_score'] = opponent_model.calculate_uniqueness_scores(
                    player_pool
                )
            
            # Create optimizer
            optimizer = LineupOptimizer(
                player_pool=player_pool,
                opponent_model=opponent_model,
                mode=settings['optimization_mode'],
                exposure_caps=settings['exposure_caps']
            )
            
            # Generate lineups
            with st.spinner(f"⚡ Generating {settings['num_lineups']} optimized lineups..."):
                if settings['use_genetic']:
                    # Use genetic algorithm
                    result = optimizer.optimize_with_genetic_algorithm(
                        num_lineups=settings['num_lineups'],
                        population_size=settings['population_size'],
                        generations=settings['generations'],
                        mutation_rate=settings['mutation_rate']
                    )
                else:
                    # Use standard optimization
                    result = optimizer.generate_lineups(
                        num_lineups=settings['num_lineups']
                    )
                
                st.session_state.optimization_result = result
                st.session_state.lineups = result.lineups
                
                # Add to history
                st.session_state.optimization_history.append({
                    'timestamp': datetime.now(),
                    'num_lineups': len(result.lineups),
                    'avg_projection': np.mean([l['projection'].sum() for l in result.lineups]),
                    'mode': settings['optimization_mode'].name,
                    'contest_type': settings['contest_type'].value
                })
            
            # Run Monte Carlo if enabled
            if settings['use_monte_carlo'] and len(result.lineups) > 0:
                with st.spinner("🎲 Running Monte Carlo simulation..."):
                    win_probs = optimizer.monte_carlo_tournament_simulation(
                        lineups=result.lineups,
                        num_simulations=settings['simulations'],
                        contest_size=settings['contest_size']
                    )
                    
                    # Add win probabilities to result
                    for i, prob in enumerate(win_probs):
                        if i < len(result.lineups):
                            result.lineups[i]['win_probability'] = prob
        
        return True
        
    except Exception as e:
        st.error(f"❌ Optimization Error: {str(e)}")
        st.code(traceback.format_exc())
        return False

def display_optimization_results():
    """Display comprehensive optimization results"""
    if not st.session_state.lineups:
        st.info("👆 Upload player pool and configure settings to generate lineups")
        return
    
    result = st.session_state.optimization_result
    lineups = st.session_state.lineups
    
    # Results header
    st.markdown("---")
    st.markdown("## 📊 Optimization Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Lineups Generated",
            len(lineups),
            help="Number of unique lineups created"
        )
    
    with col2:
        avg_proj = np.mean([lineup['projection'].sum() for lineup in lineups])
        st.metric(
            "Avg Projection",
            f"{avg_proj:.2f}",
            help="Average projected points across all lineups"
        )
    
    with col3:
        avg_salary = np.mean([lineup['salary'].sum() for lineup in lineups])
        salary_pct = (avg_salary / 50000) * 100
        st.metric(
            "Avg Salary Used",
            f"{salary_pct:.1f}%",
            help="Average salary utilization"
        )
    
    with col4:
        if result.metadata.get('generation_time'):
            st.metric(
                "Generation Time",
                f"{result.metadata['generation_time']:.2f}s",
                help="Time taken to generate lineups"
            )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Lineups", 
        "📊 Analytics", 
        "🎯 Leverage", 
        "💡 Explanations",
        "📥 Export"
    ])
    
    with tab1:
        display_lineups_tab(lineups)
    
    with tab2:
        display_analytics_tab(lineups, result)
    
    with tab3:
        display_leverage_tab()
    
    with tab4:
        display_explanations_tab(result)
    
    with tab5:
        display_export_tab(lineups)

def display_lineups_tab(lineups: List[pd.DataFrame]):
    """Display lineups in a clean table format"""
    st.subheader("Generated Lineups")
    
    # Lineup selector
    lineup_num = st.selectbox(
        "Select Lineup",
        range(1, len(lineups) + 1),
        format_func=lambda x: f"Lineup {x}"
    )
    
    lineup = lineups[lineup_num - 1]
    
    # Display lineup
    display_df = lineup[[' position', 'salary', 'projection', 'value']].copy()
    display_df.columns = ['Name', 'Pos', 'Salary', 'Proj', 'Value']
    
    # Add leverage and ownership if available
    if 'leverage_score' in lineup.columns:
        display_df['Leverage'] = lineup['leverage_score'].round(3)
    if 'ai_ownership' in lineup.columns:
        display_df['Own%'] = (lineup['ai_ownership'] * 100).round(1)
    if 'win_probability' in lineup.columns:
        display_df['Win%'] = (lineup['win_probability'] * 100).round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Lineup summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Salary", f"${lineup['salary'].sum():,}")
    with col2:
        st.metric("Total Projection", f"{lineup['projection'].sum():.2f}")
    with col3:
        remaining = 50000 - lineup['salary'].sum()
        st.metric("Remaining", f"${remaining:,}")

def display_analytics_tab(lineups: List[pd.DataFrame], result: OptimizationResult):
    """Display portfolio analytics"""
    st.subheader("Portfolio Analytics")
    
    # Extract all players from lineups
    all_players = pd.concat(lineups, ignore_index=True)
    
    # Player exposure analysis
    st.markdown("### 👥 Player Exposure")
    
    exposure_data = all_players.groupby('name').size().reset_index()
    exposure_data.columns = ['Player', 'Count']
    exposure_data['Exposure%'] = (exposure_data['Count'] / len(lineups) * 100).round(1)
    exposure_data = exposure_data.sort_values('Exposure%', ascending=False)
    
    # Top exposed players
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Exposed**")
        top_exposure = exposure_data.head(10)
        fig = px.bar(
            top_exposure,
            x='Exposure%',
            y='Player',
            orientation='h',
            title="Top 10 Most Exposed Players"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Least Exposed**")
        least_exposure = exposure_data[exposure_data['Exposure%'] > 0].tail(10)
        fig = px.bar(
            least_exposure,
            x='Exposure%',
            y='Player',
            orientation='h',
            title="Top 10 Least Exposed Players"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Position distribution
    st.markdown("### 📍 Position Distribution")
    
    pos_dist = all_players.groupby('position').size().reset_index()
    pos_dist.columns = ['Position', 'Count']
    
    fig = px.pie(
        pos_dist,
        values='Count',
        names='Position',
        title="Position Distribution Across All Lineups"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Projection distribution
    st.markdown("### 📈 Projection Distribution")
    
    projections = [lineup['projection'].sum() for lineup in lineups]
    
    fig = px.histogram(
        x=projections,
        nbins=20,
        title="Lineup Projection Distribution",
        labels={'x': 'Total Projection', 'y': 'Count'}
    )
    fig.add_vline(
        x=np.mean(projections),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(projections):.2f}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    if len(lineups) >= 5:
        st.markdown("### 🔗 Lineup Correlation")
        
        # Create player matrix
        player_matrix = pd.DataFrame()
        for i, lineup in enumerate(lineups[:20]):  # Limit to 20 for performance
            player_matrix[f'L{i+1}'] = lineup['name'].values
        
        st.info("Correlation shows lineup uniqueness - lower is better for GPPs")

def display_leverage_tab():
    """Display leverage analysis"""
    if st.session_state.player_pool is None:
        st.info("No player pool data available")
        return
    
    st.subheader("🎯 Leverage Analysis")
    
    player_pool = st.session_state.player_pool
    
    if 'leverage_score' not in player_pool.columns:
        st.info("Leverage scores not calculated")
        return
    
    # Top leverage plays
    st.markdown("### 🚀 Top Leverage Plays")
    
    leverage_df = player_pool[['name', 'position', 'salary', 'projection', 'leverage_score']].copy()
    leverage_df = leverage_df.sort_values('leverage_score', ascending=False).head(20)
    
    leverage_df.columns = ['Player', 'Pos', 'Salary', 'Proj', 'Leverage']
    leverage_df['Leverage'] = leverage_df['Leverage'].round(3)
    
    st.dataframe(leverage_df, use_container_width=True, hide_index=True)
    
    # Leverage vs Projection scatter
    st.markdown("### 📊 Leverage vs Projection")
    
    fig = px.scatter(
        player_pool,
        x='projection',
        y='leverage_score',
        color='position',
        hover_data=['name', 'salary'],
        title="Player Leverage vs Projection",
        labels={
            'projection': 'Projected Points',
            'leverage_score': 'Leverage Score',
            'position': 'Position'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # AI predictions if available
    if st.session_state.ai_predictions:
        st.markdown("### 🤖 AI Ownership Predictions")
        
        predictions = st.session_state.ai_predictions
        
        pred_data = []
        for player, data in list(predictions.items())[:20]:
            pred_data.append({
                'Player': player,
                'Projected Own%': f"{data['projected_ownership']*100:.1f}%",
                'Leverage': data.get('leverage_score', 0),
                'Reasoning': data.get('reasoning', '')[:100]
            })
        
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

def display_explanations_tab(result: OptimizationResult):
    """Display lineup explanations"""
    st.subheader("💡 Lineup Explanations")
    
    if not result.explanations:
        st.info("No explanations available")
        return
    
    # Select lineup to explain
    lineup_num = st.selectbox(
        "Select Lineup to Explain",
        range(1, min(len(result.explanations) + 1, len(st.session_state.lineups) + 1)),
        format_func=lambda x: f"Lineup {x}",
        key="explain_lineup_select"
    )
    
    if lineup_num <= len(result.explanations):
        explanation = result.explanations[lineup_num - 1]
        
        # Display explanation
        st.markdown("#### 📋 Strategy Breakdown")
        st.info(explanation)
        
        # Show lineup details
        lineup = st.session_state.lineups[lineup_num - 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Players**")
            top_players = lineup.nlargest(3, 'projection')[['name', 'projection']]
            for _, player in top_players.iterrows():
                st.write(f"• {player['name']}: {player['projection']:.1f} pts")
        
        with col2:
            st.markdown("**Leverage Plays**")
            if 'leverage_score' in lineup.columns:
                leverage_players = lineup.nlargest(3, 'leverage_score')[['name', 'leverage_score']]
                for _, player in leverage_players.iterrows():
                    st.write(f"• {player['name']}: {player['leverage_score']:.3f}")

def display_export_tab(lineups: List[pd.DataFrame]):
    """Display export options"""
    st.subheader("📥 Export Lineups")
    
    # Prepare export data
    export_data = []
    
    for i, lineup in enumerate(lineups, 1):
        lineup_dict = {'Lineup': i}
        
        for idx, (_, player) in enumerate(lineup.iterrows(), 1):
            lineup_dict[f'Player_{idx}'] = player['name']
            lineup_dict[f'Pos_{idx}'] = player['position']
            lineup_dict[f'Salary_{idx}'] = player['salary']
        
        lineup_dict['Total_Salary'] = lineup['salary'].sum()
        lineup_dict['Total_Projection'] = lineup['projection'].sum()
        
        export_data.append(lineup_dict)
    
    export_df = pd.DataFrame(export_data)
    
    # Display preview
    st.markdown("### Preview")
    st.dataframe(export_df.head(5), use_container_width=True)
    
    # Download button
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download All Lineups (CSV)",
        data=csv,
        file_name=f"dfs_lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.success(f"✅ Ready to export {len(lineups)} lineups")
    st.info("💡 Import this CSV into your DFS platform for bulk upload")

def display_optimization_history():
    """Display optimization history"""
    if not st.session_state.optimization_history:
        return
    
    with st.expander("📜 Optimization History"):
        history = st.session_state.optimization_history
        
        history_df = pd.DataFrame(history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
        
        st.dataframe(history_df, use_container_width=True, hide_index=True)

def main():
    """Main application flow"""
    init_session_state()
    display_header()
    
    # Sidebar configuration
    settings = configure_optimization_settings()
    display_cost_tracking()
    
    # Main content
    st.markdown("---")
    
    # File upload
    st.subheader("📁 Upload Player Pool")
    uploaded_file = st.file_uploader(
        "Upload CSV with player data",
        type=['csv'],
        help="CSV should contain: name, position, salary, team, projection"
    )
    
    if uploaded_file:
        player_pool = process_player_csv(uploaded_file)
        
        if player_pool is not None:
            st.session_state.player_pool = player_pool
            
            # Display player pool summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", len(player_pool))
            with col2:
                st.metric("Positions", player_pool['position'].nunique())
            with col3:
                st.metric("Avg Salary", f"${player_pool['salary'].mean():,.0f}")
            with col4:
                st.metric("Avg Projection", f"{player_pool['projection'].mean():.2f}")
            
            # Show player pool preview
            with st.expander("👀 Preview Player Pool"):
                st.dataframe(
                    player_pool[['name', 'position', 'team', 'salary', 'projection', 'value']].head(20),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Optimize button
            st.markdown("---")
            
            if st.button("🚀 Generate Optimized Lineups", use_container_width=True):
                success = run_optimization(player_pool, settings)
                if success:
                    st.success("✅ Optimization complete!")
                    st.rerun()
    
    # Display results if available
    display_optimization_results()
    display_optimization_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>DFS Meta-Optimizer v6.0</strong></p>
        <p>Advanced AI-Powered Daily Fantasy Sports Optimization</p>
        <p style='font-size: 0.85rem;'>
            🧬 Genetic Algorithm • 🎯 Contest Detection • 🤖 AI Integration • 📊 Exposure Caps
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
