"""
DFS Meta-Optimizer - Main Streamlit Application

This is the core engine Phase 1: Opponent Modeling + Basic Real-Time Adaptation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.opponent_modeling import OpponentModel
from modules.optimization_engine import LineupOptimizer
from modules.claude_assistant import ClaudeAssistant, ANTHROPIC_AVAILABLE
from config.settings import (
    STREAMLIT_CONFIG,
    SALARY_CAP,
    ROSTER_SIZE,
    OPTIMIZATION_MODES,
    HIGH_OWNERSHIP_THRESHOLD,
    LOW_OWNERSHIP_THRESHOLD,
    COLORS,
    ENABLE_CLAUDE_AI,
    AI_OWNERSHIP_PREDICTION,
    AI_NEWS_ANALYSIS,
    AI_STRATEGIC_ADVICE
)

# Page config
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .chalk-player {
        background-color: #FFE5E5;
        padding: 5px;
        border-radius: 5px;
    }
    .leverage-player {
        background-color: #E5F5E5;
        padding: 5px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'players_df' not in st.session_state:
    st.session_state.players_df = None
if 'opponent_model' not in st.session_state:
    st.session_state.opponent_model = None
if 'generated_lineups' not in st.session_state:
    st.session_state.generated_lineups = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'claude_assistant' not in st.session_state:
    st.session_state.claude_assistant = None
if 'ai_usage_stats' not in st.session_state:
    st.session_state.ai_usage_stats = {'requests': 0, 'estimated_cost': 0.0}


def load_sample_data():
    """Load sample player data from CSV file"""
    try:
        # Try to load from data directory
        sample_path = 'data/sample_players.csv'
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
        else:
            st.error("❌ Sample data file not found at data/sample_players.csv")
            return None
    except Exception as e:
        st.error(f"❌ Error loading sample data: {str(e)}")
        return None


def initialize_claude_assistant():
    """Initialize Claude AI assistant with proper validation"""
    if not ENABLE_CLAUDE_AI or not ANTHROPIC_AVAILABLE:
        st.warning("⚠️ Claude AI is disabled or anthropic package not available")
        return None
    
    try:
        # Step 1: Check if key exists in secrets
        if "ANTHROPIC_API_KEY" not in st.secrets:
            st.error("❌ ANTHROPIC_API_KEY not found in Streamlit secrets!")
            st.info("Add it in Settings → Secrets on Streamlit Cloud, or in .streamlit/secrets.toml locally")
            return None
        
        # Step 2: Get the key
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        
        # Step 3: Validate key format
        if not api_key or len(api_key) < 50:
            st.error(f"❌ API key too short: {len(api_key) if api_key else 0} chars (need 50+)")
            return None
        
        if not api_key.startswith('sk-ant-'):
            st.error(f"❌ API key format invalid. Should start with 'sk-ant-'")
            st.info(f"Your key starts with: {api_key[:10]}")
            return None
        
        # Step 4: Show confirmation (without revealing full key)
        with st.spinner("Initializing Claude AI..."):
            st.info(f"🔑 Using API key: {api_key[:15]}...{api_key[-10:]} ({len(api_key)} chars)")
            
            # Step 5: Create assistant - FORCE the key, never let it fall back
            assistant = ClaudeAssistant(api_key=str(api_key).strip())
            
            st.success("✅ Claude AI initialized successfully!")
            return assistant
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Claude AI")
        st.error(f"Error: {str(e)}")
        
        # Show full traceback for debugging
        import traceback
        with st.expander("Show detailed error"):
            st.code(traceback.format_exc())
        
        return None


def display_header():
    """Display application header"""
    st.title("🎯 DFS Meta-Optimizer")
    st.markdown("""
    **The Revolutionary Optimizer That Beats The Field, Not Just The Slate**
    
    Phase 1.5: Core Engine + AI-Powered Analysis
    """)
    
    # Real-time status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", "✅ Active")
    with col2:
        if st.session_state.last_update:
            st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
        else:
            st.metric("Last Update", "Never")
    with col3:
        if st.session_state.players_df is not None:
            st.metric("Players Loaded", len(st.session_state.players_df))
        else:
            st.metric("Players Loaded", 0)


def data_input_section():
    """Handle data input"""
    st.header("📊 Step 1: Load Player Data")
    
    tab1, tab2 = st.tabs(["Upload CSV", "Use Sample Data"])
    
    with tab1:
        st.markdown("**Upload your player pool CSV**")
        st.markdown("**Minimum required:** name (or first_name + last_name), team, position, salary, projection")
        st.info("💡 Missing columns (ceiling, floor, ownership) will be auto-calculated!")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Handle first_name + last_name columns
                if 'first_name' in df.columns and 'last_name' in df.columns:
                    df['name'] = df['first_name'] + ' ' + df['last_name']
                    df = df.drop(['first_name', 'last_name'], axis=1)
                    st.info("✅ Combined first_name and last_name into name")
                
                # Check minimum required columns
                required = ['name', 'team', 'position', 'salary', 'projection']
                missing = [col for col in required if col not in df.columns]
                
                if missing:
                    st.error(f"❌ Missing required columns: {', '.join(missing)}")
                    st.info(f"Available columns: {', '.join(df.columns)}")
                else:
                    # Auto-calculate missing columns
                    auto_calcs = []
                    
                    if 'ceiling' not in df.columns:
                        df['ceiling'] = df['projection'] * 1.7
                        auto_calcs.append("ceiling (projection × 1.7)")
                    
                    if 'floor' not in df.columns:
                        df['floor'] = df['projection'] * 0.4
                        auto_calcs.append("floor (projection × 0.4)")
                    
                    if 'ownership' not in df.columns:
                        df['ownership'] = 15
                        auto_calcs.append("ownership (placeholder - use AI to predict)")
                    
                    if auto_calcs:
                        st.success(f"✅ Auto-calculated: {', '.join(auto_calcs)}")
                    
                    st.session_state.players_df = df
                    st.success(f"✅ Loaded {len(df)} players successfully!")
                    st.dataframe(df)
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with tab2:
        st.markdown("**Load pre-configured sample data (KC vs BUF)**")
        
        if st.button("Load Sample Data"):
            sample_df = load_sample_data()
            if sample_df is not None:
                st.session_state.players_df = sample_df
                st.success("✅ Sample data loaded!")
                st.dataframe(st.session_state.players_df)


def opponent_modeling_section():
    """Display opponent modeling analysis"""
    if st.session_state.players_df is None:
        st.warning("⚠️ Please load player data first")
        return
    
    st.header("🧠 Step 2: Opponent Modeling Analysis")
    
    # Create opponent model
    if st.button("🔄 Analyze Field", type="primary"):
        with st.spinner("Running opponent modeling analysis..."):
            st.session_state.opponent_model = OpponentModel(st.session_state.players_df)
            st.session_state.last_update = datetime.now()
            st.success("✅ Analysis complete!")
    
    if st.session_state.opponent_model is None:
        return
    
    model = st.session_state.opponent_model
    players_df = model.get_players_dataframe()
    
    # Display field distribution
    col1, col2, col3, col4 = st.columns(4)
    
    field_dist = model.predict_field_distribution()
    
    with col1:
        st.metric("Avg Ownership", f"{field_dist['avg_ownership']:.1f}%")
    with col2:
        st.metric("Chalk Players", field_dist['chalk_count'])
    with col3:
        st.metric("Avg Leverage", f"{field_dist['avg_leverage']:.1f}")
    with col4:
        concentration = field_dist['field_concentration']
        st.metric("Field Concentration", f"{concentration:.3f}")
        if concentration > 0.15:
            st.caption("🔴 Highly chalky")
        elif concentration > 0.10:
            st.caption("🟡 Moderate chalk")
        else:
            st.caption("🟢 Well distributed")
    
    # Strategy recommendation
    strategy = model.recommend_anti_chalk_strategy()
    
    st.markdown("### 🎯 Recommended Strategy")
    st.info(f"**{strategy['strategy']}**: {strategy['recommendation']}")
    
    # Display chalk players to fade
    st.markdown("### 🚫 Top Chalk Players to Consider Fading")
    chalk_df = model.get_chalk_players()[['name', 'team', 'position', 'ownership', 'projection', 'leverage']]
    st.dataframe(chalk_df.head(5), use_container_width=True)
    
    # Display leverage plays
    st.markdown("### ⚡ High Leverage Plays")
    leverage_df = model.get_leverage_plays()[['name', 'team', 'position', 'ownership', 'projection', 'leverage', 'ceiling']]
    st.dataframe(leverage_df.head(5), use_container_width=True)
    
    # Visualization: Ownership vs Leverage
    st.markdown("### 📊 Player Positioning: Ownership vs Leverage")
    
    fig = px.scatter(
        players_df,
        x='ownership',
        y='leverage',
        size='projection',
        color='position',
        hover_data=['name', 'salary', 'ceiling'],
        labels={'ownership': 'Projected Ownership %', 'leverage': 'Leverage Score'},
        title='Player Leverage Matrix'
    )
    
    # Add threshold lines
    fig.add_hline(y=1.5, line_dash="dash", line_color="gray", 
                  annotation_text="Min Leverage Threshold")
    fig.add_vline(x=HIGH_OWNERSHIP_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text="Chalk Threshold")
    fig.add_vline(x=LOW_OWNERSHIP_THRESHOLD, line_dash="dash", line_color="green",
                  annotation_text="Contrarian Threshold")
    
    st.plotly_chart(fig, use_container_width=True)


def ai_assistant_section():
    """AI-powered analysis section"""
    if not ENABLE_CLAUDE_AI or not ANTHROPIC_AVAILABLE:
        st.info("🤖 AI Assistant features require Claude API. Set ANTHROPIC_API_KEY in Streamlit secrets or .env")
        return
    
    st.header("🤖 Step 2.5: AI-Powered Analysis (Phase 1.5)")
    
    # Initialize Claude assistant if needed
    if st.session_state.claude_assistant is None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Initialize AI Assistant", type="primary"):
                st.session_state.claude_assistant = initialize_claude_assistant()
                if st.session_state.claude_assistant:
                    st.rerun()
        with col2:
            if st.button("🔄 Clear Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session cleared! Reloading...")
                st.rerun()
        return
    
    assistant = st.session_state.claude_assistant
    
    # Add force reinitialize option
    with st.expander("⚙️ Assistant Settings"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reinitialize Assistant"):
                st.session_state.claude_assistant = None
                st.rerun()
        with col2:
            if st.button("🧪 Test API Connection"):
                try:
                    # Make a simple test call
                    test_response = assistant._call_claude("Say 'API test successful'")
                    st.success(f"✅ API Working! Response: {test_response}")
                except Exception as e:
                    st.error(f"❌ API Test Failed: {str(e)}")
    
    # Display AI usage stats
    stats = assistant.get_usage_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Requests", stats['requests'])
    with col2:
        st.metric("Est. Cost", f"${stats['estimated_cost']:.3f}")
    with col3:
        st.metric("Model", "Claude Sonnet 4")
    
    # Create tabs for different AI features
    ai_tab1, ai_tab2, ai_tab3 = st.tabs([
        "🔮 Ownership Prediction",
        "📰 News Analysis", 
        "🎯 Strategic Advice"
    ])
    
    with ai_tab1:
        st.markdown("### 🔮 AI Ownership Prediction")
        st.markdown("Let Claude analyze each player and predict ownership based on DFS psychology")
        
        if st.session_state.players_df is None:
            st.warning("⚠️ Load player data first")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Context (Optional):**")
                vegas_info = st.text_input("Vegas Lines", placeholder="KC -6.5, Total 52.5")
                news_info = st.text_area("Recent News", placeholder="Mahomes coming off 4 TD game...")
                
            with col2:
                st.markdown("**Settings:**")
                contest_type = st.selectbox("Contest Type", ["GPP", "Cash Game", "Single Entry"])
            
            if st.button("🎯 Predict All Ownership", type="primary"):
                context = {
                    'vegas': vegas_info if vegas_info else None,
                    'news': news_info if news_info else None,
                    'contest_type': contest_type
                }
                
                try:
                    with st.spinner("🤖 Claude is analyzing ownership patterns..."):
                        updated_df = assistant.batch_predict_ownership(
                            st.session_state.players_df,
                            context
                        )
                        st.session_state.players_df = updated_df
                        st.session_state.last_update = datetime.now()
                        
                        st.success("✅ AI ownership predictions updated!")
                        
                        # Show before/after comparison
                        st.markdown("#### 📊 Ownership Changes")
                        comparison_df = updated_df[['name', 'ownership', 'ai_confidence', 'ai_reasoning']].copy()
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        st.info("💡 **Tip:** Now run 'Analyze Field' to recalculate leverage with AI predictions")
                
                except Exception as e:
                    st.error(f"❌ Ownership prediction failed: {str(e)}")
                    
                    # Show detailed error
                    import traceback
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())
                    
                    # Suggest solutions
                    st.warning("💡 Try these solutions:")
                    st.markdown("""
                    1. Click "🔄 Reinitialize Assistant" in Assistant Settings above
                    2. Check that your API key is valid at https://console.anthropic.com
                    3. Clear session state and try again
                    """)
    
    with ai_tab2:
        st.markdown("### 📰 Breaking News Analysis")
        st.markdown("Paste breaking news and Claude will analyze the impact on projections and ownership")
        
        news_input = st.text_area(
            "Breaking News:",
            placeholder="Example: Travis Kelce ACTIVE, no snap count. Weather shows 15 MPH winds.",
            height=100
        )
        
        if st.button("🔍 Analyze News Impact", disabled=not news_input):
            if st.session_state.players_df is not None:
                with st.spinner("🤖 Claude is analyzing news impact..."):
                    # Pass list of player names, not the whole dataframe
                    player_names = st.session_state.players_df['name'].tolist()
                    
                    impact = assistant.analyze_news_impact(
                        news_input,
                        player_names
                    )
                    
                    st.markdown("#### 📊 Impact Analysis")
                    
                    # Display overall strategy
                    st.info(f"**Strategy:** {impact.get('overall_strategy', 'N/A')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Urgency", impact.get('urgency', 'unknown').upper())
                    with col2:
                        st.markdown(f"**Key Takeaway:** {impact.get('key_takeaway', 'N/A')}")
                    
                    # Display impacted players
                    if impact.get('impacted_players'):
                        st.markdown("#### 🎯 Impacted Players")
                        
                        for player in impact['impacted_players']:
                            impact_color = "green" if player['impact_type'] == 'positive' else "red" if player['impact_type'] == 'negative' else "gray"
                            
                            with st.expander(f"{'📈' if player['impact_type'] == 'positive' else '📉' if player['impact_type'] == 'negative' else '➡️'} {player['name']}"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Projection Δ", f"{player['projection_change']:+.1f}")
                                with col2:
                                    st.metric("Ownership Δ", f"{player['ownership_change']:+.1f}%")
                                with col3:
                                    st.metric("Leverage", player['new_leverage'].upper())
                                
                                st.markdown(f"**Reasoning:** {player['reasoning']}")
                    
                    # Leverage opportunities
                    if impact.get('leverage_opportunities'):
                        st.markdown("#### 💎 Leverage Opportunities")
                        st.success(", ".join(impact['leverage_opportunities']))
            else:
                st.warning("⚠️ Load player data first")
    
    with ai_tab3:
        st.markdown("### 🎯 AI Strategic Advice")
        st.markdown("Get sophisticated game theory analysis from Claude")
        
        if st.session_state.opponent_model is None:
            st.warning("⚠️ Run 'Analyze Field' first")
        else:
            col1, col2 = st.columns(2)
            with col1:
                contest_type_adv = st.selectbox("Contest Type", ["GPP", "Double-Up", "Single-Entry GPP"], key="contest_adv")
            with col2:
                field_size = st.number_input("Field Size", min_value=10, max_value=100000, value=10000, step=100)
            
            if st.button("🧠 Get Strategic Advice", type="primary"):
                try:
                    with st.spinner("🤖 Claude is analyzing the field..."):
                        field_dist = st.session_state.opponent_model.predict_field_distribution()
                        player_metrics = st.session_state.opponent_model.get_players_dataframe()
                        
                        contest_info = {
                            'type': contest_type_adv,
                            'entries': field_size,
                            'payout': 'Top-heavy' if contest_type_adv == 'GPP' else 'Double-up'
                        }
                        
                        advice = assistant.get_strategic_advice(
                            field_dist,
                            player_metrics,
                            contest_info
                        )
                        
                        st.markdown("#### 🎯 Strategic Recommendations")
                        
                        # Display key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Contrarian Threshold", f"{advice.get('contrarian_threshold', 50)}%")
                        with col2:
                            st.metric("Optimal Mode", advice.get('optimal_mode', 'BALANCED'))
                        with col3:
                            st.metric("Confidence", f"{advice.get('confidence', 0)}%")
                        
                        st.markdown("---")
                        
                        # Key insight
                        st.info(f"**💡 Key Insight:** {advice.get('key_insight', 'N/A')}")
                        
                        # Strategic details in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Captain Strategy:**")
                            st.write(advice.get('captain_philosophy', 'N/A'))
                            
                            st.markdown("**Correlation Advice:**")
                            st.write(advice.get('correlation_advice', 'N/A'))
                        
                        with col2:
                            if advice.get('chalk_to_fade'):
                                st.markdown("**🚫 Chalk to Fade:**")
                                for player in advice['chalk_to_fade']:
                                    st.write(f"• {player}")
                            
                            if advice.get('chalk_to_play'):
                                st.markdown("**✅ Chalk to Play:**")
                                for player in advice['chalk_to_play']:
                                    st.write(f"• {player}")
                            
                            if advice.get('leverage_targets'):
                                st.markdown("**💎 Leverage Targets:**")
                                for player in advice['leverage_targets']:
                                    st.write(f"• {player}")
                        
                        st.info("💡 Use these insights when generating lineups in the next step")
                
                except Exception as e:
                    st.error(f"❌ Strategic advice failed: {str(e)}")
                    
                    # Show detailed error
                    import traceback
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())
                    
                    # Suggest solutions
                    st.warning("💡 Try these solutions:")
                    st.markdown("""
                    1. Make sure you've run "Analyze Field" first
                    2. Check that player data includes required columns (name, ownership, projection)
                    3. Click "🔄 Reinitialize Assistant" if you see API errors
                    """)


def lineup_optimization_section():
    """Generate optimized lineups"""
    if st.session_state.opponent_model is None:
        st.warning("⚠️ Please run opponent modeling analysis first")
        return
    
    st.header("🚀 Step 3: Generate Optimized Lineups")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_lineups = st.number_input(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=20,
            step=1
        )
    
    with col2:
        mode = st.selectbox(
            "Optimization Mode",
            options=list(OPTIMIZATION_MODES.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()} - {OPTIMIZATION_MODES[x]['description']}"
        )
    
    with col3:
        max_exposure = st.slider(
            "Max Player Exposure %",
            min_value=10,
            max_value=100,
            value=70,
            step=5
        ) / 100
    
    if st.button("⚡ Generate Lineups", type="primary"):
        with st.spinner(f"Generating {num_lineups} optimized lineups..."):
            optimizer = LineupOptimizer(
                st.session_state.opponent_model.get_players_dataframe(),
                st.session_state.opponent_model
            )
            
            lineups = optimizer.generate_lineups(
                num_lineups=num_lineups,
                mode=mode,
                max_exposure=max_exposure
            )
            
            st.session_state.generated_lineups = lineups
            st.session_state.optimizer = optimizer
            st.session_state.last_update = datetime.now()
            
            st.success(f"✅ Generated {len(lineups)} lineups successfully!")


def display_lineups():
    """Display generated lineups"""
    if not st.session_state.generated_lineups:
        return
    
    st.header("📋 Generated Lineups")
    
    # Portfolio analysis
    if hasattr(st.session_state, 'optimizer'):
        portfolio = st.session_state.optimizer.get_portfolio_analysis()
        comparison = st.session_state.optimizer.compare_to_traditional()
        
        st.markdown("### 📊 Portfolio Analysis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Lineups", portfolio['total_lineups'])
        with col2:
            st.metric("Avg Projection", f"{portfolio['avg_projection']:.1f}")
        with col3:
            st.metric("Avg Ceiling", f"{portfolio['avg_ceiling']:.1f}")
        with col4:
            st.metric("Avg Ownership", f"{portfolio['avg_ownership']:.1f}%")
        with col5:
            st.metric("Avg Uniqueness", f"{portfolio['avg_uniqueness']:.1f}%")
        
        # Comparison to traditional
        st.markdown("### ⚔️ vs Traditional Optimization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Ownership Difference",
                f"{comparison['ownership_difference']:.1f}%",
                delta=f"{'Lower' if comparison['ownership_difference'] < 0 else 'Higher'} than chalk"
            )
        with col2:
            st.metric(
                "Differentiation Score",
                f"{comparison['differentiation_score']:.1f}%"
            )
        with col3:
            st.metric(
                "Uniqueness Advantage",
                f"{comparison['uniqueness_advantage']:.1f}%"
            )
    
    # Display individual lineups
    st.markdown("### 📄 Individual Lineups")
    
    for lineup in st.session_state.generated_lineups[:10]:  # Show first 10
        with st.expander(f"Lineup #{lineup['lineup_id']} - Proj: {lineup['metrics']['total_projection']:.1f} pts"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Captain:** {lineup['captain']}")
                st.markdown(f"**Flex:** {', '.join(lineup['flex'])}")
            
            with col2:
                metrics = lineup['metrics']
                st.metric("Total Salary", f"${metrics['total_salary']:,}")
                st.metric("Ceiling", f"{metrics['total_ceiling']:.1f}")
                st.metric("Ownership", f"{metrics['avg_ownership']:.1f}%")
                st.metric("Uniqueness", f"{metrics['uniqueness']:.1f}%")
    
    # Export button
    if st.button("💾 Export to CSV"):
        filename = st.session_state.optimizer.export_lineups(
            f"lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        st.success(f"✅ Exported to {filename}")
        
        # Provide download link
        try:
            with open(filename, 'r') as f:
                st.download_button(
                    label="📥 Download CSV",
                    data=f.read(),
                    file_name=filename,
                    mime='text/csv'
                )
        except:
            st.info("File saved locally. Check your project directory.")


def main():
    """Main application flow"""
    display_header()
    
    st.markdown("---")
    
    # Step 1: Data Input
    data_input_section()
    
    st.markdown("---")
    
    # Step 2: Opponent Modeling
    opponent_modeling_section()
    
    st.markdown("---")
    
    # Step 2.5: AI Assistant (Phase 1.5)
    if ENABLE_CLAUDE_AI:
        ai_assistant_section()
        st.markdown("---")
    
    # Step 3: Lineup Optimization
    lineup_optimization_section()
    
    st.markdown("---")
    
    # Step 4: Display Results
    display_lineups()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("### Contest Info")
        st.text(f"Salary Cap: ${SALARY_CAP:,}")
        st.text(f"Roster Size: {ROSTER_SIZE}")
        st.text(f"Captain Multiplier: {1.5}x")
        
        st.markdown("### Thresholds")
        st.text(f"High Ownership: >{HIGH_OWNERSHIP_THRESHOLD}%")
        st.text(f"Low Ownership: <{LOW_OWNERSHIP_THRESHOLD}%")
        
        st.markdown("---")
        
        # AI Status
        if ENABLE_CLAUDE_AI and ANTHROPIC_AVAILABLE:
            st.markdown("### 🤖 AI Assistant")
            if st.session_state.claude_assistant:
                st.success("✅ Active")
                stats = st.session_state.claude_assistant.get_usage_stats()
                st.caption(f"Requests: {stats['requests']}")
                st.caption(f"Cost: ${stats['estimated_cost']:.3f}")
            else:
                st.info("⏸️ Not initialized")
        
        st.markdown("---")
        
        st.markdown("### 📚 Quick Guide")
        st.markdown("""
        1. **Load Data** - Upload CSV (auto-calculates missing columns!)
        2. **Analyze Field** - Run opponent modeling
        3. **AI Analysis** - Get ownership predictions & advice
        4. **Generate Lineups** - Select mode & optimize
        5. **Export** - Download for DraftKings
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Optimization Modes")
        for mode_key, mode_data in OPTIMIZATION_MODES.items():
            st.markdown(f"**{mode_key.replace('_', ' ').title()}**")
            st.caption(mode_data['description'])
        
        st.markdown("---")
        
        if st.button("🔄 Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
