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
    """Load sample player data"""
    return pd.DataFrame({
        'name': ['Patrick Mahomes', 'Travis Kelce', 'Isiah Pacheco', 'Rashee Rice', 
                 'Josh Allen', 'Stefon Diggs', 'James Cook', 'Dalton Kincaid',
                 'Khalil Shakir', 'Justin Watson', 'Dawson Knox', 'Gabe Davis'],
        'team': ['KC', 'KC', 'KC', 'KC', 'BUF', 'BUF', 'BUF', 'BUF', 
                 'BUF', 'KC', 'BUF', 'BUF'],
        'position': ['QB', 'TE', 'RB', 'WR', 'QB', 'WR', 'RB', 'TE',
                     'WR', 'WR', 'TE', 'WR'],
        'salary': [11200, 9800, 8600, 7400, 11400, 9200, 8800, 6200,
                   5600, 4400, 5000, 6800],
        'projection': [24.5, 18.2, 15.8, 13.5, 25.1, 16.9, 14.7, 11.3,
                       9.8, 7.2, 8.5, 12.1],
        'ownership': [35, 28, 22, 18, 38, 24, 20, 15, 12, 8, 10, 14],
        'ceiling': [42, 32, 28, 26, 44, 30, 27, 22, 20, 16, 18, 24],
        'floor': [15, 10, 8, 6, 16, 8, 7, 5, 4, 2, 3, 5]
    })


def initialize_claude_assistant():
    """Initialize Claude AI assistant"""
    if not ENABLE_CLAUDE_AI or not ANTHROPIC_AVAILABLE:
        return None
    
    try:
        assistant = ClaudeAssistant()
        return assistant
    except Exception as e:
        st.error(f"⚠️ Could not initialize Claude AI: {str(e)}")
        st.info("Make sure ANTHROPIC_API_KEY is set in Streamlit secrets or .env file")
        return None


def display_header():
    """Display application header"""
    st.title("🎯 DFS Meta-Optimizer")
    st.markdown("""
    **The Revolutionary Optimizer That Beats The Field, Not Just The Slate**
    
    Phase 1: Core Engine - Opponent Modeling + Strategic Optimization
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
    
    tab1, tab2, tab3 = st.tabs(["Upload CSV", "Use Sample Data", "Manual Entry"])
    
    with tab1:
        st.markdown("**Upload your player pool CSV**")
        st.markdown("**Minimum required:** name, team, position, salary, projection")
        st.info("💡 Missing columns (ceiling, floor, ownership) will be auto-calculated!")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
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
            st.session_state.players_df = load_sample_data()
            st.success("✅ Sample data loaded!")
            st.dataframe(st.session_state.players_df)
    
    with tab3:
        st.markdown("**Manual player entry (coming soon)**")
        st.info("This feature will be added in the next iteration")


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
        if st.button("🚀 Initialize AI Assistant"):
            st.session_state.claude_assistant = initialize_claude_assistant()
            if st.session_state.claude_assistant:
                st.success("✅ AI Assistant ready!")
                st.rerun()
        return
    
    assistant = st.session_state.claude_assistant
    
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
                    impact = assistant.analyze_news_impact(
                        news_input,
                        st.session_state.players_df
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
                    
                    st.markdown("#### 🎓 Strategic Analysis")
                    st.markdown(advice['recommendation'])
                    
                    st.info("💡 Use these insights when generating lineups in the next step")


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
        with open(filename, 'r') as f:
            st.download_button(
                label="📥 Download CSV",
                data=f.read(),
                file_name=filename,
                mime='text/csv'
            )


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
        3. **AI Analysis** - Get ownership predictions & advice (Phase 1.5!)
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
