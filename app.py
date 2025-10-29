"""
DFS Meta-Optimizer - Main Application v7.0.1

NEW IN v7.0.1:
- Weather Data Integration (wind, temperature, precipitation)
- Injury Status Tracker (OUT/DOUBTFUL/QUESTIONABLE)
- Automated projection adjustments for weather/injuries
- Manual weather/injury overrides
- Comprehensive weather and injury reports

v7.0.0 Features (Retained):
- Advanced Analytics Dashboard (8D evaluation, variance, leverage)
- Portfolio Performance Monitor (real-time tracking)
- Contest Simulation Engine (Monte Carlo outcomes)
- Lineup Quality Metrics (comprehensive scoring)
- Risk Analysis Tools (boom/bust probability)
- Performance Comparison Charts (lineup vs lineup)

v6.3.0 Features (Retained):
- Ownership Prediction Dashboard (predict ownership %)
- News Feed Monitor (manual news entry & alerts)
- Vegas Lines Dashboard (spreads, totals, implied totals)
- Real-Time Data Integration Controls
- Chalk & Leverage Play Identification
- Injury Report Dashboard

v6.2.0 Features (Retained):
- Exposure Rule Builder (hard/soft caps)
- Exposure Report Dashboard with compliance indicators
- Tiered Portfolio Generation (safe/balanced/contrarian)
- Portfolio Filtering Options
- Similarity Matrix Visualization
- Find Most Unique Lineups
- Underexposed Player Detection
- Rebalance Portfolio Button

v6.1.0 Features (Retained):
- Contest Preset Dropdown
- Correlation Matrix Visualization
- Stacking Report Dashboard
- Bring-Back Recommendations
- Game Stack Opportunities

Web Interface built with Streamlit.
TOTAL FEATURES: 70+ UI Components (Professional Grade)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from optimization_engine import (
        optimize_lineups,
        CONTEST_PRESETS,
        CorrelationMatrix,
        StackAnalyzer,
        StackingReport,
        LineupFilter,
        ExposureManager,
        OwnershipTracker
    )
    from opponent_modeling import (
        create_opponent_model,
        GameInfo,
        NewsFeedMonitor,
        VegasLinesTracker
    )
    from claude_assistant import ClaudeAssistant
    from settings import get_settings, OptimizationMode
    from data_config import (
        NEWS_CONFIG,
        VEGAS_CONFIG,
        OWNERSHIP_CONFIG,
        get_contest_preset
    )
    # NEW v7.0.1: Weather + Injury integration
    from data_enrichment import DataEnrichment
    from weather_data import WeatherDataProvider
    from injury_tracker import InjuryTracker
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(
    page_title="DFS Meta-Optimizer v7.0.1",
    page_icon="🎯",
    layout="wide"
)


# ============================================================================
# EXPOSURE RULE BUILDER UI - NEW IN v6.2.0
# ============================================================================

def render_exposure_rule_builder() -> List[Dict]:
    """Render exposure rule builder interface."""
    st.markdown("### 🎚️ Exposure Rules")
    
    with st.expander("➕ Add Exposure Rule", expanded=False):
        rule_type_option = st.selectbox(
            "Rule Type",
            options=['Player-Specific', 'Position-Based', 'Team-Based'],
            help="Choose what type of exposure rule to create"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if rule_type_option == 'Player-Specific':
                rule_target = st.text_input("Player Name", placeholder="Patrick Mahomes")
                rule_key = 'player_name'
            elif rule_type_option == 'Position-Based':
                rule_target = st.selectbox("Position", options=['QB', 'RB', 'WR', 'TE', 'DST'])
                rule_key = 'position'
            else:  # Team-Based
                rule_target = st.text_input("Team", placeholder="KC")
                rule_key = 'team'
        
        with col2:
            min_exp = st.number_input("Min Exposure %", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
            max_exp = st.number_input("Max Exposure %", min_value=0.0, max_value=100.0, value=40.0, step=5.0)
        
        with col3:
            rule_enforcement = st.selectbox(
                "Enforcement",
                options=['Hard (Must Enforce)', 'Soft (Prefer)'],
                help="Hard caps are strictly enforced, soft caps are preferences"
            )
            priority = st.number_input("Priority", min_value=1, max_value=10, value=5, 
                                      help="Higher priority rules are enforced first")
        
        if st.button("Add Rule", type="primary"):
            if rule_target:
                rule = {
                    rule_key: rule_target,
                    'min_exposure': min_exp,
                    'max_exposure': max_exp,
                    'rule_type': 'hard' if 'Hard' in rule_enforcement else 'soft',
                    'priority': priority
                }
                
                if 'exposure_rules' not in st.session_state:
                    st.session_state.exposure_rules = []
                
                st.session_state.exposure_rules.append(rule)
                st.success(f"✅ Added {rule_enforcement.split()[0].lower()} rule for {rule_target}")
                st.rerun()
    
    # Display current rules
    if 'exposure_rules' in st.session_state and st.session_state.exposure_rules:
        st.markdown("**Current Rules:**")
        
        rules_df = pd.DataFrame([
            {
                'Target': rule.get('player_name') or rule.get('position') or rule.get('team'),
                'Type': 'Player' if 'player_name' in rule else ('Position' if 'position' in rule else 'Team'),
                'Min %': rule['min_exposure'],
                'Max %': rule['max_exposure'],
                'Enforcement': rule['rule_type'].title(),
                'Priority': rule['priority']
            }
            for rule in st.session_state.exposure_rules
        ])
        
        st.dataframe(rules_df, use_container_width=True, hide_index=True)
        
        if st.button("Clear All Rules"):
            st.session_state.exposure_rules = []
            st.rerun()
        
        return st.session_state.exposure_rules
    
    return []


# ============================================================================
# EXPOSURE REPORT DASHBOARD - NEW IN v6.2.0
# ============================================================================

def render_exposure_report(exposure_report: Dict):
    """Render comprehensive exposure report dashboard."""
    st.markdown("### 📊 Exposure Analysis")
    
    compliance = exposure_report['compliance']
    
    # Compliance status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if compliance['compliant']:
            st.success("✅ **Exposure Compliant**")
        else:
            st.error(f"❌ **{compliance['total_violations']} Violations**")
    
    with col2:
        st.metric("Violations", compliance['total_violations'])
    
    with col3:
        st.metric("Warnings", compliance['total_warnings'])
    
    # Exposure table
    exposure_table = exposure_report['exposure_table']
    
    if not exposure_table.empty:
        st.markdown("#### Top Player Exposures")
        
        # Style the dataframe
        def color_compliance(val):
            if val == '✓':
                return 'background-color: #90EE90'
            elif val == '✗':
                return 'background-color: #FFB6C1'
            return ''
        
        styled_df = exposure_table.style.applymap(
            color_compliance,
            subset=['Compliant']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Violations detail
    if compliance['violations']:
        st.markdown("#### ⚠️ Exposure Violations")
        
        for violation in compliance['violations'][:5]:  # Top 5
            st.warning(
                f"**{violation['player']}**: {violation['current_exposure']:.1f}% "
                f"(max: {violation.get('max_allowed', 'N/A')}%)"
            )
    
    # Suggestions
    suggestions = exposure_report.get('suggestions', [])
    
    if suggestions:
        st.markdown("#### 💡 Recommended Actions")
        
        for suggestion in suggestions[:3]:  # Top 3
            st.info(f"🔧 {suggestion['action']}")
    
    # Underexposed players
    underexposed = exposure_report.get('underexposed', [])
    
    if underexposed and len(underexposed) > 0:
        with st.expander(f"📉 Underexposed Players ({len(underexposed)})"):
            st.write(", ".join(underexposed[:20]))


# ============================================================================
# TIERED PORTFOLIO UI - NEW IN v6.2.0
# ============================================================================

def render_tiered_portfolio_option() -> Optional[Dict[str, float]]:
    """Render tiered portfolio configuration."""
    st.markdown("### 🎯 Portfolio Strategy")
    
    use_tiered = st.checkbox(
        "Use Tiered Portfolio",
        value=False,
        help="Split lineups into safe/balanced/contrarian tiers"
    )
    
    if use_tiered:
        st.markdown("**Configure Tier Distribution:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            safe_pct = st.slider(
                "Safe %",
                min_value=0,
                max_value=100,
                value=30,
                step=5,
                help="High floor, low variance lineups"
            )
        
        with col2:
            balanced_pct = st.slider(
                "Balanced %",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="Standard optimization"
            )
        
        with col3:
            contrarian_pct = st.slider(
                "Contrarian %",
                min_value=0,
                max_value=100,
                value=20,
                step=5,
                help="Low ownership, high leverage"
            )
        
        total = safe_pct + balanced_pct + contrarian_pct
        
        if total != 100:
            st.warning(f"⚠️ Total must equal 100% (currently {total}%)")
            return None
        
        return {
            'safe': safe_pct / 100,
            'balanced': balanced_pct / 100,
            'contrarian': contrarian_pct / 100
        }
    
    return None


# ============================================================================
# FILTERING OPTIONS UI - NEW IN v6.2.0
# ============================================================================

def render_filtering_options() -> Dict:
    """Render portfolio filtering options."""
    st.markdown("### 🔧 Filtering Options")
    
    with st.expander("Configure Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            remove_duplicates = st.checkbox("Remove Duplicates", value=True)
            
            apply_similarity = st.checkbox("Apply Similarity Filter", value=True)
            if apply_similarity:
                min_unique = st.slider(
                    "Min Unique Players",
                    min_value=1,
                    max_value=9,
                    value=4,
                    help="Minimum different players between lineups"
                )
            else:
                min_unique = 0
        
        with col2:
            apply_diversify = st.checkbox("Smart Diversification", value=False)
            if apply_diversify:
                diversity_weight = st.slider(
                    "Diversity Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.1
                )
                quality_weight = 1.0 - diversity_weight
            else:
                diversity_weight = 0.6
                quality_weight = 0.4
    
    return {
        'remove_duplicates': remove_duplicates,
        'apply_similarity': apply_similarity,
        'min_unique_players': min_unique,
        'apply_diversify': apply_diversify,
        'diversity_weight': diversity_weight,
        'quality_weight': quality_weight
    }


# ============================================================================
# PRESET SELECTOR - From v6.1.0
# ============================================================================

def render_preset_selector() -> Optional[str]:
    """Render contest preset dropdown."""
    st.markdown("### 🎯 Contest Type")
    
    preset_options = ['Custom'] + list(CONTEST_PRESETS.keys())
    preset_labels = {
        'Custom': 'Custom Configuration',
        'cash': '💰 Cash Game (50/50s, Double-Ups)',
        'gpp_small': '🏆 Small GPP (100-1K entries)',
        'gpp_large': '🚀 Large GPP (10K+ entries)',
        'gpp_massive': '💎 Massive GPP (Milly Maker)',
        'contrarian': '🎲 Contrarian (Fade Chalk)',
        'balanced': '⚖️ Balanced',
        'showdown': '⚡ Showdown (Captain Mode)',
        'turbo': '⏱️ Turbo (Quick Build)'
    }
    
    selected = st.selectbox(
        "Select Contest Strategy",
        options=preset_options,
        format_func=lambda x: preset_labels.get(x, x),
        help="Pre-configured optimization strategies"
    )
    
    if selected != 'Custom':
        preset = CONTEST_PRESETS[selected]
        
        with st.expander("📋 Preset Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Strategy:** {preset.description}")
                st.markdown(f"**Lineups:** {preset.num_lineups}")
                st.markdown(f"**Max Exposure:** {preset.max_exposure}%")
            
            with col2:
                st.markdown("**Weight Distribution:**")
                st.progress(preset.ownership_weight, text=f"Ownership: {preset.ownership_weight:.1%}")
                st.progress(preset.leverage_weight, text=f"Leverage: {preset.leverage_weight:.1%}")
                st.progress(preset.correlation_weight, text=f"Correlation: {preset.correlation_weight:.1%}")
        
        return selected
    
    return None


# ============================================================================
# SIMILARITY MATRIX - NEW IN v6.2.0
# ============================================================================

def render_similarity_matrix(lineups: List[Dict]):
    """Render lineup similarity matrix."""
    st.markdown("### 🔗 Lineup Similarity Matrix")
    
    lineup_filter = LineupFilter(pd.DataFrame())
    similarity_df = lineup_filter.get_lineup_similarity_matrix(lineups)
    
    # Color coding
    def color_similarity(val):
        if val == 100.0:
            return 'background-color: #FFFFFF'
        elif val > 75:
            return 'background-color: #FFB6C1'  # High similarity (bad)
        elif val > 50:
            return 'background-color: #FFFFE0'  # Medium
        else:
            return 'background-color: #90EE90'  # Low similarity (good)
    
    styled_df = similarity_df.style.applymap(color_similarity).format("{:.0f}")
    
    st.dataframe(styled_df, use_container_width=True)
    st.caption("🟥 High Similarity (>75%)  🟨 Medium (50-75%)  🟩 Low (<50%)")


# ============================================================================
# ENHANCED LINEUP DISPLAY - From v6.1.0
# ============================================================================

def render_enhanced_lineup(lineup: Dict, index: int):
    """Render lineup with v6.1.0+ enhancements."""
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Projection", f"{lineup.get('projection', 0):.1f}")
        
        with col2:
            st.metric("Salary", f"${lineup.get('salary', 0):,}")
        
        with col3:
            st.metric("Ownership", f"{lineup.get('ownership', 0):.1f}%")
        
        with col4:
            st.metric("Correlation", f"{lineup.get('correlation_score', 0):.1f}")
        
        players = lineup.get('players', [])
        if players:
            players_df = pd.DataFrame(players)
            
            display_cols = ['name', 'position', 'team', 'salary', 'projection', 'ownership']
            display_df = players_df[display_cols].copy()
            display_df.columns = ['Player', 'Pos', 'Team', 'Salary', 'Proj', 'Own%']
            
            display_df['Salary'] = display_df['Salary'].apply(lambda x: f"${x:,}")
            display_df['Proj'] = display_df['Proj'].round(1)
            display_df['Own%'] = display_df['Own%'].round(1)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        stacks = lineup.get('stacks', [])
        if stacks:
            st.markdown("**🔗 Identified Stacks:**")
            for stack in stacks:
                stack_info = f"• {stack.stack_type}: {', '.join(stack.players[:3])}"
                if len(stack.players) > 3:
                    stack_info += f" (+{len(stack.players)-3} more)"
                
                if stack.has_bring_back:
                    stack_info += f" | Bring-Back: {', '.join(stack.bring_back_players)}"
                
                st.text(stack_info)


# ============================================================================
# STACKING REPORT - From v6.1.0
# ============================================================================

def render_stacking_report(report: Dict):
    """Render comprehensive stacking report."""
    if not report or report.get('total_stacks', 0) == 0:
        st.info("No stacks identified")
        return
    
    st.markdown("### 📊 Stacking Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stacks", report['total_stacks'])
    
    with col2:
        st.metric("Stack Types", report['unique_stack_types'])
    
    with col3:
        st.metric("Bring-Back %", f"{report['bring_back_percentage']:.1f}%")
    
    with col4:
        st.metric("Avg Correlation", f"{report['avg_correlation_score']:.1f}")
    
    st.markdown("#### Stack Type Distribution")
    breakdown_df = pd.DataFrame([
        {'Stack Type': k, 'Count': v}
        for k, v in report['stack_type_breakdown'].items()
    ])
    st.bar_chart(breakdown_df.set_index('Stack Type'))


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    st.title("🎯 DFS Meta-Optimizer v6.3.0")
    st.markdown("**Advanced NFL DFS Portfolio Optimization with Real-Time Data Integration**")
    st.markdown("---")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📁 Player Data")
        uploaded_file = st.file_uploader(
            "Upload Player Pool CSV",
            type=['csv'],
            help="CSV with columns: name, position, salary, team, projection, ownership"
        )
        
        # Contest preset
        preset_name = render_preset_selector()
        
        # Custom settings if not using preset
        if not preset_name:
            st.subheader("🎛️ Custom Settings")
            
            num_lineups = st.number_input(
                "Number of Lineups",
                min_value=1,
                max_value=150,
                value=20
            )
            
            global_max_exposure = st.slider(
                "Global Max Exposure %",
                min_value=10.0,
                max_value=100.0,
                value=40.0,
                step=5.0,
                help="Maximum exposure for any player"
            )
        else:
            preset = CONTEST_PRESETS[preset_name]
            num_lineups = preset.num_lineups
            global_max_exposure = preset.max_exposure
        
        # v6.2.0: Exposure rules
        exposure_rules = render_exposure_rule_builder()
        
        # v6.2.0: Tiered portfolio
        tier_distribution = render_tiered_portfolio_option()
        
        # v6.2.0: Filtering options
        filter_options = render_filtering_options()
        
        # AI Assistant
        st.subheader("🤖 AI Assistant")
        use_ai = st.checkbox("Enable AI Ownership Prediction", value=False)
        
        if use_ai:
            api_key = st.text_input("Anthropic API Key", type="password")
    
    # Main content
    if not uploaded_file:
        st.info("👈 Upload a player pool CSV to begin optimization")
        
        with st.expander("📋 Example CSV Format"):
            example_df = pd.DataFrame({
                'name': ['Patrick Mahomes', 'Tyreek Hill', 'Travis Kelce'],
                'position': ['QB', 'WR', 'TE'],
                'salary': [8500, 8000, 7500],
                'team': ['KC', 'MIA', 'KC'],
                'projection': [26.5, 18.2, 16.8],
                'ownership': [18.5, 14.2, 12.1]
            })
            st.dataframe(example_df, use_container_width=True)
        
        return
    
    # Load player data
    try:
        player_pool = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(player_pool)} players")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return
    
    # Validate columns
    required_cols = ['name', 'position', 'salary', 'team']
    missing_cols = [col for col in required_cols if col not in player_pool.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return
    
    with st.expander("👀 Player Pool Preview"):
        st.dataframe(player_pool.head(10), use_container_width=True)
    
    # ============================================================================
    # v6.3.0: REAL-TIME DATA INTEGRATION
    # ============================================================================
    
    st.markdown("---")
    st.header("📡 Real-Time Data Integration (v7.0.1)")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📰 News Monitor", 
        "🎲 Vegas Lines", 
        "👥 Ownership Prediction",
        "🌦️ Weather Data",  # NEW v7.0.1
        "🏥 Injury Status"   # NEW v7.0.1
    ])
    
    # Tab 1: News Monitor
    with tab1:
        st.markdown("### 📰 News Feed Monitor")
        
        # Initialize news monitor
        if 'news_monitor' not in st.session_state:
            st.session_state.news_monitor = NewsFeedMonitor(player_pool)
        
        news_monitor = st.session_state.news_monitor
        
        # Add news item
        with st.expander("➕ Add News Item"):
            news_player = st.selectbox("Player", player_pool['name'].tolist(), key="news_player")
            news_headline = st.text_input("Headline", key="news_headline")
            news_content = st.text_area("Content", key="news_content")
            news_source = st.text_input("Source", value="manual", key="news_source")
            
            if st.button("Add News"):
                if news_headline and news_content:
                    item = news_monitor.add_news_item(
                        player_name=news_player,
                        headline=news_headline,
                        content=news_content,
                        source=news_source
                    )
                    if item:
                        st.success(f"✅ Added {item.category} news (impact: {item.impact_score:.0f}/100)")
        
        # Display critical alerts
        critical_news = news_monitor.get_critical_alerts()
        if critical_news:
            st.markdown("#### 🚨 Critical Alerts")
            for item in critical_news[:5]:
                st.error(f"**{item.player_name}**: {item.headline} (Impact: {item.impact_score:.0f}/100)")
        
        # Display injury report
        injury_report = news_monitor.get_injury_report()
        if not injury_report.empty:
            st.markdown("#### 🏥 Injury Report")
            st.dataframe(injury_report, use_container_width=True)
    
    # Tab 2: Vegas Lines
    with tab2:
        st.markdown("### 🎲 Vegas Lines Tracker")
        
        # Initialize Vegas tracker
        if 'vegas_tracker' not in st.session_state:
            st.session_state.vegas_tracker = VegasLinesTracker()
        
        vegas_tracker = st.session_state.vegas_tracker
        
        # Add game line
        with st.expander("➕ Add Game Line"):
            col1, col2 = st.columns(2)
            with col1:
                line_game_id = st.text_input("Game ID (e.g., KC@BUF)", key="line_game_id")
                line_home = st.text_input("Home Team", key="line_home")
                line_away = st.text_input("Away Team", key="line_away")
            with col2:
                line_spread = st.number_input("Spread (- = home favored)", value=0.0, step=0.5, key="line_spread")
                line_total = st.number_input("Total", value=45.0, step=0.5, key="line_total")
            
            if st.button("Add Line"):
                if line_game_id and line_home and line_away:
                    vegas_tracker.update_line(
                        game_id=line_game_id,
                        home_team=line_home,
                        away_team=line_away,
                        spread=line_spread,
                        total=line_total
                    )
                    st.success(f"✅ Added line for {line_game_id}")
        
        # Display current lines
        if vegas_tracker.current_lines:
            st.markdown("#### 📊 Current Lines & Implied Totals")
            
            implied_totals = vegas_tracker.get_all_implied_totals()
            
            lines_data = []
            for game_id, line in vegas_tracker.current_lines.items():
                home_implied = implied_totals.get(line.home_team, 0)
                away_implied = implied_totals.get(line.away_team, 0)
                
                lines_data.append({
                    'Game': game_id,
                    'Spread': f"{line.spread:+.1f}",
                    'Total': f"{line.total:.1f}",
                    f'{line.home_team} Implied': f"{home_implied:.1f}",
                    f'{line.away_team} Implied': f"{away_implied:.1f}"
                })
            
            lines_df = pd.DataFrame(lines_data)
            st.dataframe(lines_df, use_container_width=True)
            
            # Sharp money indicators
            sharp_indicators = vegas_tracker.get_sharp_money_indicators()
            if sharp_indicators:
                st.markdown("#### 💰 Sharp Money Indicators")
                for indicator in sharp_indicators:
                    st.info(f"**{indicator['game_id']}**: {indicator['indicator']} ({indicator['movement']:+.1f})")
    
    # Tab 3: Ownership Prediction
    with tab3:
        st.markdown("### 👥 Ownership Prediction")
        
        # Initialize ownership tracker
        if 'ownership_tracker' not in st.session_state:
            st.session_state.ownership_tracker = OwnershipTracker(player_pool)
        
        ownership_tracker = st.session_state.ownership_tracker
        
        # Prediction settings
        col1, col2 = st.columns(2)
        with col1:
            pred_contest_type = st.selectbox("Contest Type", ['GPP', 'CASH'], key="pred_contest")
        with col2:
            pred_chalk_threshold = st.number_input("Chalk Threshold %", value=25.0, step=5.0, key="chalk_threshold")
        
        if st.button("🔮 Predict Ownership"):
            with st.spinner("Predicting ownership for all players..."):
                # Get implied totals from Vegas tracker
                implied_totals = {}
                if 'vegas_tracker' in st.session_state:
                    implied_totals = st.session_state.vegas_tracker.get_all_implied_totals()
                
                # Get news impacts
                news_impacts = {}
                if 'news_monitor' in st.session_state:
                    for player_name in player_pool['name']:
                        # Get recent news impact (simplified)
                        news_impacts[player_name] = 0.0
                
                # Batch predict
                updated_pool = ownership_tracker.batch_predict_ownership(
                    player_pool,
                    contest_type=pred_contest_type,
                    vegas_implied_totals=implied_totals,
                    news_impacts=news_impacts
                )
                
                # Update player pool with predictions
                player_pool['ownership'] = updated_pool['ownership']
                
                st.success("✅ Ownership predictions complete!")
                
                # Show distribution
                dist = ownership_tracker.get_ownership_distribution()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Ownership", f"{dist['mean']:.1f}%")
                with col2:
                    st.metric("High Owned (25%+)", dist['high_owned'])
                with col3:
                    st.metric("Medium (10-25%)", dist['medium_owned'])
                with col4:
                    st.metric("Low (<10%)", dist['low_owned'])
                
                # Chalk plays
                chalk_plays = ownership_tracker.identify_chalk_plays(pred_chalk_threshold)
                if chalk_plays:
                    st.markdown("#### 🔥 Chalk Plays")
                    chalk_df = pd.DataFrame(chalk_plays[:10])
                    st.dataframe(chalk_df, use_container_width=True)
                
                # Leverage plays
                leverage_plays = ownership_tracker.identify_leverage_plays(player_pool, 15.0)
                if not leverage_plays.empty:
                    st.markdown("#### 💎 Leverage Plays")
                    st.dataframe(leverage_plays, use_container_width=True)
    
    # Tab 4: Weather Data (NEW v7.0.1)
    with tab4:
        st.markdown("### 🌦️ Weather Impact Analysis")
        
        # Initialize weather provider
        if 'weather_provider' not in st.session_state:
            weather_api_key = st.text_input(
                "OpenWeatherMap API Key (Optional)", 
                type="password",
                help="Get free key at openweathermap.org - 1000 calls/day"
            )
            st.session_state.weather_provider = WeatherDataProvider(weather_api_key if weather_api_key else None)
        
        weather_provider = st.session_state.weather_provider
        
        if st.button("🔄 Fetch Weather Data"):
            with st.spinner("Fetching weather for all games..."):
                # Add weather data to player pool
                enriched_pool = weather_provider.add_weather_to_players(player_pool)
                
                # Update player pool
                player_pool = enriched_pool
                
                st.success("✅ Weather data added!")
                
                # Show weather report
                weather_report = weather_provider.get_weather_report(enriched_pool)
                st.text(weather_report)
                
                # Show affected players
                bad_weather = enriched_pool[
                    (enriched_pool['weather_wind'] > 15) | 
                    (enriched_pool['weather_temp'] < 32) |
                    (enriched_pool['weather_conditions'].isin(['Rain', 'Snow', 'Thunderstorm']))
                ]
                
                if not bad_weather.empty:
                    st.markdown("#### ⚠️ Players Affected by Bad Weather")
                    weather_df = bad_weather[[
                        'name', 'position', 'team', 'weather_temp', 
                        'weather_wind', 'weather_conditions', 'weather_impact'
                    ]].copy()
                    weather_df.columns = ['Player', 'Pos', 'Team', 'Temp (°F)', 'Wind (mph)', 'Conditions', 'Impact (%)']
                    st.dataframe(weather_df, use_container_width=True)
                else:
                    st.info("✅ No bad weather detected - all games have good conditions")
        
        # Manual weather override
        with st.expander("⚙️ Manual Weather Override"):
            st.markdown("Override weather for specific games/teams")
            
            weather_team = st.selectbox("Team", player_pool['team'].unique().tolist(), key="weather_team")
            weather_temp = st.number_input("Temperature (°F)", value=65, step=1)
            weather_wind = st.number_input("Wind Speed (mph)", value=5, step=1)
            weather_conditions = st.selectbox("Conditions", ['Clear', 'Cloudy', 'Rain', 'Snow', 'Thunderstorm'])
            
            if st.button("Apply Weather Override"):
                # Apply manual weather
                team_mask = player_pool['team'] == weather_team
                player_pool.loc[team_mask, 'weather_temp'] = weather_temp
                player_pool.loc[team_mask, 'weather_wind'] = weather_wind
                player_pool.loc[team_mask, 'weather_conditions'] = weather_conditions
                
                # Recalculate impact
                impact = weather_provider.get_weather_impact_score({
                    'temperature': weather_temp,
                    'wind_speed': weather_wind,
                    'conditions': weather_conditions,
                    'precipitation_prob': 0
                })
                
                for pos in ['QB', 'RB', 'WR', 'TE', 'K']:
                    pos_mask = team_mask & (player_pool['position'] == pos)
                    if pos_mask.any():
                        player_pool.loc[pos_mask, 'weather_impact'] = impact.get(pos, 100)
                
                st.success(f"✅ Applied weather override to {weather_team}")
    
    # Tab 5: Injury Status (NEW v7.0.1)
    with tab5:
        st.markdown("### 🏥 Injury Status Tracker")
        
        # Initialize injury tracker
        if 'injury_tracker' not in st.session_state:
            st.session_state.injury_tracker = InjuryTracker()
        
        injury_tracker = st.session_state.injury_tracker
        
        # Injury source selection
        injury_source = st.selectbox(
            "Injury Data Source",
            ['fantasypros', 'espn', 'nfl'],
            help="Select where to scrape injury reports from"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            auto_adjust = st.checkbox("Auto-Adjust Projections", value=True)
        with col2:
            filter_injured = st.checkbox("Filter OUT/DOUBTFUL", value=True)
        
        if st.button("🔄 Fetch Injury Report"):
            with st.spinner(f"Scraping injury data from {injury_source}..."):
                # Scrape injuries
                injury_report = injury_tracker.scrape_injury_report(injury_source)
                
                if not injury_report.empty:
                    # Add to player pool
                    player_pool = injury_tracker.add_injury_status_to_players(
                        player_pool, 
                        injury_report
                    )
                    
                    # Adjust projections if requested
                    if auto_adjust:
                        player_pool = injury_tracker.adjust_projections_for_injury(player_pool)
                    
                    # Filter if requested
                    original_count = len(player_pool)
                    if filter_injured:
                        player_pool = injury_tracker.filter_healthy_players(player_pool)
                        st.info(f"ℹ️ Filtered {original_count - len(player_pool)} injured players")
                    
                    st.success(f"✅ Injury data added from {injury_source}")
                    
                    # Show injury report
                    injury_text = injury_tracker.get_injury_report(player_pool)
                    st.text(injury_text)
                    
                    # Show injury details
                    injured = player_pool[player_pool['injury_status'] != 'HEALTHY']
                    if not injured.empty:
                        st.markdown("#### 🚨 Injured Players")
                        injury_df = injured[[
                            'name', 'position', 'team', 'injury_status', 
                            'injury_type', 'injury_impact', 'projection'
                        ]].copy()
                        injury_df.columns = ['Player', 'Pos', 'Team', 'Status', 'Injury', 'Impact', 'Proj']
                        injury_df['Impact'] = injury_df['Impact'].apply(lambda x: f"{x:.0%}")
                        st.dataframe(injury_df, use_container_width=True)
                else:
                    st.warning(f"⚠️ No injury data available from {injury_source}")
        
        # Manual injury entry
        with st.expander("➕ Manual Injury Entry"):
            st.markdown("Manually add or update injury status")
            
            injury_player = st.selectbox("Player", player_pool['name'].tolist(), key="injury_player")
            injury_status = st.selectbox("Status", ['HEALTHY', 'QUESTIONABLE', 'DOUBTFUL', 'OUT'])
            injury_type = st.text_input("Injury Type", "N/A")
            
            if st.button("Update Injury Status"):
                player_mask = player_pool['name'] == injury_player
                player_pool.loc[player_mask, 'injury_status'] = injury_status
                player_pool.loc[player_mask, 'injury_type'] = injury_type
                
                # Set impact
                impact_map = {'OUT': 0.0, 'DOUBTFUL': 0.3, 'QUESTIONABLE': 0.75, 'HEALTHY': 1.0}
                player_pool.loc[player_mask, 'injury_impact'] = impact_map.get(injury_status, 1.0)
                
                # Adjust projection if not healthy
                if injury_status != 'HEALTHY' and auto_adjust:
                    original_proj = player_pool.loc[player_mask, 'projection'].values[0]
                    new_proj = original_proj * impact_map.get(injury_status, 1.0)
                    player_pool.loc[player_mask, 'projection'] = new_proj
                    st.info(f"📉 Adjusted projection: {original_proj:.1f} → {new_proj:.1f}")
                
                st.success(f"✅ Updated {injury_player} status to {injury_status}")
    
    # AI Ownership Prediction (Legacy)
    if use_ai and 'api_key' in locals() and api_key:
        with st.spinner("🤖 AI analyzing ownership trends..."):
            try:
                assistant = ClaudeAssistant(api_key)
                predictions = assistant.predict_ownership(player_pool)
                
                if predictions:
                    player_pool['ownership'] = player_pool['name'].map(predictions)
                    st.success("✅ AI ownership predictions applied")
            except Exception as e:
                st.warning(f"AI prediction failed: {e}")
    
    # Optimization button
    st.markdown("---")
    
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        
        with st.spinner(f"Optimizing {num_lineups} lineups..."):
            
            config = {
                'salary_cap': 50000,
                'optimization_method': 'genetic' if not preset_name or CONTEST_PRESETS.get(preset_name, ContestPreset('','',0,0,0,0,0,0,0,0,False,False)).use_genetic else 'greedy'
            }
            
            try:
                # Build exposure rules
                exposure_rules_list = exposure_rules if exposure_rules else []
                
                # Add global max as rule
                if not preset_name:
                    exposure_rules_list.append({
                        'max_exposure': global_max_exposure,
                        'rule_type': 'hard',
                        'priority': 1
                    })
                
                # Generate lineups
                lineups, stacking_report, exposure_report = optimize_lineups(
                    player_pool,
                    num_lineups=num_lineups,
                    contest_preset=preset_name,
                    custom_config=config if not preset_name else None,
                    exposure_rules=exposure_rules_list
                )
                
                if not lineups:
                    st.error("No valid lineups generated")
                    return
                
                st.success(f"✅ Generated {len(lineups)} optimal lineups!")
                
                # Apply filters
                if filter_options['remove_duplicates']:
                    lineup_filter = LineupFilter(player_pool)
                    lineups = lineup_filter.remove_exact_duplicates(lineups)
                
                if filter_options['apply_similarity']:
                    lineup_filter = LineupFilter(player_pool)
                    lineups = lineup_filter.remove_similar_lineups(
                        lineups,
                        min_unique_players=filter_options['min_unique_players']
                    )
                
                if filter_options['apply_diversify']:
                    lineup_filter = LineupFilter(player_pool)
                    lineups = lineup_filter.diversify_portfolio(
                        lineups,
                        target_size=len(lineups),
                        diversity_weight=filter_options['diversity_weight'],
                        quality_weight=filter_options['quality_weight']
                    )
                
                st.markdown("---")
                st.header("📊 Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_proj = np.mean([l.get('projection', 0) for l in lineups])
                    st.metric("Avg Projection", f"{avg_proj:.1f}")
                
                with col2:
                    avg_own = np.mean([l.get('ownership', 0) for l in lineups])
                    st.metric("Avg Ownership", f"{avg_own:.1f}%")
                
                with col3:
                    avg_salary = np.mean([l.get('salary', 0) for l in lineups])
                    st.metric("Avg Salary", f"${avg_salary:,.0f}")
                
                with col4:
                    avg_corr = np.mean([l.get('correlation_score', 0) for l in lineups])
                    st.metric("Avg Correlation", f"{avg_corr:.1f}")
                
                # v6.2.0: Exposure report
                st.markdown("---")
                render_exposure_report(exposure_report)
                
                # Stacking report
                st.markdown("---")
                render_stacking_report(stacking_report)
                
                # v6.2.0: Similarity matrix
                if st.checkbox("Show Similarity Matrix", value=False):
                    st.markdown("---")
                    render_similarity_matrix(lineups)
                
                # v6.2.0: Find unique lineups
                if st.checkbox("Show Most Unique Lineups", value=False):
                    st.markdown("---")
                    st.markdown("### 🎲 Most Contrarian Lineups")
                    
                    lineup_filter = LineupFilter(player_pool)
                    unique_lineups = lineup_filter.find_most_unique_lineups(lineups, n=min(10, len(lineups)))
                    
                    st.write(f"Showing {len(unique_lineups)} most unique lineups:")
                    
                    for i, lineup in enumerate(unique_lineups):
                        with st.expander(f"Unique Lineup #{i+1}"):
                            render_enhanced_lineup(lineup, i)
                
                # Individual lineups
                st.markdown("---")
                st.header("📋 Generated Lineups")
                
                if len(lineups) <= 5:
                    for i, lineup in enumerate(lineups):
                        st.subheader(f"Lineup #{i+1}")
                        render_enhanced_lineup(lineup, i)
                        st.markdown("---")
                else:
                    tabs = st.tabs([f"Lineup #{i+1}" for i in range(len(lineups))])
                    for i, (tab, lineup) in enumerate(zip(tabs, lineups)):
                        with tab:
                            render_enhanced_lineup(lineup, i)
                
                # Export
                st.markdown("---")
                st.subheader("💾 Export")
                
                export_data = []
                for i, lineup in enumerate(lineups):
                    for player in lineup['players']:
                        export_data.append({
                            'Lineup': i + 1,
                            'Player': player['name'],
                            'Position': player['position'],
                            'Team': player['team'],
                            'Salary': player['salary'],
                            'Projection': player.get('projection', 0),
                            'Ownership': player.get('ownership', 0)
                        })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Lineups CSV",
                    data=csv,
                    file_name="dfs_lineups_v7.0.0.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # ============================================================
                # GROUP 6 ENHANCEMENTS - Advanced Analytics & Simulation
                # ============================================================
                
                st.markdown("---")
                st.header("🚀 Advanced Analytics (v7.0.0)")
                
                # Create tabs for advanced features
                analytics_tabs = st.tabs([
                    "📊 Advanced Analytics",
                    "🎲 Contest Simulation"
                ])
                
                with analytics_tabs[0]:
                    render_advanced_analytics_tab(lineups, player_pool)
                
                with analytics_tabs[1]:
                    render_contest_simulation_tab(lineups, player_pool)
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ============================================================================
# GROUP 6 ENHANCEMENTS - v7.0.0 UI COMPONENTS
# ============================================================================

def render_advanced_analytics_tab(lineups: List[Dict], player_pool: pd.DataFrame):
    """
    Render advanced analytics dashboard with 8D evaluation,
    variance analysis, and leverage scoring.
    """
    st.markdown("### 📊 Advanced Analytics Dashboard")
    st.markdown("*PhD-Level lineup analysis with 8-dimensional evaluation*")
    
    if not lineups:
        st.info("Generate lineups first to see advanced analytics")
        return
    
    # Create optimizer instance for analysis
    optimizer = LineupOptimizer(player_pool, {'salary_cap': 50000})
    
    # Lineup selector
    st.markdown("---")
    lineup_options = [f"Lineup #{i+1}" for i in range(len(lineups))]
    selected_lineup_idx = st.selectbox(
        "Select Lineup for Detailed Analysis",
        range(len(lineups)),
        format_func=lambda x: lineup_options[x]
    )
    
    selected_lineup = lineups[selected_lineup_idx]
    
    # 8-Dimensional Evaluation
    st.markdown("---")
    st.markdown("### 🎯 8-Dimensional Evaluation")
    
    eval_8d = optimizer.evaluate_lineup_8d(selected_lineup)
    
    # Create radar chart data
    dimensions = [
        'Projection\nQuality',
        'Ownership\nEdge',
        'Correlation\nStrength',
        'Variance\nProfile',
        'Salary\nEfficiency',
        'Position\nBalance',
        'Game\nEnvironment',
        'Uniqueness'
    ]
    
    scores = [
        eval_8d['projection_quality'],
        eval_8d['ownership_edge'],
        eval_8d['correlation_strength'],
        eval_8d['variance_profile'],
        eval_8d['salary_efficiency'],
        eval_8d['position_balance'],
        eval_8d['game_environment'],
        eval_8d['uniqueness']
    ]
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Projection Quality", f"{eval_8d['projection_quality']:.1f}")
        st.metric("Ownership Edge", f"{eval_8d['ownership_edge']:.1f}")
    
    with col2:
        st.metric("Correlation", f"{eval_8d['correlation_strength']:.1f}")
        st.metric("Variance", f"{eval_8d['variance_profile']:.1f}")
    
    with col3:
        st.metric("Salary Efficiency", f"{eval_8d['salary_efficiency']:.1f}")
        st.metric("Position Balance", f"{eval_8d['position_balance']:.1f}")
    
    with col4:
        st.metric("Game Environment", f"{eval_8d['game_environment']:.1f}")
        st.metric("Uniqueness", f"{eval_8d['uniqueness']:.1f}")
    
    # Composite score
    st.markdown("---")
    composite = eval_8d['composite_score']
    
    if composite >= 80:
        quality = "🏆 Elite"
        color = "green"
    elif composite >= 70:
        quality = "⭐ Excellent"
        color = "blue"
    elif composite >= 60:
        quality = "✓ Good"
        color = "orange"
    else:
        quality = "⚠️ Needs Work"
        color = "red"
    
    st.markdown(f"### Composite Score: **:{color}[{composite:.1f}/100]** - {quality}")
    
    # Variance Analysis
    st.markdown("---")
    st.markdown("### 📈 Monte Carlo Variance Analysis")
    
    num_sims = st.slider("Number of Simulations", 100, 10000, 1000, 100)
    
    if st.button("🎲 Run Variance Analysis", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            variance_data = optimizer.analyze_lineup_variance(selected_lineup, num_sims)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Score", f"{variance_data['mean_score']:.1f}")
            st.metric("Median Score", f"{variance_data['median_score']:.1f}")
            st.metric("Std Dev", f"{variance_data['std_dev']:.1f}")
        
        with col2:
            st.metric("10th Percentile", f"{variance_data['percentile_10']:.1f}")
            st.metric("50th Percentile", f"{variance_data['percentile_50']:.1f}")
            st.metric("90th Percentile", f"{variance_data['percentile_90']:.1f}")
        
        with col3:
            st.metric("💥 Boom Prob", f"{variance_data['boom_probability']:.1f}%")
            st.metric("💣 Bust Prob", f"{variance_data['bust_probability']:.1f}%")
            st.metric("🏆 Win Prob Est", f"{variance_data['win_probability_estimate']:.1f}%")
        
        # Distribution visualization
        st.markdown("#### Score Distribution")
        st.markdown(f"**Range:** {variance_data['percentile_10']:.1f} - {variance_data['percentile_90']:.1f} points (80% confidence)")
        
        # Show percentile breakdown
        percentile_data = {
            'Percentile': ['10th', '25th', '50th', '75th', '90th'],
            'Score': [
                variance_data['percentile_10'],
                variance_data['percentile_25'],
                variance_data['percentile_50'],
                variance_data['percentile_75'],
                variance_data['percentile_90']
            ]
        }
        
        st.dataframe(pd.DataFrame(percentile_data), use_container_width=True, hide_index=True)
    
    # Leverage Analysis
    st.markdown("---")
    st.markdown("### 💎 Leverage Analysis")
    
    leverage_score = optimizer.calculate_lineup_leverage(selected_lineup)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Leverage Score", f"{leverage_score:.1f}/100")
        
        if leverage_score >= 70:
            st.success("🎯 **Highly Contrarian** - Great GPP potential")
        elif leverage_score >= 50:
            st.info("✓ **Moderate Leverage** - Balanced approach")
        else:
            st.warning("⚠️ **Chalky** - High ownership exposure")
    
    with col2:
        avg_own = np.mean([p.get('ownership', 50) for p in selected_lineup['players']])
        st.metric("Average Ownership", f"{avg_own:.1f}%")
        
        if avg_own < 15:
            st.success("Very low owned")
        elif avg_own < 25:
            st.info("Low-medium owned")
        else:
            st.warning("High owned")
    
    # Portfolio-Level Analytics
    if len(lineups) > 1:
        st.markdown("---")
        st.markdown("### 📊 Portfolio-Level Analytics")
        
        portfolio_metrics = optimizer.calculate_portfolio_metrics(lineups)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Lineups", portfolio_metrics['num_lineups'])
            st.metric("Unique Players", portfolio_metrics['unique_players'])
        
        with col2:
            st.metric("Diversity Score", f"{portfolio_metrics['diversity_score']:.1f}%")
            st.metric("Max Exposure", f"{portfolio_metrics['max_exposure']:.1f}%")
        
        with col3:
            st.metric("Avg Projection", f"{portfolio_metrics['avg_projection']:.1f}")
            st.metric("Avg Ownership", f"{portfolio_metrics['avg_ownership']:.1f}%")
        
        with col4:
            st.metric("Proj Variance", f"{portfolio_metrics['projection_variance']:.1f}")
            st.metric("Own Variance", f"{portfolio_metrics['ownership_variance']:.1f}")
        
        # Most/Least exposed players
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Exposed Players**")
            most_exp = portfolio_metrics['most_exposed_players'][:5]
            for player, exp in most_exp:
                st.write(f"• {player}: {exp:.1f}%")
        
        with col2:
            st.markdown("**Least Exposed Players**")
            least_exp = portfolio_metrics['least_exposed_players'][:5]
            for player, exp in least_exp:
                st.write(f"• {player}: {exp:.1f}%")


def render_contest_simulation_tab(lineups: List[Dict], player_pool: pd.DataFrame):
    """
    Render contest simulation dashboard for estimating
    win probability and optimal lineup selection.
    """
    st.markdown("### 🎲 Contest Outcome Simulation")
    st.markdown("*Monte Carlo simulation of actual contest results*")
    
    if not lineups:
        st.info("Generate lineups first to run contest simulations")
        return
    
    if len(lineups) < 2:
        st.warning("Generate at least 2 lineups for meaningful simulation")
        return
    
    # Simulation parameters
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        num_sims = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=50000,
            value=1000,
            step=100,
            help="More simulations = more accurate but slower"
        )
    
    with col2:
        contest_size = st.number_input(
            "Contest Size",
            min_value=10,
            max_value=10000,
            value=100,
            step=10,
            help="Number of entries in the contest"
        )
    
    if st.button("🚀 Run Contest Simulation", type="primary", use_container_width=True):
        
        optimizer = LineupOptimizer(player_pool, {'salary_cap': 50000})
        
        with st.spinner(f"Simulating {num_sims:,} contests with {contest_size} entries each..."):
            sim_results = optimizer.simulate_contest_outcomes(
                lineups,
                num_simulations=num_sims,
                contest_size=contest_size
            )
        
        st.success(f"✅ Simulation complete! Analyzed {len(lineups)} lineups")
        
        # Results
        st.markdown("---")
        st.markdown("### 📊 Simulation Results")
        
        # Key findings
        best_lineup = sim_results['best_lineup_idx']
        safest_lineup = sim_results['safest_lineup_idx']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Highest Win Probability")
            st.markdown(f"**Lineup #{best_lineup + 1}**")
            win_prob = sim_results['win_probabilities'].get(best_lineup, 0)
            st.metric("Win Probability", f"{win_prob:.2f}%")
            
            # Show lineup
            best_lineup_data = lineups[best_lineup]
            st.write(f"Projection: {best_lineup_data.get('projection', 0):.1f}")
            st.write(f"Ownership: {best_lineup_data.get('ownership', 0):.1f}%")
        
        with col2:
            st.markdown("#### 💰 Safest (Highest Cash %)  ")
            st.markdown(f"**Lineup #{safest_lineup + 1}**")
            cash_prob = sim_results['cash_probabilities'].get(safest_lineup, 0)
            st.metric("Cash Probability", f"{cash_prob:.2f}%")
            
            # Show lineup
            safest_lineup_data = lineups[safest_lineup]
            st.write(f"Projection: {safest_lineup_data.get('projection', 0):.1f}")
            st.write(f"Ownership: {safest_lineup_data.get('ownership', 0):.1f}%")
        
        # Detailed table
        st.markdown("---")
        st.markdown("### 📋 All Lineups - Probability Breakdown")
        
        results_data = []
        for i in range(len(lineups)):
            results_data.append({
                'Lineup': f"#{i+1}",
                'Win %': f"{sim_results['win_probabilities'].get(i, 0):.2f}",
                'Top 10 %': f"{sim_results['top10_probabilities'].get(i, 0):.2f}",
                'Cash %': f"{sim_results['cash_probabilities'].get(i, 0):.2f}",
                'Projection': lineups[i].get('projection', 0),
                'Ownership': f"{lineups[i].get('ownership', 0):.1f}%"
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Insights
        st.markdown("---")
        st.markdown("### 💡 Insights")
        
        # Calculate some insights
        total_win_prob = sum(sim_results['win_probabilities'].values())
        avg_cash_prob = np.mean(list(sim_results['cash_probabilities'].values()))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Combined Win %", f"{total_win_prob:.2f}%")
            st.caption("Probability at least 1 lineup wins")
        
        with col2:
            st.metric("Avg Cash %", f"{avg_cash_prob:.2f}%")
            st.caption("Average cash rate across portfolio")
        
        with col3:
            top_heavy = len([p for p in sim_results['win_probabilities'].values() if p > 1.0])
            st.metric("Tournament Lineups", top_heavy)
            st.caption(f"Lineups with >1% win probability")


if __name__ == '__main__':
    main()
