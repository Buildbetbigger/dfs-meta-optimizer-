"""
DFS Meta-Optimizer - Main Application v6.2.0

NEW IN v6.2.0 UI:
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
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from optimization_engine_v6 import (
        optimize_lineups,
        CONTEST_PRESETS,
        CorrelationMatrix,
        StackAnalyzer,
        StackingReport,
        LineupFilter,
        ExposureManager
    )
    from opponent_modeling import create_opponent_model, GameInfo
    from claude_assistant import ClaudeAssistant
    from settings import get_settings, OptimizationMode
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(
    page_title="DFS Meta-Optimizer v6.2.0",
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
    
    st.title("🎯 DFS Meta-Optimizer v6.2.0")
    st.markdown("**Advanced NFL DFS Portfolio Optimization with Exposure Management**")
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
    
    # AI Ownership Prediction
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
                    file_name="dfs_lineups_v6.2.0.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == '__main__':
    main()
