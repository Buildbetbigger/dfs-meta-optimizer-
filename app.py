"""
DFS Meta-Optimizer - Main Application v6.1.0

NEW IN v6.1.0 UI:
- Contest Preset Dropdown (8 pre-configured strategies)
- Correlation Matrix Visualization
- Stacking Report Dashboard
- Bring-Back Recommendations Display
- Game Stack Opportunity Table
- Enhanced Lineup Cards with Stack Details

Web Interface built with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules
try:
    from optimization_engine import (
        optimize_lineups, 
        CONTEST_PRESETS,
        CorrelationMatrix,
        StackAnalyzer,
        StackingReport
    )
    from opponent_modeling import create_opponent_model, GameInfo
    from claude_assistant import ClaudeAssistant
    from settings import get_settings, OptimizationMode
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="DFS Meta-Optimizer v6.1.0",
    page_icon="🎯",
    layout="wide"
)


# ============================================================================
# CONTEST PRESET UI - NEW IN v6.1.0
# ============================================================================

def render_preset_selector() -> Optional[str]:
    """Render contest preset dropdown and display configuration."""
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
        help="Pre-configured optimization strategies for different contest types"
    )
    
    if selected != 'Custom':
        preset = CONTEST_PRESETS[selected]
        
        # Display preset configuration
        with st.expander("📋 Preset Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Strategy:** {preset.description}")
                st.markdown(f"**Lineups:** {preset.num_lineups}")
                st.markdown(f"**Stack Range:** {preset.stack_min}-{preset.stack_max} players")
                st.markdown(f"**Genetic Algorithm:** {'✓' if preset.use_genetic else '✗'}")
            
            with col2:
                st.markdown("**Weight Distribution:**")
                st.progress(preset.ownership_weight, text=f"Ownership: {preset.ownership_weight:.1%}")
                st.progress(preset.leverage_weight, text=f"Leverage: {preset.leverage_weight:.1%}")
                st.progress(preset.ceiling_weight, text=f"Ceiling: {preset.ceiling_weight:.1%}")
                st.progress(preset.correlation_weight, text=f"Correlation: {preset.correlation_weight:.1%}")
        
        return selected
    
    return None


# ============================================================================
# CORRELATION MATRIX UI - NEW IN v6.1.0
# ============================================================================

def render_correlation_matrix(lineup: Dict):
    """Render correlation matrix for a lineup."""
    players = lineup.get('players', [])
    
    if len(players) < 2:
        return
    
    # Get correlation matrix
    matrix = CorrelationMatrix.get_full_matrix(players)
    
    # Create DataFrame for display
    player_names = [p['name'] for p in players]
    df = pd.DataFrame(matrix, columns=player_names, index=player_names)
    
    # Color coding
    st.markdown("#### 🔗 Lineup Correlation Matrix")
    
    def color_correlation(val):
        """Color code correlation values."""
        if val > 0.4:
            return 'background-color: #90EE90'  # Light green (positive)
        elif val > 0.2:
            return 'background-color: #FFFFE0'  # Light yellow (weak positive)
        elif val < -0.2:
            return 'background-color: #FFB6C1'  # Light red (negative)
        else:
            return ''
    
    styled_df = df.style.applymap(color_correlation).format("{:.2f}")
    st.dataframe(styled_df, use_container_width=True)
    
    st.caption("🟩 Strong Positive (>0.4)  🟨 Weak Positive (0.2-0.4)  🟥 Negative (<-0.2)")


# ============================================================================
# STACKING REPORT UI - NEW IN v6.1.0
# ============================================================================

def render_stacking_report(report: Dict):
    """Render comprehensive stacking report."""
    if not report or report.get('total_stacks', 0) == 0:
        st.info("No stacks identified in generated lineups")
        return
    
    st.markdown("### 📊 Stacking Analysis Report")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Stacks",
            report['total_stacks'],
            help="Number of stacks across all lineups"
        )
    
    with col2:
        st.metric(
            "Stack Types",
            report['unique_stack_types'],
            help="Different stack configurations used"
        )
    
    with col3:
        st.metric(
            "Bring-Back %",
            f"{report['bring_back_percentage']:.1f}%",
            help="Percentage of stacks with bring-back plays"
        )
    
    with col4:
        st.metric(
            "Avg Correlation",
            f"{report['avg_correlation_score']:.1f}",
            help="Average correlation score (0-100 scale)"
        )
    
    # Stack type breakdown
    st.markdown("#### Stack Type Distribution")
    breakdown_df = pd.DataFrame([
        {'Stack Type': k, 'Count': v}
        for k, v in report['stack_type_breakdown'].items()
    ])
    st.bar_chart(breakdown_df.set_index('Stack Type'))
    
    # Top stacks
    if 'top_stacks' in report and report['top_stacks']:
        st.markdown("#### 🏆 Top 5 Stacks by Correlation")
        
        for i, stack in enumerate(report['top_stacks'][:5], 1):
            with st.expander(f"#{i}: {stack.stack_type} - {stack.team} ({stack.correlation_score:.1f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Stack Players:**")
                    for player, pos in zip(stack.players, stack.positions):
                        st.text(f"  • {player} ({pos})")
                
                with col2:
                    st.markdown("**Stack Metrics:**")
                    st.text(f"Correlation Score: {stack.correlation_score:.1f}/100")
                    st.text(f"Total Salary: ${stack.stack_salary:,}")
                    st.text(f"Avg Ownership: {stack.stack_ownership:.1f}%")
                    st.text(f"Bring-Back: {'✓' if stack.has_bring_back else '✗'}")
                    
                    if stack.bring_back_players:
                        st.markdown("**Bring-Back Players:**")
                        for player in stack.bring_back_players:
                            st.text(f"  • {player}")
    
    # Recommendations
    if 'recommendations' in report and report['recommendations']:
        st.markdown("#### 💡 Recommendations")
        for rec in report['recommendations']:
            st.info(rec)


# ============================================================================
# GAME STACK OPPORTUNITIES UI - NEW IN v6.1.0
# ============================================================================

def render_game_stack_opportunities(opportunities: List):
    """Render game stacking opportunities table."""
    if not opportunities:
        st.info("No high-scoring game stack opportunities found")
        return
    
    st.markdown("### 🎮 Game Stack Opportunities")
    st.caption("High-correlation multi-team stacks for shootout games")
    
    # Build DataFrame
    data = []
    for opp in opportunities[:10]:  # Top 10
        data.append({
            'Game': opp.game_id,
            'Total': opp.game_total,
            'Primary QB': opp.primary_qb or 'N/A',
            'Primary WRs': ', '.join(opp.primary_receivers[:2]),
            'Bring-Back Team': opp.bring_back_team,
            'Top Bring-Back': opp.bring_back_candidates[0]['name'] if opp.bring_back_candidates else 'N/A',
            'Correlation': f"{opp.correlation_score:.2f}",
            'Leverage': f"{opp.leverage_score:.2f}",
            'Own Discount': f"{opp.ownership_discount:.1%}"
        })
    
    df = pd.DataFrame(data)
    
    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Detailed view
    selected_game = st.selectbox(
        "View Detailed Stack",
        options=range(len(opportunities[:10])),
        format_func=lambda x: opportunities[x].game_id
    )
    
    if selected_game is not None:
        opp = opportunities[selected_game]
        
        with st.expander("📋 Full Stack Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Primary Stack (Offense):**")
                st.text(f"QB: {opp.primary_qb}")
                for receiver in opp.primary_receivers:
                    st.text(f"WR: {receiver}")
            
            with col2:
                st.markdown("**Bring-Back Candidates:**")
                for candidate in opp.bring_back_candidates[:5]:
                    st.text(
                        f"{candidate['name']} ({candidate['position']}) - "
                        f"${candidate['salary']:,} | {candidate['ownership']:.1f}% own"
                    )


# ============================================================================
# BRING-BACK RECOMMENDATIONS UI - NEW IN v6.1.0
# ============================================================================

def render_bring_back_recommendations(bring_backs: List[Dict]):
    """Render bring-back recommendations."""
    if not bring_backs:
        st.info("No bring-back recommendations available")
        return
    
    st.markdown("### 🔄 Bring-Back Recommendations")
    st.caption("Opposing players that correlate with your primary stack")
    
    # Build DataFrame
    df = pd.DataFrame(bring_backs)
    
    # Format columns
    display_df = df[[
        'name', 'position', 'team', 'salary', 
        'projection', 'ownership', 'correlation', 'bring_back_score'
    ]].copy()
    
    display_df.columns = [
        'Player', 'Pos', 'Team', 'Salary', 
        'Proj', 'Own%', 'Corr', 'Score'
    ]
    
    display_df['Salary'] = display_df['Salary'].apply(lambda x: f"${x:,}")
    display_df['Proj'] = display_df['Proj'].round(1)
    display_df['Own%'] = display_df['Own%'].round(1)
    display_df['Corr'] = display_df['Corr'].round(2)
    display_df['Score'] = display_df['Score'].round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# ENHANCED LINEUP DISPLAY - v6.1.0
# ============================================================================

def render_enhanced_lineup(lineup: Dict, index: int):
    """Render lineup with v6.1.0 enhancements."""
    with st.container():
        # Header with key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Projection",
                f"{lineup.get('projection', 0):.1f}",
                help="Expected fantasy points"
            )
        
        with col2:
            st.metric(
                "Salary",
                f"${lineup.get('salary', 0):,}",
                help="Total salary used"
            )
        
        with col3:
            st.metric(
                "Ownership",
                f"{lineup.get('ownership', 0):.1f}%",
                help="Average projected ownership"
            )
        
        with col4:
            # NEW v6.1.0 - Correlation score
            corr_score = lineup.get('correlation_score', 0)
            st.metric(
                "Correlation",
                f"{corr_score:.1f}",
                help="Stack correlation score (0-100)"
            )
        
        # Players table
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
        
        # NEW v6.1.0 - Stack information
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
        
        # Correlation matrix toggle
        if st.checkbox(f"Show Correlation Matrix (Lineup #{index+1})", key=f"corr_matrix_{index}"):
            render_correlation_matrix(lineup)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.title("🎯 DFS Meta-Optimizer v6.1.0")
    st.markdown("**Advanced NFL DFS Lineup Optimization with Contest Presets & Stacking Intelligence**")
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
        
        # Contest preset selector (v6.1.0)
        preset_name = render_preset_selector()
        
        # Custom configuration (if not using preset)
        if not preset_name:
            st.subheader("🎛️ Custom Settings")
            
            num_lineups = st.number_input(
                "Number of Lineups",
                min_value=1,
                max_value=150,
                value=20
            )
            
            contest_size = st.number_input(
                "Contest Size",
                min_value=2,
                max_value=500000,
                value=10000
            )
            
            optimization_method = st.selectbox(
                "Optimization Method",
                options=['genetic', 'greedy', 'monte_carlo']
            )
        else:
            preset = CONTEST_PRESETS[preset_name]
            num_lineups = preset.num_lineups
            contest_size = st.number_input(
                "Contest Size",
                min_value=2,
                max_value=500000,
                value=10000
            )
            optimization_method = 'genetic' if preset.use_genetic else 'greedy'
        
        # AI Assistant toggle
        st.subheader("🤖 AI Assistant")
        use_ai = st.checkbox(
            "Enable AI Ownership Prediction",
            value=False,
            help="Use Claude AI to predict ownership from social media analysis"
        )
        
        if use_ai:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Your Claude API key for AI predictions"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            diversity_threshold = st.slider(
                "Diversity Threshold",
                min_value=0.0,
                max_value=9.0,
                value=4.0,
                step=0.5,
                help="Minimum player differences between lineups"
            )
            
            enable_stacking_report = st.checkbox(
                "Generate Stacking Report",
                value=True,
                help="Comprehensive stack analysis"
            )
            
            show_game_stacks = st.checkbox(
                "Show Game Stack Opportunities",
                value=True,
                help="Multi-team stacking recommendations"
            )
    
    # Main content area
    if not uploaded_file:
        st.info("👈 Upload a player pool CSV to begin optimization")
        
        # Show example
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
    
    # Validate required columns
    required_cols = ['name', 'position', 'salary', 'team']
    missing_cols = [col for col in required_cols if col not in player_pool.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return
    
    # Show player pool preview
    with st.expander("👀 Player Pool Preview"):
        st.dataframe(player_pool.head(10), use_container_width=True)
    
    # AI Ownership Prediction (if enabled)
    if use_ai and 'api_key' in locals() and api_key:
        with st.spinner("🤖 AI analyzing ownership trends..."):
            try:
                assistant = ClaudeAssistant(api_key)
                
                # Get AI predictions
                predictions = assistant.predict_ownership(player_pool)
                
                if predictions:
                    player_pool['ownership'] = player_pool['name'].map(predictions)
                    st.success("✅ AI ownership predictions applied")
                    
                    # Show AI insights
                    with st.expander("🧠 AI Insights"):
                        top_chalk = sorted(
                            predictions.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                        
                        st.markdown("**Top Chalk Plays:**")
                        for name, own in top_chalk:
                            st.text(f"  • {name}: {own:.1f}%")
                        
            except Exception as e:
                st.warning(f"AI prediction failed: {e}. Using default ownership.")
    
    # Optimization button
    st.markdown("---")
    
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        
        with st.spinner(f"Optimizing {num_lineups} lineups..."):
            
            # Build config
            config = {
                'salary_cap': 50000,
                'optimization_method': optimization_method,
                'diversity_threshold': diversity_threshold
            }
            
            try:
                # Run optimization
                lineups, stacking_report = optimize_lineups(
                    player_pool,
                    num_lineups=num_lineups,
                    contest_preset=preset_name,
                    custom_config=config if not preset_name else None
                )
                
                if not lineups:
                    st.error("No valid lineups generated. Try adjusting settings.")
                    return
                
                st.success(f"✅ Generated {len(lineups)} optimal lineups!")
                
                # Display results
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
                
                # Stacking report (v6.1.0)
                if enable_stacking_report and stacking_report:
                    st.markdown("---")
                    render_stacking_report(stacking_report)
                
                # Game stack opportunities (v6.1.0)
                if show_game_stacks:
                    st.markdown("---")
                    
                    # Create opponent model to get game stacks
                    try:
                        opponent_model = create_opponent_model(
                            player_pool,
                            contest_size=contest_size
                        )
                        
                        opportunities = opponent_model.get_game_stack_opportunities(
                            min_game_total=47.0
                        )
                        
                        if opportunities:
                            render_game_stack_opportunities(opportunities)
                        
                    except Exception as e:
                        st.info("Game stack detection unavailable (need game data)")
                
                # Individual lineups
                st.markdown("---")
                st.header("📋 Generated Lineups")
                
                # Tabs for each lineup
                if len(lineups) <= 5:
                    # Show all if 5 or fewer
                    for i, lineup in enumerate(lineups):
                        st.subheader(f"Lineup #{i+1}")
                        render_enhanced_lineup(lineup, i)
                        st.markdown("---")
                else:
                    # Use tabs for many lineups
                    tabs = st.tabs([f"Lineup #{i+1}" for i in range(len(lineups))])
                    for i, (tab, lineup) in enumerate(zip(tabs, lineups)):
                        with tab:
                            render_enhanced_lineup(lineup, i)
                
                # Export functionality
                st.markdown("---")
                st.subheader("💾 Export")
                
                # Convert to CSV
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
                    file_name="dfs_lineups.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == '__main__':
    main()
