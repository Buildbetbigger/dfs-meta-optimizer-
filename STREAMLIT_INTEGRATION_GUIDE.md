# 🎨 Streamlit Integration Guide - Module 2

## Overview

This guide shows you how to integrate Module 2 (Advanced Lineup Generation) into your existing Streamlit app.

---

## Step 1: Update Imports

Add these imports to the top of your `app.py`:

```python
# Existing Phase 1 imports
from modules.opponent_modeling import OpponentModel
from modules.optimization_engine import LineupOptimizer
from modules.claude_assistant import ClaudeAssistant, ANTHROPIC_AVAILABLE

# NEW: Module 2 imports
from modules.advanced_optimizer import AdvancedOptimizer
from modules.stacking_engine import StackingEngine
from modules.genetic_optimizer import GeneticOptimizer
```

---

## Step 2: Initialize in Session State

Add Module 2 initialization to your session state setup:

```python
def initialize_session_state():
    """Initialize session state variables"""
    if 'players_df' not in st.session_state:
        st.session_state.players_df = None
    
    if 'opponent_model' not in st.session_state:
        st.session_state.opponent_model = None
    
    if 'generated_lineups' not in st.session_state:
        st.session_state.generated_lineups = []
    
    # NEW: Module 2 components
    if 'advanced_optimizer' not in st.session_state:
        st.session_state.advanced_optimizer = None
    
    if 'module2_lineups' not in st.session_state:
        st.session_state.module2_lineups = []

# Call this at the start of your app
initialize_session_state()
```

---

## Step 3: Create Advanced Optimization Section

Add this new section to your app (after your Phase 1 optimization):

```python
def advanced_optimization_section():
    """Module 2: Advanced lineup generation with stacking and genetic algorithm"""
    
    # Require opponent model to be initialized
    if st.session_state.opponent_model is None:
        st.warning("⚠️ Please run opponent modeling analysis first (Step 2)")
        return
    
    st.header("🚀 Step 3.5: Advanced Optimization (Module 2)")
    
    st.markdown("""
    **Enhanced lineup generation with:**
    - 🧬 Genetic algorithm optimization
    - 🔗 Automatic QB + pass-catcher stacking
    - 💎 Leverage-first strategies
    - 📊 Multi-objective fitness function
    """)
    
    # Initialize advanced optimizer if needed
    if st.session_state.advanced_optimizer is None:
        st.session_state.advanced_optimizer = AdvancedOptimizer(
            st.session_state.players_df,
            st.session_state.opponent_model
        )
        st.success("✅ Advanced optimizer initialized")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_lineups = st.number_input(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=20,
            step=1,
            help="Number of diverse lineups to generate"
        )
    
    with col2:
        optimization_mode = st.selectbox(
            "Optimization Mode",
            options=[
                'GENETIC_GPP',
                'GENETIC_CASH', 
                'GENETIC_CONTRARIAN',
                'LEVERAGE_FIRST'
            ],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Choose optimization strategy"
        )
    
    with col3:
        enforce_stacks = st.checkbox(
            "Enforce QB Stacks",
            value=True,
            help="Require QB + pass-catcher stacks"
        )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            generations = st.slider(
                "GA Generations",
                min_value=50,
                max_value=200,
                value=100,
                step=10,
                help="More generations = better results but slower"
            )
        
        with col2:
            population_size = st.slider(
                "Population Size",
                min_value=100,
                max_value=400,
                value=200,
                step=50,
                help="Larger population = more diversity"
            )
        
        max_ownership = st.number_input(
            "Max Total Ownership (optional)",
            min_value=0.0,
            max_value=400.0,
            value=0.0,
            step=10.0,
            help="Leave at 0 for no limit"
        )
    
    # Contest presets
    st.subheader("🎯 Or Use Contest Preset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        contest_type = st.selectbox(
            "Contest Type",
            options=[
                'GPP',
                'MILLY_MAKER',
                'CASH',
                'DOUBLE_UP',
                'SINGLE_ENTRY',
                'SATELLITE',
                'H2H',
                'THREE_MAX'
            ]
        )
    
    with col2:
        use_preset = st.checkbox(
            "Use Contest Preset",
            value=False,
            help="Override settings with contest-specific preset"
        )
    
    # Generate button
    st.markdown("---")
    
    if st.button("🧬 Generate Advanced Lineups", type="primary"):
        with st.spinner("Running genetic algorithm evolution..."):
            try:
                if use_preset:
                    # Use contest preset
                    lineups = st.session_state.advanced_optimizer.optimize_for_contest(
                        contest_type=contest_type,
                        num_lineups=num_lineups
                    )
                else:
                    # Use custom settings
                    max_own = max_ownership if max_ownership > 0 else None
                    
                    lineups = st.session_state.advanced_optimizer.generate_with_stacking(
                        num_lineups=num_lineups,
                        mode=optimization_mode,
                        enforce_stacks=enforce_stacks,
                        max_ownership=max_own,
                        generations=generations,
                        population_size=population_size
                    )
                
                st.session_state.module2_lineups = lineups
                
                # Show evolution stats
                stats = st.session_state.advanced_optimizer.get_evolution_stats()
                
                st.success(f"✅ Generated {len(lineups)} lineups!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial Fitness", f"{stats['initial_fitness']:.2f}")
                with col2:
                    st.metric("Final Fitness", f"{stats['final_fitness']:.2f}")
                with col3:
                    st.metric("Improvement", f"+{stats['improvement_pct']:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Error generating lineups: {str(e)}")
                return
    
    # Display lineups
    if st.session_state.module2_lineups:
        display_module2_lineups()


def display_module2_lineups():
    """Display Module 2 generated lineups"""
    
    st.markdown("---")
    st.subheader("📋 Generated Lineups")
    
    lineups = st.session_state.module2_lineups
    
    # Portfolio analysis
    with st.expander("📊 Portfolio Analysis", expanded=True):
        portfolio_stats = st.session_state.advanced_optimizer.analyze_portfolio(lineups)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Correlation",
                f"{portfolio_stats['correlation_stats']['mean']:.1f}",
                help="Average correlation score across lineups"
            )
        
        with col2:
            st.metric(
                "QB Stack %",
                f"{portfolio_stats['stacking_coverage']['qb_stack_pct']:.1f}%",
                help="Percentage of lineups with QB stacks"
            )
        
        with col3:
            st.metric(
                "Captain Diversity",
                f"{portfolio_stats['stacking_coverage']['captain_diversity']:.1f}%",
                help="Unique captains / total lineups"
            )
        
        with col4:
            st.metric(
                "Avg Leverage",
                f"{portfolio_stats['leverage_stats']['mean']:.2f}",
                help="Average leverage across lineups"
            )
        
        # Top exposures
        st.markdown("**Top Player Exposures:**")
        
        exposure_df = pd.DataFrame([
            {
                'Player': player,
                'Count': count,
                'Exposure %': f"{(count / len(lineups)) * 100:.1f}%"
            }
            for player, count in portfolio_stats['player_exposure']['top_10']
        ])
        
        st.dataframe(exposure_df, use_container_width=True, hide_index=True)
    
    # Individual lineups
    st.markdown("### 👥 Individual Lineups")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        min_correlation = st.slider(
            "Min Correlation",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            help="Filter lineups by minimum correlation"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            options=['Projection', 'Ceiling', 'Correlation', 'Leverage'],
            index=1
        )
    
    # Filter and sort
    filtered_lineups = [
        l for l in lineups
        if l['metrics']['correlation'] >= min_correlation
    ]
    
    sort_key_map = {
        'Projection': lambda x: x['metrics']['total_projection'],
        'Ceiling': lambda x: x['metrics']['total_ceiling'],
        'Correlation': lambda x: x['metrics']['correlation'],
        'Leverage': lambda x: x['metrics'].get('avg_leverage', 0)
    }
    
    filtered_lineups.sort(key=sort_key_map[sort_by], reverse=True)
    
    # Display lineups
    for i, lineup in enumerate(filtered_lineups[:50], 1):  # Show max 50
        with st.expander(
            f"Lineup {i} | Captain: {lineup['captain']} | "
            f"Proj: {lineup['metrics']['total_projection']:.1f} | "
            f"Corr: {lineup['metrics']['correlation']:.1f}"
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Players:**")
                
                lineup_df = st.session_state.players_df[
                    st.session_state.players_df['name'].isin(lineup['players'])
                ]
                
                display_df = lineup_df[['name', 'position', 'team', 'salary', 'projection', 'ceiling', 'ownership']].copy()
                display_df.columns = ['Player', 'Pos', 'Team', 'Salary', 'Proj', 'Ceil', 'Own%']
                
                # Mark captain
                display_df['Role'] = display_df['Player'].apply(
                    lambda x: 'CPT' if x == lineup['captain'] else 'FLEX'
                )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Metrics:**")
                st.metric("Projection", f"{lineup['metrics']['total_projection']:.1f}")
                st.metric("Ceiling", f"{lineup['metrics']['total_ceiling']:.1f}")
                st.metric("Ownership", f"{lineup['metrics']['total_ownership']:.1f}%")
                st.metric("Salary", f"${lineup['metrics']['total_salary']:,}")
                st.metric("Correlation", f"{lineup['metrics']['correlation']:.1f}")
                
                st.markdown("**Stacking:**")
                st.metric("QB Stacks", lineup['stacking_metrics']['qb_stacks'])
                st.metric("Game Stacks", lineup['stacking_metrics']['game_stacks'])
    
    # Export button
    st.markdown("---")
    
    if st.button("📥 Export to CSV"):
        export_data = []
        
        for i, lineup in enumerate(lineups, 1):
            players_str = ','.join(lineup['players'])
            
            export_data.append({
                'Lineup': i,
                'Captain': lineup['captain'],
                'Players': players_str,
                'Projection': lineup['metrics']['total_projection'],
                'Ceiling': lineup['metrics']['total_ceiling'],
                'Ownership': lineup['metrics']['total_ownership'],
                'Salary': lineup['metrics']['total_salary'],
                'Correlation': lineup['metrics']['correlation'],
                'QB_Stacks': lineup['stacking_metrics']['qb_stacks']
            })
        
        export_df = pd.DataFrame(export_data)
        
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="module2_lineups.csv",
            mime="text/csv"
        )
```

---

## Step 4: Add to Main App Flow

Insert the new section into your main app flow:

```python
def main():
    st.title("🏈 DFS Meta-Optimizer")
    
    # Step 1: Load data
    load_data_section()
    
    # Step 2: Opponent modeling
    if st.session_state.players_df is not None:
        opponent_modeling_section()
    
    # Step 3: Basic optimization (Phase 1)
    if st.session_state.opponent_model is not None:
        basic_optimization_section()
    
    # NEW: Step 3.5: Advanced optimization (Module 2)
    if st.session_state.opponent_model is not None:
        advanced_optimization_section()
    
    # Step 4: AI Assistant (Phase 1.5)
    if st.session_state.opponent_model is not None and ANTHROPIC_AVAILABLE:
        ai_assistant_section()


if __name__ == "__main__":
    main()
```

---

## Step 5: Test Your Integration

### Testing Checklist

1. **Start your Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Load player data** (Step 1)

3. **Run opponent modeling** (Step 2)

4. **Go to "Advanced Optimization" section** (Step 3.5)

5. **Test quick generation:**
   - Set: 5 lineups, GENETIC_GPP mode
   - Click "Generate Advanced Lineups"
   - Verify: Should complete in ~15-30 seconds

6. **Check results:**
   - ✅ Portfolio analysis shows stats
   - ✅ Lineups display correctly
   - ✅ Correlation scores present
   - ✅ QB stacks identified
   - ✅ Export to CSV works

7. **Test with contest preset:**
   - Select "MILLY_MAKER"
   - Check "Use Contest Preset"
   - Generate 20 lineups
   - Verify: Different settings applied

---

## Optional Enhancements

### 1. Add Progress Bar

```python
# In advanced_optimization_section(), after button click:

if st.button("🧬 Generate Advanced Lineups", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Initializing genetic algorithm...")
    progress_bar.progress(0.1)
    
    # ... generation code ...
    
    status_text.text("Evolution complete!")
    progress_bar.progress(1.0)
    
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
```

### 2. Add Comparison View

```python
def compare_phase1_module2():
    """Compare Phase 1 vs Module 2 lineups"""
    
    if not st.session_state.generated_lineups or not st.session_state.module2_lineups:
        st.info("Generate lineups with both Phase 1 and Module 2 to compare")
        return
    
    st.subheader("📊 Phase 1 vs Module 2 Comparison")
    
    comparison = st.session_state.advanced_optimizer.compare_to_phase1(
        st.session_state.generated_lineups,
        st.session_state.module2_lineups
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Phase 1 Correlation",
            f"{comparison['phase1']['avg_correlation']:.1f}"
        )
    
    with col2:
        st.metric(
            "Module 2 Correlation",
            f"{comparison['module2']['avg_correlation']:.1f}",
            delta=f"+{comparison['improvement']['correlation_pct']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Improvement",
            f"+{comparison['improvement']['correlation_pct']:.1f}%"
        )
```

### 3. Add Stack Visualization

```python
def show_stacking_chart():
    """Visualize stacking distribution"""
    
    if not st.session_state.module2_lineups:
        return
    
    import plotly.express as px
    
    stack_counts = {'No Stack': 0, 'QB Stack': 0, 'Game Stack': 0}
    
    for lineup in st.session_state.module2_lineups:
        metrics = lineup['stacking_metrics']
        
        if metrics['qb_stacks'] > 0:
            stack_counts['QB Stack'] += 1
        elif metrics['game_stacks'] > 0:
            stack_counts['Game Stack'] += 1
        else:
            stack_counts['No Stack'] += 1
    
    fig = px.pie(
        values=list(stack_counts.values()),
        names=list(stack_counts.keys()),
        title="Stack Distribution",
        color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

---

## Troubleshooting

### Import Error

**Error:** `ModuleNotFoundError: No module named 'modules.advanced_optimizer'`

**Fix:** Verify file placement:
```bash
ls modules/advanced_optimizer.py
```

### Session State Error

**Error:** `KeyError: 'advanced_optimizer'`

**Fix:** Make sure you called `initialize_session_state()` at app start

### Optimizer Not Initialized

**Error:** `AttributeError: 'NoneType' object has no attribute 'generate_with_stacking'`

**Fix:** Check that opponent model is initialized before creating advanced optimizer

---

## Summary

You've now integrated Module 2 into your Streamlit app! Your users can:

✅ Generate lineups with genetic algorithm
✅ Use automatic QB stacking
✅ Choose from multiple optimization modes
✅ Use contest-specific presets
✅ Analyze portfolio metrics
✅ Export lineups to CSV

**Next step:** Test with real data and consider adding Module 3 (Portfolio Optimization)!
