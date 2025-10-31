"""
DFS Lineup Generation Diagnostic Tool
Identifies why lineups aren't generating
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def diagnose_lineup_failure(
    player_pool: pd.DataFrame,
    salary_cap: int = 50000,
    positions: Dict[str, int] = None
) -> Dict:
    """
    Diagnose why lineup generation is failing.
    
    Returns detailed report of issues found.
    """
    if positions is None:
        positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1}
    
    issues = []
    warnings = []
    stats = {}
    
    # Check 1: Data exists
    if player_pool is None or len(player_pool) == 0:
        issues.append("❌ CRITICAL: No player data loaded")
        return {'issues': issues, 'warnings': warnings, 'stats': stats, 'can_generate': False}
    
    stats['total_players'] = len(player_pool)
    
    # Check 2: Required columns
    required_cols = ['name', 'position', 'salary', 'projection']
    missing_cols = [col for col in required_cols if col not in player_pool.columns]
    
    if missing_cols:
        issues.append(f"❌ CRITICAL: Missing columns: {', '.join(missing_cols)}")
        return {'issues': issues, 'warnings': warnings, 'stats': stats, 'can_generate': False}
    
    # Check 3: Projections
    stats['avg_projection'] = player_pool['projection'].mean()
    stats['max_projection'] = player_pool['projection'].max()
    stats['zero_projection_count'] = (player_pool['projection'] == 0).sum()
    
    if stats['avg_projection'] == 0:
        issues.append("❌ CRITICAL: All projections are 0 - Need to add projections")
    elif stats['avg_projection'] < 1:
        warnings.append("⚠️  WARNING: Very low projections (avg < 1)")
    
    if stats['zero_projection_count'] > len(player_pool) * 0.8:
        warnings.append(f"⚠️  WARNING: {stats['zero_projection_count']}/{len(player_pool)} players have 0 projection")
    
    # Check 4: Salaries
    stats['avg_salary'] = player_pool['salary'].mean()
    stats['total_min_salary'] = player_pool.groupby('position').apply(
        lambda x: x.nsmallest(positions.get(x.name, 0), 'salary')['salary'].sum()
        if x.name in positions else 0
    ).sum()
    
    if stats['avg_salary'] == 0:
        issues.append("❌ CRITICAL: All salaries are 0")
    elif stats['avg_salary'] < 1000:
        warnings.append("⚠️  WARNING: Very low salaries")
    
    # Check 5: Position counts
    position_counts = player_pool['position'].value_counts().to_dict()
    stats['position_counts'] = position_counts
    
    for pos, required in positions.items():
        if pos == 'FLEX':
            flex_eligible = sum([
                position_counts.get('RB', 0),
                position_counts.get('WR', 0),
                position_counts.get('TE', 0)
            ])
            if flex_eligible < required:
                issues.append(f"❌ CRITICAL: Need {required} FLEX-eligible players, have {flex_eligible}")
        else:
            available = position_counts.get(pos, 0)
            if available < required:
                issues.append(f"❌ CRITICAL: Need {required} {pos}, have only {available}")
    
    # Check 6: Salary cap feasibility
    min_salary_needed = sum([
        player_pool[player_pool['position'] == pos].nsmallest(count, 'salary')['salary'].sum()
        for pos, count in positions.items()
        if pos != 'FLEX' and pos in player_pool['position'].values
    ])
    
    if 'FLEX' in positions:
        flex_eligible = player_pool[player_pool['position'].isin(['RB', 'WR', 'TE'])]
        if not flex_eligible.empty:
            min_salary_needed += flex_eligible.nsmallest(positions['FLEX'], 'salary')['salary'].sum()
    
    stats['min_salary_needed'] = min_salary_needed
    stats['salary_cap'] = salary_cap
    
    if min_salary_needed > salary_cap:
        issues.append(f"❌ CRITICAL: Cheapest lineup costs ${min_salary_needed:,}, salary cap is ${salary_cap:,}")
    
    # Check 7: Max salary feasibility
    max_salary_possible = sum([
        player_pool[player_pool['position'] == pos].nlargest(count, 'salary')['salary'].sum()
        for pos, count in positions.items()
        if pos != 'FLEX' and pos in player_pool['position'].values
    ])
    
    if 'FLEX' in positions:
        flex_eligible = player_pool[player_pool['position'].isin(['RB', 'WR', 'TE'])]
        if not flex_eligible.empty:
            max_salary_possible += flex_eligible.nlargest(positions['FLEX'], 'salary')['salary'].sum()
    
    stats['max_salary_possible'] = max_salary_possible
    
    min_salary_threshold = salary_cap * 0.95  # Usually need to use 95%+ of cap
    if max_salary_possible < min_salary_threshold:
        warnings.append(f"⚠️  WARNING: Max possible salary ${max_salary_possible:,} < threshold ${min_salary_threshold:,}")
    
    # Check 8: Value distribution
    if 'value' in player_pool.columns:
        stats['avg_value'] = player_pool['value'].mean()
    elif player_pool['projection'].sum() > 0:
        player_pool['temp_value'] = player_pool['projection'] / player_pool['salary'].replace(0, 1) * 1000
        stats['avg_value'] = player_pool['temp_value'].mean()
    
    # Final assessment
    can_generate = len(issues) == 0
    
    return {
        'issues': issues,
        'warnings': warnings,
        'stats': stats,
        'can_generate': can_generate
    }


def print_diagnostic_report(diagnosis: Dict):
    """Print formatted diagnostic report."""
    print("\n" + "="*60)
    print(" DFS LINEUP GENERATION DIAGNOSTIC REPORT")
    print("="*60)
    
    # Summary
    if diagnosis['can_generate']:
        print("\n✅ STATUS: Can generate lineups")
    else:
        print("\n❌ STATUS: Cannot generate lineups")
    
    # Critical Issues
    if diagnosis['issues']:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in diagnosis['issues']:
            print(f"  {issue}")
    
    # Warnings
    if diagnosis['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in diagnosis['warnings']:
            print(f"  {warning}")
    
    # Stats
    if diagnosis['stats']:
        print("\n📊 STATISTICS:")
        stats = diagnosis['stats']
        
        if 'total_players' in stats:
            print(f"  Total Players: {stats['total_players']}")
        
        if 'position_counts' in stats:
            print(f"  Position Breakdown:")
            for pos, count in sorted(stats['position_counts'].items()):
                print(f"    {pos}: {count}")
        
        if 'avg_projection' in stats:
            print(f"  Projections:")
            print(f"    Average: {stats['avg_projection']:.2f}")
            print(f"    Maximum: {stats['max_projection']:.2f}")
            print(f"    Zero count: {stats['zero_projection_count']}")
        
        if 'avg_salary' in stats:
            print(f"  Salaries:")
            print(f"    Average: ${stats['avg_salary']:,.0f}")
        
        if 'salary_cap' in stats:
            print(f"  Salary Cap Analysis:")
            print(f"    Cap: ${stats['salary_cap']:,}")
            print(f"    Min needed: ${stats['min_salary_needed']:,}")
            print(f"    Max possible: ${stats['max_salary_possible']:,}")
    
    print("\n" + "="*60)


def get_quick_fixes(diagnosis: Dict) -> List[str]:
    """Get list of actionable quick fixes."""
    fixes = []
    
    issues = diagnosis.get('issues', [])
    stats = diagnosis.get('stats', {})
    
    # Check for specific issues
    for issue in issues:
        if "All projections are 0" in issue:
            fixes.append("ADD PROJECTIONS: Your CSV needs a 'projection' column with estimated points")
            fixes.append("  Option 1: Add projections manually to CSV")
            fixes.append("  Option 2: Enable Claude AI for AI-generated projections")
            fixes.append("  Option 3: Use free projections from FantasyPros or RotoGrinders")
        
        if "Missing columns" in issue:
            fixes.append("FIX COLUMNS: Run the updated fix_csv_columns function")
            fixes.append("  See FIX_PROJECTION_ERROR.md for details")
        
        if "Need" in issue and "have only" in issue:
            fixes.append(f"MORE PLAYERS: {issue}")
            fixes.append("  Upload a CSV with more players for each position")
        
        if "Cheapest lineup costs" in issue:
            fixes.append("SALARY TOO HIGH: Player salaries exceed salary cap")
            fixes.append("  Check if salary values are correct (should be 3000-10000 range)")
    
    if stats.get('zero_projection_count', 0) > 0:
        fixes.append(f"UPDATE PROJECTIONS: {stats['zero_projection_count']} players have 0 projection")
    
    return fixes


if __name__ == "__main__":
    # Test with sample data
    test_df = pd.DataFrame({
        'name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
        'position': ['QB', 'RB', 'WR', 'WR', 'TE'],
        'salary': [8000, 7000, 6000, 5500, 5000],
        'projection': [0, 0, 0, 0, 0],  # All zeros - will fail
        'team': ['KC', 'KC', 'BUF', 'BUF', 'KC']
    })
    
    diagnosis = diagnose_lineup_failure(test_df)
    print_diagnostic_report(diagnosis)
    
    if not diagnosis['can_generate']:
        print("\n🔧 QUICK FIXES:")
        fixes = get_quick_fixes(diagnosis)
        for i, fix in enumerate(fixes, 1):
            print(f"{i}. {fix}")
