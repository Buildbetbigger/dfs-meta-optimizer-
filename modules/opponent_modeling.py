"""
Opponent Modeling Module

Analyzes field behavior and calculates leverage scores.

BUG FIXES APPLIED:
- Safe player lookups with _get_player_safe()
- Handles whitespace and case issues in names
- Never crashes on missing players
- Proper NaN handling in calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class OpponentModeler:
    """
    Models opponent behavior and calculates leverage opportunities
    
    Key Features:
    - Leverage scoring (ceiling / ownership)
    - Chalk identification
    - Field behavior prediction
    - Stack analysis
    """
    
    def __init__(self, player_data: pd.DataFrame):
        """
        Initialize opponent modeler
        
        Args:
            player_data: DataFrame with player information
        """
        self.player_data = player_data.copy()
        # Create clean name column for matching
        self.player_data['name_clean'] = self.player_data['name'].astype(str).str.strip().str.lower()
        
        print(f"✅ OpponentModeler initialized with {len(self.player_data)} players")
    
    def _get_player_safe(self, player_name: str) -> Optional[pd.Series]:
        """
        Safely retrieve player data with multiple matching strategies
        
        BUG FIX #2: Multiple fallback strategies for name matching
        
        Args:
            player_name: Name of player to lookup
            
        Returns:
            Player Series or None if not found
        """
        if pd.isna(player_name):
            return None
        
        name_clean = str(player_name).strip().lower()
        
        # Strategy 1: Exact match on clean name
        mask = self.player_data['name_clean'] == name_clean
        if mask.any():
            return self.player_data[mask].iloc[0]
        
        # Strategy 2: Partial match (contains)
        mask = self.player_data['name_clean'].str.contains(name_clean, na=False, regex=False)
        if mask.any():
            return self.player_data[mask].iloc[0]
        
        # Strategy 3: Try original name column
        mask = self.player_data['name'].astype(str).str.strip().str.lower() == name_clean
        if mask.any():
            return self.player_data[mask].iloc[0]
        
        # Not found - return None instead of crashing
        return None
    
    def calculate_leverage_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate leverage scores for all players
        
        Leverage = (Ceiling / Ownership%) * 100
        
        BUG FIX #6: Safe .get() for missing columns with defaults
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with leverage_score column added
        """
        df = df.copy()
        
        # Ensure required columns exist
        if 'ceiling' not in df.columns:
            df['ceiling'] = df['projection'] * 1.4
        else:
            df['ceiling'] = pd.to_numeric(df['ceiling'], errors='coerce').fillna(df['projection'] * 1.4)
        
        if 'ownership' not in df.columns:
            df['ownership'] = 15.0
        else:
            df['ownership'] = pd.to_numeric(df['ownership'], errors='coerce').fillna(15.0)
        
        # Calculate leverage with NaN protection
        def safe_leverage(row):
            try:
                ceiling = float(row['ceiling'])
                ownership = float(row['ownership'])
                
                # Avoid division by zero
                if ownership < 0.1:
                    ownership = 0.1
                
                leverage = (ceiling / ownership) * 100
                
                # Cap extreme values
                if np.isnan(leverage) or np.isinf(leverage):
                    return 100.0
                
                return min(leverage, 1000.0)  # Cap at 1000
                
            except (ValueError, TypeError, ZeroDivisionError):
                return 100.0
        
        df['leverage_score'] = df.apply(safe_leverage, axis=1)
        
        return df
    
    def predict_field_ownership(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict field ownership based on heuristics
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with predicted_ownership column
        """
        df = df.copy()
        
        # Calculate value (proj per $1000)
        df['value'] = df['projection'] / (df['salary'] / 1000)
        
        # Normalize factors (0-1 scale)
        salary_norm = (df['salary'].max() - df['salary']) / (df['salary'].max() - df['salary'].min() + 0.001)
        proj_norm = (df['projection'] - df['projection'].min()) / (df['projection'].max() - df['projection'].min() + 0.001)
        value_norm = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min() + 0.001)
        
        # Weighted combination
        ownership_score = (
            salary_norm * 0.3 +
            proj_norm * 0.4 +
            value_norm * 0.3
        )
        
        # Scale to 5-40% ownership range
        df['predicted_ownership'] = 5 + (ownership_score * 35)
        
        # Clean up NaN values
        df['predicted_ownership'] = df['predicted_ownership'].fillna(15.0)
        
        return df
    
    def identify_chalk_plays(self, df: pd.DataFrame, threshold: float = 25.0) -> List[str]:
        """
        Identify highly-owned "chalk" plays
        
        Args:
            df: DataFrame with ownership data
            threshold: Ownership threshold for chalk (default 25%)
            
        Returns:
            List of player names considered chalk
        """
        ownership_col = 'ownership' if 'ownership' in df.columns else 'predicted_ownership'
        
        if ownership_col not in df.columns:
            return []
        
        chalk_players = df[df[ownership_col] >= threshold]['name'].tolist()
        
        return chalk_players
    
    def identify_leverage_plays(self, df: pd.DataFrame, min_leverage: float = 150.0) -> List[str]:
        """
        Identify high-leverage plays
        
        Args:
            df: DataFrame with leverage scores
            min_leverage: Minimum leverage score threshold
            
        Returns:
            List of player names with high leverage
        """
        if 'leverage_score' not in df.columns:
            df = self.calculate_leverage_scores(df)
        
        leverage_players = df[df['leverage_score'] >= min_leverage]['name'].tolist()
        
        return leverage_players
    
    def calculate_lineup_leverage(self, lineup: pd.DataFrame) -> float:
        """
        Calculate total leverage score for a lineup
        
        Args:
            lineup: DataFrame with selected players
            
        Returns:
            Total leverage score
        """
        if 'leverage_score' not in lineup.columns:
            lineup = self.calculate_leverage_scores(lineup)
        
        return float(lineup['leverage_score'].sum())
    
    def calculate_lineup_differentiation(self, lineup: pd.DataFrame, 
                                        chalk_threshold: float = 25.0) -> Dict:
        """
        Calculate how differentiated a lineup is from the field
        
        Args:
            lineup: DataFrame with selected players
            chalk_threshold: Ownership threshold for chalk
            
        Returns:
            Dictionary with differentiation metrics
        """
        ownership_col = 'ownership' if 'ownership' in lineup.columns else 'predicted_ownership'
        
        if ownership_col not in lineup.columns:
            return {
                'avg_ownership': 15.0,
                'chalk_count': 0,
                'leverage_score': 0,
                'differentiation_score': 85.0
            }
        
        avg_ownership = float(lineup[ownership_col].mean())
        chalk_count = int(len(lineup[lineup[ownership_col] >= chalk_threshold]))
        leverage_score = self.calculate_lineup_leverage(lineup) if 'leverage_score' in lineup.columns else 0
        
        metrics = {
            'avg_ownership': avg_ownership,
            'chalk_count': chalk_count,
            'leverage_score': float(leverage_score),
            'differentiation_score': (100 - avg_ownership) + (leverage_score / 10)
        }
        
        return metrics
    
    def compare_to_field(self, lineup: pd.DataFrame) -> Dict:
        """
        Compare lineup construction to typical field approach
        
        Args:
            lineup: DataFrame with selected players
            
        Returns:
            Comparison metrics
        """
        ownership_col = 'ownership' if 'ownership' in lineup.columns else 'predicted_ownership'
        
        if ownership_col not in lineup.columns:
            return {
                'ownership_diff': 0.0,
                'salary_diff': 0.0,
                'contrarian_score': 0.0
            }
        
        lineup_own = float(lineup[ownership_col].mean())
        field_own = float(self.player_data[ownership_col].mean()) if ownership_col in self.player_data.columns else 15.0
        
        lineup_salary = float(lineup['salary'].sum())
        avg_salary = float(self.player_data['salary'].mean() * 9)  # 9 players
        
        metrics = {
            'ownership_diff': lineup_own - field_own,
            'salary_diff': lineup_salary - avg_salary,
            'contrarian_score': max(0, field_own - lineup_own)
        }
        
        return metrics
    
    def calculate_lineup_metrics(self, lineup: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive metrics for a lineup
        
        BUG FIX #2: Safe player lookups, handles missing players gracefully
        
        Args:
            lineup: DataFrame with selected players
            
        Returns:
            Dictionary of metrics
        """
        # Safely get player data for each player in lineup
        lineup_with_data = []
        for _, player in lineup.iterrows():
            player_data = self._get_player_safe(player['name'])
            if player_data is not None:
                lineup_with_data.append(player_data)
        
        if not lineup_with_data:
            # Return safe defaults if no players found
            return {
                'total_projection': 0,
                'total_salary': 0,
                'avg_ownership': 15.0,
                'total_ceiling': 0,
                'leverage_score': 0,
                'salary_utilization': 0
            }
        
        # Create DataFrame from found players
        lineup_df = pd.DataFrame(lineup_with_data)
        
        # Calculate leverage if not present
        if 'leverage_score' not in lineup_df.columns:
            lineup_df = self.calculate_leverage_scores(lineup_df)
        
        # Safe calculations with .get()
        metrics = {
            'total_projection': float(lineup_df['projection'].sum()),
            'total_salary': int(lineup_df['salary'].sum()),
            'avg_ownership': float(lineup_df.get('ownership', pd.Series([15.0])).mean()),
            'total_ceiling': float(lineup_df.get('ceiling', lineup_df['projection'] * 1.4).sum()),
            'leverage_score': float(lineup_df['leverage_score'].sum()),
            'salary_utilization': (float(lineup_df['salary'].sum()) / 50000) * 100
        }
        
        return metrics
    
    def get_leverage_opportunities(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top leverage opportunities
        
        Args:
            top_n: Number of top opportunities to return
            
        Returns:
            DataFrame with top leverage plays
        """
        df = self.player_data.copy()
        
        if 'leverage_score' not in df.columns:
            df = self.calculate_leverage_scores(df)
        
        top_leverage = df.nlargest(top_n, 'leverage_score')[[
            'name', 'position', 'team', 'salary', 'projection', 'ceiling', 
            'ownership', 'leverage_score'
        ]].copy()
        
        return top_leverage
    
    def analyze_stacking_correlation(self, lineup: pd.DataFrame) -> Dict:
        """
        Analyze QB-pass catcher correlation in lineup
        
        Args:
            lineup: DataFrame with selected players
            
        Returns:
            Stacking analysis
        """
        qb_team = lineup[lineup['position'] == 'QB']['team'].values
        
        if len(qb_team) == 0:
            return {'stacked': False, 'stack_count': 0}
        
        qb_team = qb_team[0]
        
        # Count pass catchers from same team
        pass_catchers = lineup[lineup['position'].isin(['WR', 'TE'])]
        stack_count = int(len(pass_catchers[pass_catchers['team'] == qb_team]))
        
        return {
            'stacked': stack_count > 0,
            'qb_team': str(qb_team),
            'stack_count': stack_count,
            'correlation_score': stack_count * 1.3  # Correlation multiplier
        }
