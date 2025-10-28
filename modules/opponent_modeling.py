"""
Opponent Modeling Module

Analyzes field behavior and calculates leverage scores for DFS optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class OpponentModeler:
    """
    Models opponent behavior and calculates leverage opportunities
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
        
    def _get_player_safe(self, player_name: str) -> Optional[pd.Series]:
        """
        Safely retrieve player data
        
        Args:
            player_name: Name of player to lookup
            
        Returns:
            Player Series or None if not found
        """
        if pd.isna(player_name):
            return None
        
        name_clean = str(player_name).strip().lower()
        
        # Try exact match on clean name
        mask = self.player_data['name_clean'] == name_clean
        if mask.any():
            return self.player_data[mask].iloc[0]
        
        # Try partial match
        mask = self.player_data['name_clean'].str.contains(name_clean, na=False)
        if mask.any():
            return self.player_data[mask].iloc[0]
        
        return None
    
    def calculate_leverage_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate leverage scores for all players
        
        Leverage = (Ceiling / Ownership%) * 100
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with leverage_score column added
        """
        df = df.copy()
        
        # Ensure required columns exist
        if 'ceiling' not in df.columns:
            df['ceiling'] = df['projection'] * 1.4
        
        if 'ownership' not in df.columns:
            df['ownership'] = 15.0  # Default
        
        # Calculate leverage (ceiling points per 1% ownership)
        df['leverage_score'] = df.apply(
            lambda row: (row['ceiling'] / max(row['ownership'], 0.1)) * 100,
            axis=1
        )
        
        # Cap extreme values
        df['leverage_score'] = df['leverage_score'].clip(upper=1000)
        
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
        
        # Factors that drive ownership:
        # 1. Salary (cheaper = more owned)
        # 2. Projection (higher = more owned)
        # 3. Value (proj/$1000)
        
        df['value'] = df['projection'] / (df['salary'] / 1000)
        
        # Normalize factors
        salary_norm = (df['salary'].max() - df['salary']) / (df['salary'].max() - df['salary'].min())
        proj_norm = (df['projection'] - df['projection'].min()) / (df['projection'].max() - df['projection'].min())
        value_norm = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())
        
        # Weighted combination
        ownership_score = (
            salary_norm * 0.3 +
            proj_norm * 0.4 +
            value_norm * 0.3
        )
        
        # Scale to ownership percentage (5% to 40%)
        df['predicted_ownership'] = 5 + (ownership_score * 35)
        
        return df
    
    def identify_chalk_plays(self, df: pd.DataFrame, threshold: float = 30.0) -> List[str]:
        """
        Identify highly-owned "chalk" plays
        
        Args:
            df: DataFrame with ownership data
            threshold: Ownership threshold for chalk (default 30%)
            
        Returns:
            List of player names considered chalk
        """
        ownership_col = 'ownership' if 'ownership' in df.columns else 'predicted_ownership'
        
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
        
        return lineup['leverage_score'].sum()
    
    def calculate_lineup_differentiation(self, lineup: pd.DataFrame, 
                                        chalk_threshold: float = 30.0) -> Dict:
        """
        Calculate how differentiated a lineup is from the field
        
        Args:
            lineup: DataFrame with selected players
            chalk_threshold: Ownership threshold for chalk
            
        Returns:
            Dictionary with differentiation metrics
        """
        ownership_col = 'ownership' if 'ownership' in lineup.columns else 'predicted_ownership'
        
        avg_ownership = lineup[ownership_col].mean()
        
        chalk_count = len(lineup[lineup[ownership_col] >= chalk_threshold])
        
        leverage_score = self.calculate_lineup_leverage(lineup) if 'leverage_score' in lineup.columns else 0
        
        metrics = {
            'avg_ownership': avg_ownership,
            'chalk_count': chalk_count,
            'leverage_score': leverage_score,
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
        
        lineup_own = lineup[ownership_col].mean()
        field_own = self.player_data[ownership_col].mean()
        
        lineup_salary = lineup['salary'].sum()
        avg_salary = self.player_data['salary'].mean() * 9  # 9 players
        
        metrics = {
            'ownership_diff': lineup_own - field_own,
            'salary_diff': lineup_salary - avg_salary,
            'contrarian_score': max(0, field_own - lineup_own)
        }
        
        return metrics
    
    def calculate_lineup_metrics(self, lineup: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive metrics for a lineup
        
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
        
        metrics = {
            'total_projection': lineup_df['projection'].sum(),
            'total_salary': lineup_df['salary'].sum(),
            'avg_ownership': lineup_df.get('ownership', pd.Series([15.0])).mean(),
            'total_ceiling': lineup_df.get('ceiling', lineup_df['projection'] * 1.4).sum(),
            'leverage_score': lineup_df['leverage_score'].sum(),
            'salary_utilization': (lineup_df['salary'].sum() / 50000) * 100
        }
        
        return metrics
    
    def generate_field_simulation(self, n_lineups: int = 100) -> List[pd.DataFrame]:
        """
        Simulate what the field's lineups might look like
        
        Args:
            n_lineups: Number of field lineups to simulate
            
        Returns:
            List of simulated field lineups
        """
        field_lineups = []
        
        # Use ownership-weighted selection
        ownership_col = 'ownership' if 'ownership' in self.player_data.columns else 'predicted_ownership'
        
        for _ in range(n_lineups):
            lineup_players = []
            
            # Select players weighted by ownership
            for position, count in [('QB', 1), ('RB', 2), ('WR', 3), ('TE', 1), ('DST', 1)]:
                pos_players = self.player_data[self.player_data['position'] == position].copy()
                
                if len(pos_players) >= count:
                    weights = pos_players[ownership_col].values
                    weights = weights / weights.sum()
                    
                    selected = pos_players.sample(count, weights=weights, replace=False)
                    lineup_players.append(selected)
            
            if lineup_players:
                lineup = pd.concat(lineup_players, ignore_index=True)
                field_lineups.append(lineup)
        
        return field_lineups
    
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
        
        top_leverage = df.nlargest(top_n, 'leverage_score')[
            ['name', 'position', 'team', 'salary', 'projection', 'ceiling', 
             'ownership', 'leverage_score']
        ].copy()
        
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
        stack_count = len(pass_catchers[pass_catchers['team'] == qb_team])
        
        return {
            'stacked': stack_count > 0,
            'qb_team': qb_team,
            'stack_count': stack_count,
            'correlation_score': stack_count * 1.3  # Correlation multiplier
        }
