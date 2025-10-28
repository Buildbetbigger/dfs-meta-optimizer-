"""
Opponent Modeling Engine
Version 5.0.0

Core opponent modeling logic for DFS lineup optimization:
- Field ownership prediction and analysis
- Leverage score calculation
- Chalk play identification
- Contrarian opportunity detection
- Strategic lineup metrics

BUG FIXES APPLIED:
- Name matching with .strip() to handle whitespace
- Safe player lookups with fallbacks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OpponentModeler:
    """
    Models opponent behavior and calculates strategic advantage metrics
    
    Key Responsibilities:
    1. Calculate leverage scores (ceiling vs ownership)
    2. Identify chalk plays (high ownership)
    3. Find contrarian opportunities (low ownership, high projection)
    4. Analyze field distribution patterns
    5. Provide strategic recommendations
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize opponent modeler with player data
        
        Args:
            players_df: DataFrame with columns: player_name, position, team,
                       salary, projection, ceiling, ownership
        """
        self.players_df = players_df.copy()
        
        # BUG FIX #2: Clean names on initialization
        self._clean_player_names()
        
        # Calculate all metrics
        self._calculate_metrics()
        
        logger.info(f"OpponentModeler initialized with {len(players_df)} players")
    
    def _clean_player_names(self):
        """Clean player names to handle whitespace issues - BUG FIX #2"""
        if 'player_name' in self.players_df.columns:
            self.players_df['player_name'] = self.players_df['player_name'].str.strip()
    
    def _get_player_safe(self, player_name: str) -> Optional[pd.Series]:
        """
        Safely lookup player by name with whitespace handling
        
        Args:
            player_name: Name of player to look up
            
        Returns:
            Player data Series or None if not found
        """
        # BUG FIX #2: Strip whitespace before lookup
        clean_name = str(player_name).strip()
        
        matches = self.players_df[self.players_df['player_name'] == clean_name]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Fallback: Try case-insensitive match
        matches = self.players_df[
            self.players_df['player_name'].str.lower() == clean_name.lower()
        ]
        
        if len(matches) > 0:
            logger.warning(f"Player '{clean_name}' found with case-insensitive match")
            return matches.iloc[0]
        
        logger.warning(f"Player '{clean_name}' not found in opponent model")
        return None
    
    def _calculate_metrics(self):
        """Calculate all opponent modeling metrics"""
        # Ensure required columns exist
        if 'ceiling' not in self.players_df.columns:
            self.players_df['ceiling'] = self.players_df['projection'] * 1.3
        
        if 'ownership' not in self.players_df.columns:
            self.players_df['ownership'] = 15.0  # Default 15%
        
        # Calculate leverage score (ceiling relative to ownership)
        ownership_safe = self.players_df['ownership'].replace(0, 0.1)  # Avoid div by 0
        self.players_df['leverage_score'] = (
            self.players_df['ceiling'] / ownership_safe * 10
        )
        
        # Calculate value score (points per $1K)
        self.players_df['value'] = (
            self.players_df['projection'] / (self.players_df['salary'] / 1000)
        )
        
        # Flag chalk plays (>30% ownership)
        self.players_df['is_chalk'] = self.players_df['ownership'] > 30.0
        
        # Flag contrarian plays (<10% ownership)
        self.players_df['is_contrarian'] = self.players_df['ownership'] < 10.0
        
        # Calculate strategic score (combination of projection, leverage, value)
        self.players_df['strategic_score'] = (
            self.players_df['projection'] * 0.4 +
            self.players_df['leverage_score'] * 0.3 +
            self.players_df['value'] * 30 * 0.3  # Scale value to similar range
        )
        
        logger.info("All metrics calculated successfully")
    
    def identify_chalk_plays(self, threshold: float = 30.0) -> pd.DataFrame:
        """
        Identify high-ownership chalk plays
        
        Args:
            threshold: Ownership percentage threshold for chalk
            
        Returns:
            DataFrame of chalk players sorted by ownership
        """
        chalk = self.players_df[self.players_df['ownership'] >= threshold].copy()
        chalk = chalk.sort_values('ownership', ascending=False)
        
        logger.info(f"Found {len(chalk)} chalk plays (>={threshold}% ownership)")
        return chalk
    
    def identify_leverage_opportunities(self, min_leverage: float = 15.0) -> pd.DataFrame:
        """
        Identify high-leverage plays (good ceiling vs ownership ratio)
        
        Args:
            min_leverage: Minimum leverage score threshold
            
        Returns:
            DataFrame of leverage plays sorted by leverage score
        """
        leverage_plays = self.players_df[
            self.players_df['leverage_score'] >= min_leverage
        ].copy()
        leverage_plays = leverage_plays.sort_values('leverage_score', ascending=False)
        
        logger.info(f"Found {len(leverage_plays)} leverage opportunities (>={min_leverage})")
        return leverage_plays
    
    def identify_contrarian_plays(self, 
                                  max_ownership: float = 10.0,
                                  min_projection: float = None) -> pd.DataFrame:
        """
        Identify contrarian plays (low ownership, solid projection)
        
        Args:
            max_ownership: Maximum ownership for contrarian
            min_projection: Minimum projection threshold (default: median)
            
        Returns:
            DataFrame of contrarian plays
        """
        if min_projection is None:
            min_projection = self.players_df['projection'].median()
        
        contrarian = self.players_df[
            (self.players_df['ownership'] <= max_ownership) &
            (self.players_df['projection'] >= min_projection)
        ].copy()
        contrarian = contrarian.sort_values('projection', ascending=False)
        
        logger.info(f"Found {len(contrarian)} contrarian plays "
                   f"(<={max_ownership}% own, >={min_projection:.1f} proj)")
        return contrarian
    
    def calculate_lineup_leverage(self, lineup_names: List[str]) -> Dict:
        """
        Calculate leverage metrics for a complete lineup
        
        Args:
            lineup_names: List of player names in lineup
            
        Returns:
            Dictionary with leverage metrics
        """
        # Get lineup players with safe lookups
        lineup_players = []
        for name in lineup_names:
            player = self._get_player_safe(name)
            if player is not None:
                lineup_players.append(player.to_dict())
        
        if not lineup_players:
            logger.warning("No valid players found in lineup")
            return {
                'total_projection': 0,
                'total_ceiling': 0,
                'avg_ownership': 0,
                'avg_leverage': 0,
                'chalk_count': 0,
                'contrarian_count': 0
            }
        
        lineup_df = pd.DataFrame(lineup_players)
        
        # Calculate metrics
        metrics = {
            'total_projection': lineup_df['projection'].sum(),
            'total_ceiling': lineup_df['ceiling'].sum(),
            'avg_ownership': lineup_df['ownership'].mean(),
            'avg_leverage': lineup_df['leverage_score'].mean(),
            'chalk_count': (lineup_df['ownership'] > 30).sum(),
            'contrarian_count': (lineup_df['ownership'] < 10).sum(),
            'total_salary': lineup_df['salary'].sum(),
            'avg_value': lineup_df['value'].mean()
        }
        
        return metrics
    
    def analyze_field_distribution(self) -> Dict:
        """
        Analyze overall field ownership distribution
        
        Returns:
            Dictionary with field analysis
        """
        ownership_stats = self.players_df['ownership'].describe()
        
        analysis = {
            'avg_ownership': ownership_stats['mean'],
            'median_ownership': ownership_stats['50%'],
            'ownership_std': ownership_stats['std'],
            'min_ownership': ownership_stats['min'],
            'max_ownership': ownership_stats['max'],
            'chalk_count': (self.players_df['ownership'] > 30).sum(),
            'contrarian_count': (self.players_df['ownership'] < 10).sum(),
            'high_leverage_count': (self.players_df['leverage_score'] > 15).sum(),
            'field_concentration': (self.players_df['ownership'] > 30).sum() / len(self.players_df)
        }
        
        logger.info(f"Field analysis: {analysis['chalk_count']} chalk, "
                   f"{analysis['contrarian_count']} contrarian")
        
        return analysis
    
    def get_strategic_recommendations(self, 
                                     lineup_names: List[str],
                                     contest_type: str = 'GPP') -> Dict:
        """
        Get strategic recommendations for a lineup
        
        Args:
            lineup_names: List of player names in lineup
            contest_type: Type of contest ('GPP', 'CASH', 'TOURNAMENT')
            
        Returns:
            Dictionary with recommendations
        """
        metrics = self.calculate_lineup_leverage(lineup_names)
        
        recommendations = {
            'lineup_type': '',
            'suggestions': [],
            'warnings': [],
            'leverage_rating': 'Unknown'
        }
        
        # Analyze lineup construction
        if metrics['chalk_count'] >= 5:
            recommendations['lineup_type'] = 'Chalk-Heavy'
            recommendations['warnings'].append(
                f"High chalk exposure ({metrics['chalk_count']} players >30% owned)"
            )
        elif metrics['contrarian_count'] >= 4:
            recommendations['lineup_type'] = 'Contrarian'
            recommendations['suggestions'].append(
                "Strong contrarian build for differentiation"
            )
        else:
            recommendations['lineup_type'] = 'Balanced'
        
        # Leverage assessment
        if metrics['avg_leverage'] > 20:
            recommendations['leverage_rating'] = 'Excellent'
        elif metrics['avg_leverage'] > 15:
            recommendations['leverage_rating'] = 'Good'
        elif metrics['avg_leverage'] > 10:
            recommendations['leverage_rating'] = 'Average'
        else:
            recommendations['leverage_rating'] = 'Poor'
            recommendations['warnings'].append(
                "Low average leverage score - consider more contrarian plays"
            )
        
        # Contest-specific recommendations
        if contest_type == 'GPP':
            if metrics['avg_ownership'] > 25:
                recommendations['warnings'].append(
                    "High avg ownership for GPP - consider fading chalk"
                )
            if metrics['avg_leverage'] < 15:
                recommendations['suggestions'].append(
                    "Consider adding higher leverage plays for GPP"
                )
        elif contest_type == 'CASH':
            if metrics['chalk_count'] < 3:
                recommendations['suggestions'].append(
                    "Consider adding more chalk for cash game safety"
                )
        
        return recommendations
    
    def get_players_dataframe(self) -> pd.DataFrame:
        """Get the full players DataFrame with all calculated metrics"""
        return self.players_df.copy()
    
    def get_player_metrics(self, player_name: str) -> Optional[Dict]:
        """
        Get metrics for a specific player
        
        Args:
            player_name: Name of player
            
        Returns:
            Dictionary of player metrics or None if not found
        """
        player = self._get_player_safe(player_name)
        
        if player is None:
            return None
        
        return {
            'player_name': player['player_name'],
            'position': player['position'],
            'team': player['team'],
            'salary': player['salary'],
            'projection': player['projection'],
            'ceiling': player['ceiling'],
            'ownership': player['ownership'],
            'leverage_score': player['leverage_score'],
            'value': player['value'],
            'is_chalk': player['is_chalk'],
            'is_contrarian': player['is_contrarian'],
            'strategic_score': player['strategic_score']
        }
    
    def compare_lineups(self, lineup1: List[str], lineup2: List[str]) -> Dict:
        """
        Compare two lineups on leverage metrics
        
        Args:
            lineup1: First lineup
            lineup2: Second lineup
            
        Returns:
            Comparison dictionary
        """
        metrics1 = self.calculate_lineup_leverage(lineup1)
        metrics2 = self.calculate_lineup_leverage(lineup2)
        
        return {
            'lineup1': metrics1,
            'lineup2': metrics2,
            'projection_diff': metrics1['total_projection'] - metrics2['total_projection'],
            'leverage_diff': metrics1['avg_leverage'] - metrics2['avg_leverage'],
            'ownership_diff': metrics1['avg_ownership'] - metrics2['avg_ownership']
        }


# Helper functions
def create_opponent_model(players_df: pd.DataFrame) -> OpponentModeler:
    """Factory function to create opponent modeler"""
    return OpponentModeler(players_df)


def quick_leverage_analysis(players_df: pd.DataFrame) -> Dict:
    """Quick leverage analysis without full model"""
    model = OpponentModeler(players_df)
    return model.analyze_field_distribution()
