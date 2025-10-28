"""
Module 2: Stacking Engine
Handles correlation-based lineup construction and stacking strategies

This module provides:
- Correlation matrix calculation for all player pairs
- QB + pass-catcher stack identification  
- Game stack detection (multiple players from same game)
- Bring-back candidate recommendations
- Lineup correlation scoring
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


@dataclass
class StackRecommendation:
    """Represents a recommended stack with all relevant metrics"""
    primary_player: str
    stack_players: List[str]
    correlation_score: float
    stack_type: str  # 'qb_stack', 'game_stack', 'bring_back'
    total_salary: int
    combined_projection: float
    combined_ceiling: float


class StackingEngine:
    """
    Manages all stacking and correlation logic for lineup construction
    
    Key Features:
    - Calculates pairwise correlation for all players based on position and game context
    - Identifies QB + pass-catcher stacks automatically
    - Detects game stack opportunities
    - Recommends optimal bring-back candidates
    - Scores lineup correlation strength (0-100 scale)
    """
    
    # Correlation coefficients based on NFL data analysis
    CORRELATIONS = {
        'qb_wr_same_team': 0.85,      # Very strong positive
        'qb_te_same_team': 0.80,      # Strong positive
        'qb_rb_same_team': 0.25,      # Weak positive
        'wr_wr_same_team': 0.35,      # Moderate positive (compete for targets)
        'wr_te_same_team': 0.30,      # Moderate positive
        'rb_wr_same_team': -0.15,     # Slight negative (game script)
        'rb_te_same_team': -0.10,     # Slight negative
        'same_team_general': 0.20,    # General same team
        'opposing_offense': 0.30,      # Game stack (high scoring)
        'defense_opposing_offense': -0.40,  # Negative correlation
        'opposing_defenses': -0.20     # Negative correlation
    }
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize stacking engine with player data
        
        Args:
            players_df: DataFrame with columns: player_name, position, team, 
                       salary, projection, ceiling, ownership
        """
        self.players_df = players_df.copy()
        self.correlation_matrix = None
        self.stack_opportunities = []
        
        # Build internal structures
        self._validate_data()
        self._build_correlation_matrix()
        self._identify_stacks()
        
        logger.info(f"StackingEngine initialized with {len(players_df)} players")
    
    def _validate_data(self):
        """Validate required columns exist"""
        required = ['player_name', 'position', 'team', 'salary', 'projection']
        missing = [col for col in required if col not in self.players_df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Add default values for optional columns
        if 'ceiling' not in self.players_df.columns:
            self.players_df['ceiling'] = self.players_df['projection'] * 1.3
        
        if 'ownership' not in self.players_df.columns:
            self.players_df['ownership'] = 15.0
    
    def _build_correlation_matrix(self):
        """Build correlation matrix for all player pairs"""
        logger.info("Building correlation matrix...")
        
        n = len(self.players_df)
        matrix = np.zeros((n, n))
        np.fill_diagonal(matrix, 1.0)  # Player correlates perfectly with self
        
        # Calculate pairwise correlations
        for i in range(n):
            for j in range(i + 1, n):
                player1 = self.players_df.iloc[i]
                player2 = self.players_df.iloc[j]
                
                corr = self._calculate_correlation(player1, player2)
                matrix[i, j] = corr
                matrix[j, i] = corr
        
        # Store as DataFrame for easy lookup
        self.correlation_matrix = pd.DataFrame(
            matrix,
            index=self.players_df['player_name'],
            columns=self.players_df['player_name']
        )
        
        logger.info(f"Correlation matrix built: {n}x{n}")
    
    def _calculate_correlation(self, p1: pd.Series, p2: pd.Series) -> float:
        """
        Calculate correlation between two players based on position and team context
        
        Args:
            p1: First player data
            p2: Second player data
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        team1, team2 = p1['team'], p2['team']
        pos1, pos2 = p1['position'], p2['position']
        
        # Same team correlations
        if team1 == team2:
            # QB + pass catcher stacks
            if (pos1 == 'QB' and pos2 in ['WR', 'TE']) or \
               (pos2 == 'QB' and pos1 in ['WR', 'TE']):
                return self.CORRELATIONS['qb_wr_same_team'] if pos2 == 'WR' or pos1 == 'WR' \
                       else self.CORRELATIONS['qb_te_same_team']
            
            # QB + RB
            if (pos1 == 'QB' and pos2 == 'RB') or (pos2 == 'QB' and pos1 == 'RB'):
                return self.CORRELATIONS['qb_rb_same_team']
            
            # Pass catchers together
            if pos1 in ['WR', 'TE'] and pos2 in ['WR', 'TE']:
                return self.CORRELATIONS['wr_wr_same_team']
            
            # RB + pass catcher (negative - game script dependency)
            if (pos1 == 'RB' and pos2 in ['WR', 'TE']) or \
               (pos2 == 'RB' and pos1 in ['WR', 'TE']):
                return self.CORRELATIONS['rb_wr_same_team']
            
            # Default same team
            return self.CORRELATIONS['same_team_general']
        
        # Opposing team correlations (game stacks)
        else:
            # Both offensive players from opposite teams (game stack)
            if pos1 != 'DST' and pos2 != 'DST':
                return self.CORRELATIONS['opposing_offense']
            
            # Defense vs opposing offense (negative)
            if pos1 == 'DST' or pos2 == 'DST':
                return self.CORRELATIONS['defense_opposing_offense']
            
            return 0.0
    
    def _identify_stacks(self):
        """Identify all viable stacking opportunities"""
        logger.info("Identifying stack opportunities...")
        
        self.stack_opportunities = []
        
        # Find QB stacks (QB + pass catchers from same team)
        qbs = self.players_df[self.players_df['position'] == 'QB']
        
        for _, qb in qbs.iterrows():
            qb_team = qb['team']
            qb_name = qb['player_name']
            
            # Find pass catchers on same team
            pass_catchers = self.players_df[
                (self.players_df['team'] == qb_team) &
                (self.players_df['position'].isin(['WR', 'TE']))
            ]
            
            for _, receiver in pass_catchers.iterrows():
                stack = StackRecommendation(
                    primary_player=qb_name,
                    stack_players=[receiver['player_name']],
                    correlation_score=self.CORRELATIONS['qb_wr_same_team'],
                    stack_type='qb_stack',
                    total_salary=qb['salary'] + receiver['salary'],
                    combined_projection=qb['projection'] + receiver['projection'],
                    combined_ceiling=qb['ceiling'] + receiver['ceiling']
                )
                self.stack_opportunities.append(stack)
        
        logger.info(f"Found {len(self.stack_opportunities)} stack opportunities")
    
    def get_qb_stacks(self, min_correlation: float = 0.7) -> List[StackRecommendation]:
        """
        Get recommended QB + pass-catcher stacks
        
        Args:
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of QB stack recommendations
        """
        qb_stacks = [s for s in self.stack_opportunities 
                     if s.stack_type == 'qb_stack' and 
                     s.correlation_score >= min_correlation]
        
        # Sort by combined ceiling projection
        qb_stacks.sort(key=lambda x: x.combined_ceiling, reverse=True)
        
        return qb_stacks
    
    def get_bring_back_candidates(self, stack_players: List[str], 
                                  max_candidates: int = 5) -> List[Dict]:
        """
        Get bring-back candidates for a given stack
        (Opposing team players to include for game stack)
        
        Args:
            stack_players: List of player names in the stack
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of bring-back candidate dictionaries
        """
        # Get teams in the stack
        stack_teams = self.players_df[
            self.players_df['player_name'].isin(stack_players)
        ]['team'].unique()
        
        if len(stack_teams) == 0:
            return []
        
        # Find players from opposing teams (for game stacks)
        # In classic format, would need game_id to match opposing teams
        # For showdown, all players are in same game
        
        candidates = []
        
        for stack_team in stack_teams:
            # Get opponents (in showdown, just different team)
            opponents = self.players_df[
                (self.players_df['team'] != stack_team) &
                (~self.players_df['player_name'].isin(stack_players))
            ].copy()
            
            # Score candidates by ceiling and correlation
            opponents['bring_back_score'] = (
                opponents['ceiling'] * 0.6 +
                opponents['projection'] * 0.4
            )
            
            top_opponents = opponents.nlargest(max_candidates, 'bring_back_score')
            
            for _, player in top_opponents.iterrows():
                candidates.append({
                    'player_name': player['player_name'],
                    'position': player['position'],
                    'team': player['team'],
                    'salary': player['salary'],
                    'projection': player['projection'],
                    'ceiling': player['ceiling'],
                    'bring_back_score': player['bring_back_score']
                })
        
        # Sort by score and limit
        candidates.sort(key=lambda x: x['bring_back_score'], reverse=True)
        return candidates[:max_candidates]
    
    def score_lineup_correlation(self, lineup_names: List[str]) -> float:
        """
        Score the overall correlation strength of a lineup (0-100 scale)
        
        Args:
            lineup_names: List of player names in the lineup
            
        Returns:
            Correlation score (0-100, higher is better)
        """
        if len(lineup_names) < 2:
            return 0.0
        
        # Get all pairwise correlations
        correlations = []
        for i in range(len(lineup_names)):
            for j in range(i + 1, len(lineup_names)):
                p1, p2 = lineup_names[i], lineup_names[j]
                
                if p1 in self.correlation_matrix.index and \
                   p2 in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[p1, p2]
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        # Average correlation, normalized to 0-100 scale
        # Convert from -1:1 range to 0:100
        avg_corr = np.mean(correlations)
        score = ((avg_corr + 1) / 2) * 100
        
        return max(0, min(100, score))
    
    def analyze_lineup_stacking(self, lineup_names: List[str]) -> Dict:
        """
        Comprehensive stacking analysis for a lineup
        
        Args:
            lineup_names: List of player names
            
        Returns:
            Dictionary with detailed stacking metrics
        """
        lineup_players = self.players_df[
            self.players_df['player_name'].isin(lineup_names)
        ]
        
        # Count QB stacks
        qb_count = len(lineup_players[lineup_players['position'] == 'QB'])
        qb_stacks = 0
        
        for _, qb in lineup_players[lineup_players['position'] == 'QB'].iterrows():
            qb_team = qb['team']
            pass_catchers = lineup_players[
                (lineup_players['team'] == qb_team) &
                (lineup_players['position'].isin(['WR', 'TE']))
            ]
            if len(pass_catchers) > 0:
                qb_stacks += 1
        
        # Team distribution
        team_counts = lineup_players['team'].value_counts()
        max_team_exposure = team_counts.max() if len(team_counts) > 0 else 0
        
        # Game stacks (players from multiple teams in same game)
        # For showdown, all players are in same game
        # Count as game stack if 3+ players from 2+ teams
        num_teams = len(team_counts)
        is_game_stack = num_teams >= 2 and len(lineup_names) >= 3
        
        return {
            'correlation_score': self.score_lineup_correlation(lineup_names),
            'qb_stacks': qb_stacks,
            'has_qb_stack': qb_stacks > 0,
            'team_distribution': team_counts.to_dict(),
            'num_teams': num_teams,
            'max_team_exposure': max_team_exposure,
            'is_game_stack': is_game_stack,
            'total_players': len(lineup_names)
        }
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get the full correlation matrix"""
        return self.correlation_matrix.copy()
    
    def enforce_stack(self, lineup_names: List[str], 
                     min_correlation: float = 50.0) -> bool:
        """
        Check if lineup meets minimum stacking requirements
        
        Args:
            lineup_names: List of player names
            min_correlation: Minimum required correlation score
            
        Returns:
            True if lineup meets stacking requirements
        """
        score = self.score_lineup_correlation(lineup_names)
        return score >= min_correlation
    
    def get_stacking_report(self, lineup_names: List[str]) -> str:
        """
        Generate human-readable stacking report
        
        Args:
            lineup_names: List of player names
            
        Returns:
            Formatted report string
        """
        analysis = self.analyze_lineup_stacking(lineup_names)
        lineup_players = self.players_df[
            self.players_df['player_name'].isin(lineup_names)
        ]
        
        report = []
        report.append("=" * 60)
        report.append("🔗 STACKING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\n📊 Overall Metrics:")
        report.append(f"  • Correlation Score: {analysis['correlation_score']:.1f}/100")
        report.append(f"  • QB Stacks: {analysis['qb_stacks']}")
        report.append(f"  • Teams Represented: {analysis['num_teams']}")
        report.append(f"  • Game Stack: {'Yes' if analysis['is_game_stack'] else 'No'}")
        
        report.append(f"\n👥 Team Distribution:")
        for team, count in sorted(analysis['team_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True):
            team_players = lineup_players[lineup_players['team'] == team]
            players_str = ', '.join(team_players['player_name'].tolist())
            report.append(f"  • {team} ({count}): {players_str}")
        
        # Identify stacks
        report.append(f"\n🎯 Detected Stacks:")
        for _, qb in lineup_players[lineup_players['position'] == 'QB'].iterrows():
            qb_team = qb['team']
            receivers = lineup_players[
                (lineup_players['team'] == qb_team) &
                (lineup_players['position'].isin(['WR', 'TE']))
            ]
            if len(receivers) > 0:
                receivers_str = ', '.join(receivers['player_name'].tolist())
                report.append(f"  • QB Stack: {qb['player_name']} + {receivers_str}")
        
        report.append("=" * 60)
        
        return '\n'.join(report)


# Helper functions
def create_stacking_engine(players_df: pd.DataFrame) -> StackingEngine:
    """Factory function to create stacking engine"""
    return StackingEngine(players_df)


def quick_stack_analysis(players_df: pd.DataFrame, 
                         lineup_names: List[str]) -> Dict:
    """Quick stacking analysis without full engine initialization"""
    engine = StackingEngine(players_df)
    return engine.analyze_lineup_stacking(lineup_names)
