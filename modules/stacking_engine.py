"""
Module 2: Stacking Engine
Handles correlation-based lineup construction and stacking strategies
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StackRecommendation:
    """Represents a recommended stack"""
    primary_player: str
    stack_players: List[str]
    correlation_score: float
    stack_type: str  # 'qb_stack', 'game_stack', 'bring_back'
    total_salary: int
    combined_projection: float
    combined_ceiling: float


class StackingEngine:
    """
    Handles all stacking and correlation logic for lineup construction.
    
    Key Features:
    - Calculates correlation matrix for all player pairs
    - Identifies QB + pass-catcher stacks
    - Detects game stacks (multiple players from same game)
    - Recommends bring-back candidates
    - Scores lineup correlation
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize the stacking engine.
        
        Args:
            players_df: DataFrame with columns:
                - name (str)
                - position (str)
                - team (str)
                - opponent (str)
                - salary (int)
                - projection (float)
                - ceiling (float)
                - ownership (float)
        """
        self.players_df = players_df.copy()
        self.correlation_matrix = None
        self._build_correlation_matrix()
        
        logger.info(f"StackingEngine initialized with {len(players_df)} players")
    
    def _build_correlation_matrix(self):
        """Build correlation matrix for all player pairs"""
        players = self.players_df['name'].tolist()
        n = len(players)
        
        # Initialize matrix
        self.correlation_matrix = pd.DataFrame(
            0.0,
            index=players,
            columns=players
        )
        
        # Calculate correlations
        for i, player1 in enumerate(players):
            p1_data = self.players_df[self.players_df['name'] == player1].iloc[0]
            
            for j, player2 in enumerate(players):
                if i >= j:  # Skip diagonal and already calculated
                    continue
                
                p2_data = self.players_df[self.players_df['name'] == player2].iloc[0]
                
                # Calculate correlation
                corr = self._calculate_player_correlation(p1_data, p2_data)
                
                # Symmetric matrix
                self.correlation_matrix.loc[player1, player2] = corr
                self.correlation_matrix.loc[player2, player1] = corr
        
        logger.info("Correlation matrix built successfully")
    
    def _calculate_player_correlation(self, p1: pd.Series, p2: pd.Series) -> float:
        """
        Calculate correlation between two players.
        
        Correlation factors:
        - Same team QB + pass-catcher: 0.85
        - Same team RB + pass-catcher: 0.30
        - Same team non-stacking positions: 0.20
        - Opposing teams (game stack): 0.40
        - Different games: 0.0
        """
        # Same player
        if p1['name'] == p2['name']:
            return 1.0
        
        # Same team correlations
        if p1['team'] == p2['team']:
            # QB + pass-catcher (WR/TE)
            if (p1['position'] == 'QB' and p2['position'] in ['WR', 'TE']) or \
               (p2['position'] == 'QB' and p1['position'] in ['WR', 'TE']):
                return 0.85
            
            # RB + pass-catcher
            if (p1['position'] == 'RB' and p2['position'] in ['WR', 'TE']) or \
               (p2['position'] == 'RB' and p1['position'] in ['WR', 'TE']):
                return 0.30
            
            # Same team, other combinations
            return 0.20
        
        # Opposing teams (game stack)
        if p1['opponent'] == p2['team'] or p2['opponent'] == p1['team']:
            return 0.40
        
        # Different games
        return 0.0
    
    def get_stack_recommendations(
        self,
        min_correlation: float = 0.5,
        max_stacks: int = 20
    ) -> List[StackRecommendation]:
        """
        Get recommended stacks based on correlation.
        
        Args:
            min_correlation: Minimum correlation threshold
            max_stacks: Maximum number of recommendations
        
        Returns:
            List of stack recommendations sorted by quality
        """
        recommendations = []
        
        # Find QB stacks (QB + pass-catchers)
        qbs = self.players_df[self.players_df['position'] == 'QB']
        
        for _, qb in qbs.iterrows():
            # Get pass-catchers from same team
            pass_catchers = self.players_df[
                (self.players_df['team'] == qb['team']) &
                (self.players_df['position'].isin(['WR', 'TE'])) &
                (self.players_df['name'] != qb['name'])
            ].copy()
            
            if len(pass_catchers) == 0:
                continue
            
            # Sort by projection
            pass_catchers = pass_catchers.sort_values('projection', ascending=False)
            
            # Create 2-player and 3-player stacks
            for stack_size in [2, 3]:
                if len(pass_catchers) < stack_size:
                    continue
                
                stack_players = pass_catchers.head(stack_size)['name'].tolist()
                
                # Calculate stack metrics
                total_salary = qb['salary'] + stack_players.iloc[:stack_size]['salary'].sum()
                combined_proj = qb['projection'] + stack_players.iloc[:stack_size]['projection'].sum()
                combined_ceil = qb['ceiling'] + stack_players.iloc[:stack_size]['ceiling'].sum()
                
                # Average correlation
                correlations = []
                for player in stack_players:
                    correlations.append(self.correlation_matrix.loc[qb['name'], player])
                avg_corr = np.mean(correlations)
                
                if avg_corr >= min_correlation:
                    recommendations.append(StackRecommendation(
                        primary_player=qb['name'],
                        stack_players=stack_players,
                        correlation_score=avg_corr,
                        stack_type='qb_stack',
                        total_salary=int(total_salary),
                        combined_projection=float(combined_proj),
                        combined_ceiling=float(combined_ceil)
                    ))
        
        # Sort by combined ceiling and correlation
        recommendations.sort(
            key=lambda x: (x.combined_ceiling * x.correlation_score),
            reverse=True
        )
        
        return recommendations[:max_stacks]
    
    def get_bring_back_candidates(
        self,
        stack_players: List[str],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get bring-back candidates for a given stack.
        
        Bring-back strategy: Add players from opposing team to hedge.
        
        Args:
            stack_players: List of player names in the stack
            top_n: Number of candidates to return
        
        Returns:
            List of candidate dictionaries
        """
        if not stack_players:
            return []
        
        # Get stack team(s)
        stack_teams = set()
        for player in stack_players:
            player_data = self.players_df[self.players_df['name'] == player]
            if not player_data.empty:
                stack_teams.add(player_data.iloc[0]['team'])
        
        # Find opposing team players
        bring_back_candidates = []
        
        for team in stack_teams:
            # Get opponent
            team_players = self.players_df[self.players_df['team'] == team]
            if team_players.empty:
                continue
            
            opponent = team_players.iloc[0]['opponent']
            
            # Get opponent players
            opp_players = self.players_df[
                (self.players_df['team'] == opponent) &
                (~self.players_df['name'].isin(stack_players))
            ].copy()
            
            # Sort by ceiling (bring-backs need upside)
            opp_players = opp_players.sort_values('ceiling', ascending=False)
            
            # Add top candidates
            for _, player in opp_players.head(top_n).iterrows():
                bring_back_candidates.append({
                    'name': player['name'],
                    'position': player['position'],
                    'team': player['team'],
                    'salary': int(player['salary']),
                    'projection': float(player['projection']),
                    'ceiling': float(player['ceiling']),
                    'ownership': float(player['ownership'])
                })
        
        # Sort by ceiling
        bring_back_candidates.sort(key=lambda x: x['ceiling'], reverse=True)
        
        return bring_back_candidates[:top_n]
    
    def calculate_lineup_correlation(self, lineup_players: List[str]) -> float:
        """
        Calculate overall correlation score for a lineup.
        
        Args:
            lineup_players: List of player names in lineup
        
        Returns:
            Correlation score (0-100)
        """
        if len(lineup_players) < 2:
            return 0.0
        
        # Get all pairwise correlations
        correlations = []
        
        for i, p1 in enumerate(lineup_players):
            for p2 in lineup_players[i+1:]:
                if p1 in self.correlation_matrix.index and p2 in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[p1, p2]
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        # Weighted average (higher correlations matter more)
        weights = np.array(correlations)
        weighted_avg = np.average(correlations, weights=weights)
        
        # Scale to 0-100
        score = weighted_avg * 100
        
        return float(score)
    
    def enforce_stack(
        self,
        lineup_players: List[str],
        stack_type: str = 'qb_stack',
        min_stack_size: int = 2
    ) -> bool:
        """
        Check if lineup has required stack.
        
        Args:
            lineup_players: List of player names
            stack_type: Type of stack required ('qb_stack', 'game_stack')
            min_stack_size: Minimum number of correlated players
        
        Returns:
            True if stack requirement is met
        """
        if stack_type == 'qb_stack':
            return self._has_qb_stack(lineup_players, min_stack_size)
        elif stack_type == 'game_stack':
            return self._has_game_stack(lineup_players, min_stack_size)
        
        return False
    
    def _has_qb_stack(self, lineup_players: List[str], min_size: int) -> bool:
        """Check if lineup has QB + pass-catcher stack"""
        # Get lineup player data
        lineup_df = self.players_df[self.players_df['name'].isin(lineup_players)]
        
        # Find QBs in lineup
        qbs = lineup_df[lineup_df['position'] == 'QB']
        
        if qbs.empty:
            return False
        
        # Check each QB for stacking
        for _, qb in qbs.iterrows():
            # Count pass-catchers from same team
            pass_catchers = lineup_df[
                (lineup_df['team'] == qb['team']) &
                (lineup_df['position'].isin(['WR', 'TE']))
            ]
            
            if len(pass_catchers) >= min_size - 1:  # -1 because QB is included
                return True
        
        return False
    
    def _has_game_stack(self, lineup_players: List[str], min_size: int) -> bool:
        """Check if lineup has game stack (multiple players from same game)"""
        lineup_df = self.players_df[self.players_df['name'].isin(lineup_players)]
        
        # Group by game (team + opponent combination)
        games = {}
        
        for _, player in lineup_df.iterrows():
            # Create unique game identifier
            game_key = tuple(sorted([player['team'], player['opponent']]))
            
            if game_key not in games:
                games[game_key] = []
            
            games[game_key].append(player['name'])
        
        # Check if any game has enough players
        for game_players in games.values():
            if len(game_players) >= min_size:
                return True
        
        return False
    
    def get_stacking_metrics(self, lineup_players: List[str]) -> Dict:
        """
        Get comprehensive stacking metrics for a lineup.
        
        Args:
            lineup_players: List of player names
        
        Returns:
            Dictionary with stacking metrics
        """
        lineup_df = self.players_df[self.players_df['name'].isin(lineup_players)]
        
        # Count QB stacks
        qb_stacks = 0
        qbs = lineup_df[lineup_df['position'] == 'QB']
        
        for _, qb in qbs.iterrows():
            pass_catchers = lineup_df[
                (lineup_df['team'] == qb['team']) &
                (lineup_df['position'].isin(['WR', 'TE']))
            ]
            if len(pass_catchers) > 0:
                qb_stacks += 1
        
        # Count game stacks
        games = {}
        for _, player in lineup_df.iterrows():
            game_key = tuple(sorted([player['team'], player['opponent']]))
            games[game_key] = games.get(game_key, 0) + 1
        
        game_stacks = sum(1 for count in games.values() if count >= 3)
        
        # Calculate correlation score
        correlation_score = self.calculate_lineup_correlation(lineup_players)
        
        return {
            'qb_stacks': qb_stacks,
            'game_stacks': game_stacks,
            'correlation_score': correlation_score,
            'unique_games': len(games),
            'max_game_exposure': max(games.values()) if games else 0
        }
    
    def get_optimal_stacks_for_salary(
        self,
        remaining_salary: int,
        required_positions: List[str] = None
    ) -> List[StackRecommendation]:
        """
        Find optimal stacks within a salary budget.
        
        Args:
            remaining_salary: Available salary
            required_positions: Positions that must be filled
        
        Returns:
            List of affordable stack recommendations
        """
        all_stacks = self.get_stack_recommendations(min_correlation=0.5, max_stacks=50)
        
        # Filter by salary
        affordable_stacks = [
            stack for stack in all_stacks
            if stack.total_salary <= remaining_salary
        ]
        
        # If positions specified, filter accordingly
        if required_positions:
            filtered_stacks = []
            for stack in affordable_stacks:
                stack_positions = set()
                
                # Add primary player position
                primary_data = self.players_df[
                    self.players_df['name'] == stack.primary_player
                ]
                if not primary_data.empty:
                    stack_positions.add(primary_data.iloc[0]['position'])
                
                # Add stack player positions
                for player in stack.stack_players:
                    player_data = self.players_df[self.players_df['name'] == player]
                    if not player_data.empty:
                        stack_positions.add(player_data.iloc[0]['position'])
                
                # Check if stack covers required positions
                if stack_positions.intersection(set(required_positions)):
                    filtered_stacks.append(stack)
            
            affordable_stacks = filtered_stacks
        
        # Sort by value (ceiling per salary)
        affordable_stacks.sort(
            key=lambda x: x.combined_ceiling / x.total_salary,
            reverse=True
        )
        
        return affordable_stacks[:10]
