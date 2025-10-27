"""
Opponent Modeling Module

Models opponent behavior and calculates leverage scores.
The core strategic engine that differentiates this from traditional optimizers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config.settings import CAPTAIN_MULTIPLIER


class OpponentModel:
    """
    Models opponent behavior to identify leverage opportunities
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize opponent model
        
        Args:
            players_df: DataFrame with player data including ownership projections
        """
        self.players_df = players_df.copy()
        
        # CRITICAL FIX: Clean all player names on initialization
        self.players_df['name'] = (
            self.players_df['name']
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        
        # Calculate leverage scores
        self._calculate_leverage_scores()
    
    def _calculate_leverage_scores(self):
        """
        Calculate leverage score for each player
        
        Leverage = (Ceiling - Projection) / Ownership
        Higher leverage = more upside relative to ownership
        """
        self.players_df['leverage_score'] = (
            (self.players_df['ceiling'] - self.players_df['projection']) /
            (self.players_df['ownership'] + 1)  # +1 to avoid division by zero
        )
        
        # Normalize to 0-100 scale
        min_lev = self.players_df['leverage_score'].min()
        max_lev = self.players_df['leverage_score'].max()
        
        if max_lev > min_lev:
            self.players_df['leverage_score'] = (
                (self.players_df['leverage_score'] - min_lev) /
                (max_lev - min_lev) * 100
            )
        else:
            self.players_df['leverage_score'] = 50  # Default if no variance
    
    def _get_player_data_safe(self, player_name: str) -> Optional[pd.Series]:
        """
        Safely retrieve player data with robust name matching
        
        Args:
            player_name: Player name to look up
            
        Returns:
            Player data as Series, or None if not found
        """
        # Clean the search name
        clean_name = str(player_name).strip()
        
        # Try exact match first
        matches = self.players_df[self.players_df['name'] == clean_name]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Try with additional cleaning
        matches = self.players_df[
            self.players_df['name'].str.strip() == clean_name
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Try case-insensitive match
        matches = self.players_df[
            self.players_df['name'].str.lower() == clean_name.lower()
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Player not found
        print(f"WARNING: Player '{player_name}' not found in DataFrame!")
        return None
    
    def analyze_field_dynamics(self) -> Dict:
        """
        Analyze field dynamics and ownership patterns
        
        Returns:
            Dictionary with analysis results
        """
        # Identify chalk plays (high ownership)
        chalk_threshold = self.players_df['ownership'].quantile(0.75)
        chalk_players = self.players_df[
            self.players_df['ownership'] >= chalk_threshold
        ].sort_values('ownership', ascending=False)
        
        # Identify leverage plays (high leverage score)
        leverage_threshold = self.players_df['leverage_score'].quantile(0.75)
        leverage_plays = self.players_df[
            self.players_df['leverage_score'] >= leverage_threshold
        ].sort_values('leverage_score', ascending=False)
        
        # Identify contrarian plays (low ownership, high projection)
        contrarian_plays = self.players_df[
            (self.players_df['ownership'] < self.players_df['ownership'].quantile(0.25)) &
            (self.players_df['projection'] > self.players_df['projection'].quantile(0.50))
        ].sort_values('projection', ascending=False)
        
        return {
            'chalk_players': chalk_players,
            'leverage_plays': leverage_plays,
            'contrarian_plays': contrarian_plays,
            'leverage_scores': self.players_df[['name', 'position', 'projection', 'ownership', 'leverage_score']]
        }
    
    def calculate_lineup_metrics(self,
                                 flex_players: List[str],
                                 captain: str) -> Dict:
        """
        Calculate comprehensive metrics for a lineup with safe lookups
        
        Args:
            flex_players: List of FLEX player names
            captain: Captain player name
            
        Returns:
            Dictionary with lineup metrics
        """
        # Clean all names first
        clean_captain = str(captain).strip() if captain else ""
        clean_flex = [str(p).strip() for p in flex_players]
        
        # CRITICAL FIX: Safe captain lookup
        captain_data = self._get_player_data_safe(clean_captain)
        
        if captain_data is None:
            # Return default metrics if captain not found
            print(f"WARNING: Captain '{clean_captain}' not found, using defaults")
            return {
                'total_projection': 0,
                'total_ceiling': 0,
                'total_floor': 0,
                'total_salary': 0,
                'avg_ownership': 15.0,
                'lineup_leverage': 0,
                'captain': clean_captain,
                'roster_construction': 'unknown'
            }
        
        # Initialize totals with captain
        total_projection = captain_data['projection'] * CAPTAIN_MULTIPLIER
        total_ceiling = captain_data['ceiling'] * CAPTAIN_MULTIPLIER
        total_floor = captain_data['floor'] * CAPTAIN_MULTIPLIER
        total_salary = captain_data['salary'] * CAPTAIN_MULTIPLIER
        total_ownership = captain_data['ownership']
        total_leverage = captain_data['leverage_score']
        
        player_count = 1
        
        # Add FLEX players with safe lookups
        for player_name in clean_flex:
            player_data = self._get_player_data_safe(player_name)
            
            if player_data is not None:
                total_projection += player_data['projection']
                total_ceiling += player_data['ceiling']
                total_floor += player_data['floor']
                total_salary += player_data['salary']
                total_ownership += player_data['ownership']
                total_leverage += player_data['leverage_score']
                player_count += 1
            else:
                print(f"WARNING: FLEX player '{player_name}' not found, skipping")
        
        # Handle case where no players found
        if player_count == 0:
            return {
                'total_projection': 0,
                'total_ceiling': 0,
                'total_floor': 0,
                'total_salary': 0,
                'avg_ownership': 15.0,
                'lineup_leverage': 0,
                'captain': clean_captain,
                'roster_construction': 'unknown'
            }
        
        # Calculate averages
        avg_ownership = total_ownership / player_count
        lineup_leverage = total_leverage / player_count
        
        # Determine roster construction style
        if avg_ownership > 30:
            construction = 'chalk-heavy'
        elif avg_ownership > 20:
            construction = 'balanced'
        else:
            construction = 'contrarian'
        
        return {
            'total_projection': total_projection,
            'total_ceiling': total_ceiling,
            'total_floor': total_floor,
            'total_salary': total_salary,
            'avg_ownership': avg_ownership,
            'lineup_leverage': lineup_leverage,
            'captain': clean_captain,
            'roster_construction': construction
        }
    
    def predict_field_distribution(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Simulate field lineup distribution based on ownership
        
        Args:
            n_samples: Number of lineups to simulate
            
        Returns:
            DataFrame with simulated field lineups
        """
        # This is a simplified implementation
        # In production, you'd use more sophisticated simulation
        
        simulated_lineups = []
        
        for _ in range(n_samples):
            # Sample players weighted by ownership
            weights = self.players_df['ownership'].values
            weights = weights / weights.sum()
            
            # Sample 6 players (1 captain + 5 flex)
            selected_indices = np.random.choice(
                len(self.players_df),
                size=6,
                replace=False,
                p=weights
            )
            
            lineup_projection = self.players_df.iloc[selected_indices]['projection'].sum()
            lineup_ownership = self.players_df.iloc[selected_indices]['ownership'].mean()
            
            simulated_lineups.append({
                'projection': lineup_projection,
                'ownership': lineup_ownership
            })
        
        return pd.DataFrame(simulated_lineups)
    
    def calculate_win_probability(self,
                                  lineup_projection: float,
                                  field_distribution: pd.DataFrame) -> float:
        """
        Estimate probability of lineup beating the field
        
        Args:
            lineup_projection: Projected points for lineup
            field_distribution: Simulated field lineup distribution
            
        Returns:
            Win probability (0-1)
        """
        # Simple implementation: what % of field does this lineup beat?
        beats = (field_distribution['projection'] < lineup_projection).sum()
        total = len(field_distribution)
        
        return beats / total if total > 0 else 0
    
    def identify_game_stacks(self) -> List[Dict]:
        """
        Identify potential game stacks based on teams
        
        Returns:
            List of stack recommendations
        """
        stacks = []
        
        # Group by team
        for team in self.players_df['team'].unique():
            team_players = self.players_df[self.players_df['team'] == team]
            
            if len(team_players) >= 2:
                # Calculate team metrics
                team_projection = team_players['projection'].sum()
                team_ownership = team_players['ownership'].mean()
                team_leverage = team_players['leverage_score'].mean()
                
                stacks.append({
                    'team': team,
                    'player_count': len(team_players),
                    'total_projection': team_projection,
                    'avg_ownership': team_ownership,
                    'avg_leverage': team_leverage,
                    'players': team_players['name'].tolist()
                })
        
        # Sort by leverage
        stacks = sorted(stacks, key=lambda x: x['avg_leverage'], reverse=True)
        
        return stacks
    
    def get_leverage_opportunities(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top leverage opportunities
        
        Args:
            top_n: Number of opportunities to return
            
        Returns:
            DataFrame with top leverage plays
        """
        return self.players_df.nlargest(top_n, 'leverage_score')[[
            'name',
            'position',
            'team',
            'salary',
            'projection',
            'ceiling',
            'ownership',
            'leverage_score'
        ]]
    
    def get_value_plays(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify value plays (high points per dollar)
        
        Args:
            top_n: Number of plays to return
            
        Returns:
            DataFrame with top value plays
        """
        self.players_df['value'] = self.players_df['projection'] / self.players_df['salary'] * 1000
        
        return self.players_df.nlargest(top_n, 'value')[[
            'name',
            'position',
            'team',
            'salary',
            'projection',
            'ownership',
            'value'
        ]]
    
    def analyze_ownership_tiers(self) -> Dict:
        """
        Analyze players by ownership tiers
        
        Returns:
            Dictionary with players grouped by ownership tiers
        """
        tiers = {
            'chalk': self.players_df[self.players_df['ownership'] >= 30],
            'popular': self.players_df[
                (self.players_df['ownership'] >= 15) &
                (self.players_df['ownership'] < 30)
            ],
            'contrarian': self.players_df[self.players_df['ownership'] < 15]
        }
        
        analysis = {}
        for tier_name, tier_df in tiers.items():
            if len(tier_df) > 0:
                analysis[tier_name] = {
                    'count': len(tier_df),
                    'avg_projection': tier_df['projection'].mean(),
                    'avg_salary': tier_df['salary'].mean(),
                    'avg_leverage': tier_df['leverage_score'].mean(),
                    'players': tier_df['name'].tolist()
                }
            else:
                analysis[tier_name] = {
                    'count': 0,
                    'avg_projection': 0,
                    'avg_salary': 0,
                    'avg_leverage': 0,
                    'players': []
                }
        
        return analysis
