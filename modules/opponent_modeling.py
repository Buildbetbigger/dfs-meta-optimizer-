"""
Opponent Modeling Engine

This module implements the core opponent modeling logic:
- Predicts field ownership
- Calculates leverage scores
- Identifies chalk plays
- Models field distribution
- Determines anti-chalk strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config.settings import (
    HIGH_OWNERSHIP_THRESHOLD,
    LOW_OWNERSHIP_THRESHOLD,
    MIN_ACCEPTABLE_LEVERAGE,
    CAPTAIN_MULTIPLIER
)


class OpponentModel:
    """
    Models opponent behavior and calculates strategic metrics
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize the opponent model
        
        Args:
            players_df: DataFrame with player data
        """
        self.players_df = players_df.copy()
        
        # CRITICAL FIX: Clean all player names on initialization
        self.players_df['name'] = (
            self.players_df['name']
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        
        self._calculate_all_metrics()
    
    def _get_player_data_safe(self, player_name: str) -> Optional[pd.Series]:
        """
        Safely retrieve player data with robust name matching
        
        Args:
            player_name: Player name to look up
            
        Returns:
            Player data as Series, or None if not found
        """
        from typing import Optional
        
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
        print(f"WARNING: Player '{player_name}' not found in opponent model DataFrame!")
        print(f"Available names: {self.players_df['name'].tolist()[:10]}...")
        return None
    
    def _calculate_all_metrics(self):
        """Calculate all opponent modeling metrics"""
        self.players_df['leverage'] = self._calculate_leverage()
        self.players_df['chalk_flag'] = self._identify_chalk()
        self.players_df['contrarian_flag'] = self._identify_contrarian()
        self.players_df['value_score'] = self._calculate_value_score()
        self.players_df['strategic_score'] = self._calculate_strategic_score()
    
    def _calculate_leverage(self) -> pd.Series:
        """
        Calculate leverage score for each player
        
        Leverage = (Ceiling / Ownership) * 100
        
        High leverage = tournament-winning upside relative to ownership
        """
        # Avoid division by zero
        ownership_safe = self.players_df['ownership'].replace(0, 0.1)
        
        leverage = (self.players_df['ceiling'] / ownership_safe) * 100
        
        return leverage
    
    def _identify_chalk(self) -> pd.Series:
        """
        Identify chalk (high-owned) players
        
        Returns:
            Boolean series indicating chalk players
        """
        return self.players_df['ownership'] >= HIGH_OWNERSHIP_THRESHOLD
    
    def _identify_contrarian(self) -> pd.Series:
        """
        Identify contrarian (low-owned) players
        
        Returns:
            Boolean series indicating contrarian players
        """
        return self.players_df['ownership'] <= LOW_OWNERSHIP_THRESHOLD
    
    def _calculate_value_score(self) -> pd.Series:
        """
        Calculate value score (points per $1000)
        
        Returns:
            Value score for each player
        """
        return (self.players_df['projection'] / self.players_df['salary']) * 1000
    
    def _calculate_strategic_score(self, 
                                   leverage_weight: float = 0.5,
                                   projection_weight: float = 0.5) -> pd.Series:
        """
        Calculate overall strategic score combining leverage and projection
        
        This is the key metric for opponent modeling - balances raw
        projection with leverage/ownership considerations
        
        Args:
            leverage_weight: Weight for leverage component (0-1)
            projection_weight: Weight for projection component (0-1)
        
        Returns:
            Strategic score for each player
        """
        # Normalize leverage and projection to 0-1 scale
        leverage_norm = (self.players_df['leverage'] - self.players_df['leverage'].min()) / \
                       (self.players_df['leverage'].max() - self.players_df['leverage'].min())
        
        proj_norm = (self.players_df['projection'] - self.players_df['projection'].min()) / \
                   (self.players_df['projection'].max() - self.players_df['projection'].min())
        
        # Combine with weights
        strategic_score = (leverage_norm * leverage_weight) + (proj_norm * projection_weight)
        
        return strategic_score
    
    def get_chalk_players(self) -> pd.DataFrame:
        """
        Get all chalk players
        
        Returns:
            DataFrame of high-owned players
        """
        return self.players_df[self.players_df['chalk_flag']].sort_values(
            'ownership', ascending=False
        )
    
    def get_leverage_plays(self, min_leverage: float = MIN_ACCEPTABLE_LEVERAGE) -> pd.DataFrame:
        """
        Get high-leverage plays
        
        Args:
            min_leverage: Minimum leverage score threshold
        
        Returns:
            DataFrame of high-leverage players
        """
        return self.players_df[
            self.players_df['leverage'] >= min_leverage
        ].sort_values('leverage', ascending=False)
    
    def get_anti_chalk_candidates(self) -> pd.DataFrame:
        """
        Get players suitable for anti-chalk strategy
        
        Returns low-owned players with decent projection/ceiling
        
        Returns:
            DataFrame of anti-chalk candidates
        """
        # Low owned, but still reasonable projection
        median_proj = self.players_df['projection'].median()
        
        return self.players_df[
            (self.players_df['contrarian_flag']) &
            (self.players_df['projection'] >= median_proj * 0.7)
        ].sort_values('leverage', ascending=False)
    
    def predict_field_distribution(self) -> Dict[str, float]:
        """
        Model the expected field distribution
        
        Returns:
            Dictionary with field distribution statistics
        """
        return {
            'avg_ownership': self.players_df['ownership'].mean(),
            'chalk_count': self.players_df['chalk_flag'].sum(),
            'contrarian_count': self.players_df['contrarian_flag'].sum(),
            'avg_leverage': self.players_df['leverage'].mean(),
            'max_leverage': self.players_df['leverage'].max(),
            'field_concentration': self._calculate_field_concentration()
        }
    
    def _calculate_field_concentration(self) -> float:
        """
        Calculate how concentrated ownership is (Herfindahl index)
        
        Higher = more concentrated (chalky)
        Lower = more distributed (contrarian-friendly)
        
        Returns:
            Concentration score (0-1)
        """
        # Normalize ownership to sum to 1
        ownership_pct = self.players_df['ownership'] / 100
        
        # Calculate Herfindahl index
        herfindahl = (ownership_pct ** 2).sum()
        
        return herfindahl
    
    def calculate_lineup_metrics(self, lineup: List[str], 
                                 captain: str) -> Dict[str, float]:
        """
        Calculate strategic metrics for a complete lineup
        
        Args:
            lineup: List of player names in the lineup
            captain: Name of the captain
        
        Returns:
            Dictionary of lineup metrics
        """
        # CRITICAL FIX: Clean captain name before lookup
        captain = str(captain).strip()
        
        # CRITICAL FIX: Use safe lookup for captain
        captain_data = self._get_player_data_safe(captain)
        
        if captain_data is None:
            # Captain not found - return ZERO salary to trigger rejection
            print(f"ERROR: Captain '{captain}' not found in opponent model!")
            return {
                'total_projection': 0.0,
                'total_ceiling': 0.0,
                'total_floor': 0.0,
                'avg_ownership': 15.0,
                'uniqueness': 85.0,
                'lineup_leverage': 0.0,
                'chalk_count': 0,
                'contrarian_count': 0,
                'total_salary': 0,  # CHANGED: Return 0 to fail validation
                'salary_remaining': 50000
            }
        
        # CRITICAL FIX: Look up each player individually using safe method
        flex_players_data = []
        missing_players = []
        
        for player_name in lineup:
            # Skip captain
            if str(player_name).strip() == captain:
                continue
            
            player_data = self._get_player_data_safe(player_name)
            if player_data is not None:
                flex_players_data.append(player_data)
            else:
                missing_players.append(player_name)
        
        # If any flex players missing, return ZERO salary to fail validation
        if len(missing_players) > 0:
            print(f"ERROR: Missing flex players: {missing_players}")
            return {
                'total_projection': 0.0,
                'total_ceiling': 0.0,
                'total_floor': 0.0,
                'avg_ownership': 15.0,
                'uniqueness': 85.0,
                'lineup_leverage': 0.0,
                'chalk_count': 0,
                'contrarian_count': 0,
                'total_salary': 0,  # CHANGED: Return 0 to fail validation
                'salary_remaining': 50000
            }
        
        # Should have exactly 5 flex players
        if len(flex_players_data) != 5:
            print(f"ERROR: Wrong number of flex players: {len(flex_players_data)} (expected 5)")
            return {
                'total_projection': 0.0,
                'total_ceiling': 0.0,
                'total_floor': 0.0,
                'avg_ownership': 15.0,
                'uniqueness': 85.0,
                'lineup_leverage': 0.0,
                'chalk_count': 0,
                'contrarian_count': 0,
                'total_salary': 0,  # CHANGED: Return 0 to fail validation
                'salary_remaining': 50000
            }
        
        # Convert list of Series to DataFrame
        flex_df = pd.DataFrame(flex_players_data)
        
        # Calculate total projection
        captain_points = captain_data['projection'] * CAPTAIN_MULTIPLIER
        flex_points = flex_df['projection'].sum()
        total_projection = captain_points + flex_points
        
        # Calculate total ceiling
        captain_ceiling = captain_data['ceiling'] * CAPTAIN_MULTIPLIER
        flex_ceiling = flex_df['ceiling'].sum()
        total_ceiling = captain_ceiling + flex_ceiling
        
        # Calculate ownership metrics
        captain_own = captain_data['ownership'] * 1.5  # Weight captain more
        flex_own = flex_df['ownership'].sum()
        avg_ownership = (captain_own + flex_own) / 6  # Always 6 players
        
        # Calculate uniqueness (inverse of ownership)
        uniqueness = 100 - avg_ownership
        
        # Calculate lineup leverage
        lineup_leverage = total_ceiling / (avg_ownership + 1)  # Avoid div by 0
        
        # Count chalk and contrarian players
        chalk_count = int(captain_data.get('chalk_flag', False)) + flex_df['chalk_flag'].sum()
        contrarian_count = int(captain_data.get('contrarian_flag', False)) + flex_df['contrarian_flag'].sum()
        
        # CRITICAL FIX: Calculate salary correctly
        captain_salary = captain_data['salary'] * CAPTAIN_MULTIPLIER
        flex_salary = flex_df['salary'].sum()
        total_salary = captain_salary + flex_salary
        
        # DEBUG: Print salary breakdown
        print(f"\n   💰 Salary breakdown:")
        print(f"      Captain ({captain}): ${captain_data['salary']:,} × 1.5 = ${captain_salary:,}")
        print(f"      Flex total (5 players): ${flex_salary:,}")
        print(f"      TOTAL: ${total_salary:,} ({(total_salary/50000)*100:.1f}%)")
        
        return {
            'total_projection': total_projection,
            'total_ceiling': total_ceiling,
            'total_floor': captain_data['floor'] * CAPTAIN_MULTIPLIER + flex_df['floor'].sum(),
            'avg_ownership': avg_ownership,
            'uniqueness': uniqueness,
            'lineup_leverage': lineup_leverage,
            'chalk_count': chalk_count,
            'contrarian_count': contrarian_count,
            'total_salary': total_salary,
            'salary_remaining': 50000 - total_salary
        }
    
    def recommend_anti_chalk_strategy(self) -> Dict:
        """
        Recommend specific anti-chalk strategy based on field analysis
        
        Returns:
            Dictionary with strategy recommendations
        """
        field_dist = self.predict_field_distribution()
        chalk_players = self.get_chalk_players()
        
        # Determine strategy intensity
        if field_dist['field_concentration'] > 0.15:
            strategy = 'AGGRESSIVE_ANTI_CHALK'
            recommendation = (
                "Field is highly concentrated. Strongly fade chalk and "
                "build maximum differentiation lineups."
            )
        elif field_dist['field_concentration'] > 0.10:
            strategy = 'MODERATE_ANTI_CHALK'
            recommendation = (
                "Field shows some concentration. Mix chalk fades with "
                "leverage plays for balanced differentiation."
            )
        else:
            strategy = 'BALANCED'
            recommendation = (
                "Field is well distributed. Focus on leverage over "
                "pure contrarian plays."
            )
        
        return {
            'strategy': strategy,
            'recommendation': recommendation,
            'top_chalk_to_fade': chalk_players.head(3)['name'].tolist(),
            'top_leverage_plays': self.get_leverage_plays().head(5)['name'].tolist(),
            'field_concentration': field_dist['field_concentration'],
            'avg_field_ownership': field_dist['avg_ownership']
        }
    
    def simulate_contest_outcomes(self, 
                                  lineups: List[List[str]], 
                                  num_simulations: int = 1000) -> pd.DataFrame:
        """
        Simulate contest outcomes using Monte Carlo method
        
        Args:
            lineups: List of lineups to simulate
            num_simulations: Number of simulations to run
        
        Returns:
            DataFrame with simulation results
        """
        results = []
        
        for sim in range(num_simulations):
            # Simulate player scores using normal distribution
            # Mean = projection, SD = (ceiling - floor) / 4
            simulated_scores = {}
            
            for _, player in self.players_df.iterrows():
                std_dev = (player['ceiling'] - player['floor']) / 4
                score = np.random.normal(player['projection'], std_dev)
                score = max(0, score)  # No negative scores
                simulated_scores[player['name']] = score
            
            # Score each lineup
            for idx, lineup in enumerate(lineups):
                captain = lineup[0]  # Assume first player is captain
                
                captain_score = simulated_scores[captain] * CAPTAIN_MULTIPLIER
                flex_scores = sum(simulated_scores[p] for p in lineup[1:])
                total_score = captain_score + flex_scores
                
                results.append({
                    'simulation': sim,
                    'lineup_id': idx,
                    'total_score': total_score
                })
        
        results_df = pd.DataFrame(results)
        
        # Calculate win probability for each lineup
        win_probs = []
        for lineup_id in range(len(lineups)):
            lineup_sims = results_df[results_df['lineup_id'] == lineup_id]
            
            # Count how many times this lineup had the highest score
            wins = 0
            for sim in range(num_simulations):
                sim_scores = results_df[results_df['simulation'] == sim]
                max_score = sim_scores['total_score'].max()
                lineup_score = sim_scores[
                    sim_scores['lineup_id'] == lineup_id
                ]['total_score'].values[0]
                
                if lineup_score >= max_score:
                    wins += 1
            
            win_probs.append({
                'lineup_id': lineup_id,
                'win_probability': wins / num_simulations,
                'avg_score': lineup_sims['total_score'].mean(),
                'max_score': lineup_sims['total_score'].max(),
                'min_score': lineup_sims['total_score'].min()
            })
        
        return pd.DataFrame(win_probs)
    
    def get_players_dataframe(self) -> pd.DataFrame:
        """
        Get the full players dataframe with all calculated metrics
        
        Returns:
            Complete players DataFrame
        """
        return self.players_df.copy()
