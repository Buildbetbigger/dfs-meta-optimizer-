"""
Lineup Optimization Engine

Generates optimal DFS lineups using intelligent algorithms.

BUG FIXES APPLIED:
#2: Safe name matching with _get_player_data_safe()
#3: True diversity with progressive randomization
#4: Salary optimization to 96-100% usage
#6: Safe .get() for all config access
#7: Proper NaN handling throughout
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import random
from itertools import combinations

class LineupOptimizer:
    """
    Advanced lineup optimizer with bug fixes for production use
    """
    
    def __init__(self, player_data: pd.DataFrame, opponent_modeler, mode_config: Dict):
        """
        Initialize the optimizer
        
        Args:
            player_data: DataFrame with player information
            opponent_modeler: OpponentModeler instance
            mode_config: Configuration dictionary for optimization mode
        """
        # Clean all player names on initialization
        self.player_data = player_data.copy()
        self.player_data['name_clean'] = self.player_data['name'].astype(str).str.strip().str.lower()
        
        self.opponent_modeler = opponent_modeler
        self.mode_config = mode_config
        
        # Extract configuration with safe defaults (BUG FIX #6)
        self.salary_cap = 50000
        self.roster_size = 9
        
        # Weights for scoring - safe access with .get()
        self.projection_weight = mode_config.get('projection_weight', 1.0)
        self.ceiling_weight = mode_config.get('ceiling_weight', 0.5)
        self.leverage_weight = mode_config.get('leverage_weight', 0.3)
        self.ownership_penalty = mode_config.get('ownership_penalty', 0.1)
        
        # Lineup generation settings
        self.population_size = mode_config.get('population_size', 100)
        self.generations = mode_config.get('generations', 50)
        self.mutation_rate = mode_config.get('mutation_rate', 0.2)
        
        # Position requirements (standard DraftKings)
        self.position_requirements = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'FLEX': 1,  # RB/WR/TE
            'DST': 1
        }
        
        # Valid FLEX positions
        self.flex_positions = {'RB', 'WR', 'TE'}
        
        # Track generated lineups
        self.generated_lineups = []
        self.player_usage = {}
        
        print(f"✅ LineupOptimizer initialized with {len(self.player_data)} players")
    
    def _get_player_data_safe(self, player_name: str) -> Optional[pd.Series]:
        """
        Safely retrieve player data with multiple matching strategies
        
        BUG FIX #2: Multiple fallback strategies for name matching
        
        Args:
            player_name: Name of the player to look up
            
        Returns:
            Player data Series or None if not found
        """
        if pd.isna(player_name):
            return None
            
        # Clean the input name
        name_clean = str(player_name).strip().lower()
        
        # Strategy 1: Exact match on cleaned name
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
        
        # Not found
        return None
    
    def calculate_lineup_score(self, lineup: pd.DataFrame) -> float:
        """
        Calculate weighted score for a lineup
        
        BUG FIX #7: NaN handling in scoring calculations
        
        Args:
            lineup: DataFrame with selected players
            
        Returns:
            Weighted score
        """
        # Base scores with NaN protection
        projection_score = float(lineup['projection'].fillna(0).sum()) * self.projection_weight
        
        # Ceiling score
        ceiling_values = lineup.get('ceiling', lineup['projection'] * 1.4)
        ceiling_score = float(ceiling_values.fillna(0).sum()) * self.ceiling_weight
        
        # Leverage score
        leverage_values = lineup.get('leverage_score', pd.Series([0] * len(lineup)))
        leverage_score = float(leverage_values.fillna(0).sum()) * self.leverage_weight
        
        # Ownership penalty
        ownership_values = lineup.get('ownership', pd.Series([15.0] * len(lineup)))
        avg_ownership = float(ownership_values.fillna(15.0).mean())
        ownership_score = -avg_ownership * self.ownership_penalty
        
        total_score = projection_score + ceiling_score + leverage_score + ownership_score
        
        # Handle NaN result
        if np.isnan(total_score) or np.isinf(total_score):
            return 0.0
        
        return total_score
    
    def is_valid_lineup(self, lineup: pd.DataFrame, min_salary_pct: float = 0.96) -> bool:
        """
        Check if lineup meets all constraints
        
        Args:
            lineup: DataFrame with selected players
            min_salary_pct: Minimum salary cap utilization (0.96 = 96%)
            
        Returns:
            True if valid, False otherwise
        """
        # Check roster size
        if len(lineup) != self.roster_size:
            return False
        
        # Check salary constraints
        total_salary = lineup['salary'].sum()
        min_salary = self.salary_cap * min_salary_pct
        if total_salary < min_salary or total_salary > self.salary_cap:
            return False
        
        # Check position requirements
        positions = lineup['position'].value_counts().to_dict()
        
        # Core positions
        if positions.get('QB', 0) != 1:
            return False
        if positions.get('TE', 0) < 1:
            return False
        if positions.get('DST', 0) != 1:
            return False
        
        # RB + WR + TE must total 6
        flex_count = positions.get('RB', 0) + positions.get('WR', 0) + positions.get('TE', 0)
        if flex_count != 6:
            return False
        
        # RB minimum 2
        if positions.get('RB', 0) < 2:
            return False
        
        # WR minimum 3
        if positions.get('WR', 0) < 3:
            return False
        
        return True
    
    def optimize_lineup_salary(self, lineup: pd.DataFrame, min_salary_pct: float = 0.96) -> pd.DataFrame:
        """
        Optimize lineup to use more salary cap
        
        BUG FIX #4: Ensures 96-100% salary cap usage
        
        Args:
            lineup: Current lineup
            min_salary_pct: Target minimum salary percentage
            
        Returns:
            Optimized lineup
        """
        current_salary = lineup['salary'].sum()
        target_salary = self.salary_cap * min_salary_pct
        
        if current_salary >= target_salary:
            return lineup
        
        # Try to upgrade cheaper players
        lineup_sorted = lineup.sort_values('salary')
        
        for idx, player in lineup_sorted.iterrows():
            if current_salary >= target_salary:
                break
            
            # Find more expensive replacements
            position = player['position']
            available_salary = self.salary_cap - (current_salary - player['salary'])
            
            # Get candidates
            candidates = self.player_data[
                (self.player_data['position'] == position) &
                (self.player_data['salary'] > player['salary']) &
                (self.player_data['salary'] <= available_salary) &
                (~self.player_data['name'].isin(lineup['name']))
            ].copy()
            
            if len(candidates) > 0:
                # Sort by value (projection per $1000)
                candidates['value'] = candidates['projection'] / (candidates['salary'] / 1000)
                best_replacement = candidates.nlargest(1, 'value')
                
                if len(best_replacement) > 0:
                    # Make the swap
                    lineup = lineup.drop(idx)
                    lineup = pd.concat([lineup, best_replacement], ignore_index=True)
                    current_salary = lineup['salary'].sum()
        
        return lineup
    
    def generate_random_lineup(self, min_salary_pct: float = 0.96, randomness: float = 0.0) -> Optional[pd.DataFrame]:
        """
        Generate a random valid lineup
        
        BUG FIX #3: Added randomness parameter for diversity
        
        Args:
            min_salary_pct: Minimum salary cap utilization
            randomness: 0-1, higher = more random selection
            
        Returns:
            Valid lineup or None if unable to generate
        """
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            lineup_players = []
            remaining_salary = self.salary_cap
            
            # Select QB (weighted by projection + randomness)
            qbs = self.player_data[self.player_data['position'] == 'QB'].copy()
            if len(qbs) == 0:
                continue
            
            # Add randomness to weights
            base_weights = qbs['projection'].values
            if randomness > 0:
                random_factor = np.random.random(len(qbs)) * randomness
                weights = base_weights * (1 + random_factor)
            else:
                weights = base_weights
            
            weights = weights / weights.sum()
            qb = qbs.sample(1, weights=weights).iloc[0]
            lineup_players.append(qb)
            remaining_salary -= qb['salary']
            
            # Select DST
            dsts = self.player_data[self.player_data['position'] == 'DST'].copy()
            if len(dsts) == 0:
                continue
            
            base_weights = dsts['projection'].values
            if randomness > 0:
                random_factor = np.random.random(len(dsts)) * randomness
                weights = base_weights * (1 + random_factor)
            else:
                weights = base_weights
            
            weights = weights / weights.sum()
            dst = dsts.sample(1, weights=weights).iloc[0]
            lineup_players.append(dst)
            remaining_salary -= dst['salary']
            
            # Select flex positions (2 RB, 3 WR, 1 TE)
            selected_names = {qb['name'], dst['name']}
            
            # RBs
            rbs = self.player_data[
                (self.player_data['position'] == 'RB') &
                (~self.player_data['name'].isin(selected_names)) &
                (self.player_data['salary'] <= remaining_salary)
            ].copy()
            
            if len(rbs) < 2:
                continue
            
            base_weights = rbs['projection'].values
            if randomness > 0:
                random_factor = np.random.random(len(rbs)) * randomness
                weights = base_weights * (1 + random_factor)
            else:
                weights = base_weights
            
            weights = weights / weights.sum()
            selected_rbs = rbs.sample(2, weights=weights, replace=False)
            
            for _, rb in selected_rbs.iterrows():
                lineup_players.append(rb)
                selected_names.add(rb['name'])
                remaining_salary -= rb['salary']
            
            # WRs
            wrs = self.player_data[
                (self.player_data['position'] == 'WR') &
                (~self.player_data['name'].isin(selected_names)) &
                (self.player_data['salary'] <= remaining_salary)
            ].copy()
            
            if len(wrs) < 3:
                continue
            
            base_weights = wrs['projection'].values
            if randomness > 0:
                random_factor = np.random.random(len(wrs)) * randomness
                weights = base_weights * (1 + random_factor)
            else:
                weights = base_weights
            
            weights = weights / weights.sum()
            selected_wrs = wrs.sample(3, weights=weights, replace=False)
            
            for _, wr in selected_wrs.iterrows():
                lineup_players.append(wr)
                selected_names.add(wr['name'])
                remaining_salary -= wr['salary']
            
            # TE
            tes = self.player_data[
                (self.player_data['position'] == 'TE') &
                (~self.player_data['name'].isin(selected_names)) &
                (self.player_data['salary'] <= remaining_salary)
            ].copy()
            
            if len(tes) < 1:
                continue
            
            base_weights = tes['projection'].values
            if randomness > 0:
                random_factor = np.random.random(len(tes)) * randomness
                weights = base_weights * (1 + random_factor)
            else:
                weights = base_weights
            
            weights = weights / weights.sum()
            te = tes.sample(1, weights=weights).iloc[0]
            lineup_players.append(te)
            remaining_salary -= te['salary']
            
            # Create lineup DataFrame
            lineup = pd.DataFrame(lineup_players)
            
            # Optimize salary usage
            lineup = self.optimize_lineup_salary(lineup, min_salary_pct)
            
            # Validate
            if self.is_valid_lineup(lineup, min_salary_pct):
                return lineup
        
        return None
    
    def is_duplicate_lineup(self, lineup: pd.DataFrame, existing_lineups: List[pd.DataFrame], 
                           threshold: int = 7) -> bool:
        """
        Check if lineup is too similar to existing lineups
        
        Args:
            lineup: Lineup to check
            existing_lineups: List of existing lineups
            threshold: Number of matching players to consider duplicate
            
        Returns:
            True if duplicate, False otherwise
        """
        lineup_names = set(lineup['name'].values)
        
        for existing in existing_lineups:
            existing_names = set(existing['name'].values)
            overlap = len(lineup_names.intersection(existing_names))
            
            if overlap >= threshold:
                return True
        
        return False
    
    def generate_lineups(self, num_lineups: int = 20, diversity_factor: float = 0.3,
                        min_salary_pct: float = 0.96) -> List[pd.DataFrame]:
        """
        Generate multiple diverse lineups
        
        BUG FIX #3: Progressive randomization for true diversity
        
        Args:
            num_lineups: Number of lineups to generate
            diversity_factor: Controls uniqueness (0-1, higher = more unique)
            min_salary_pct: Minimum salary cap utilization
            
        Returns:
            List of lineup DataFrames
        """
        lineups = []
        max_attempts = num_lineups * 20
        attempts = 0
        
        # Calculate duplicate threshold based on diversity
        duplicate_threshold = int(9 - (diversity_factor * 3))  # 9 to 6 matching players
        
        print(f"\n🧬 Generating {num_lineups} lineups (diversity: {diversity_factor:.1f})...")
        print(f"   Duplicate threshold: {duplicate_threshold} matching players")
        
        while len(lineups) < num_lineups and attempts < max_attempts:
            attempts += 1
            
            # Progressive randomization - increase as attempts grow
            randomness_boost = min(attempts / (max_attempts / 2), 1.0) * diversity_factor
            
            lineup = self.generate_random_lineup(min_salary_pct, randomness=randomness_boost)
            
            if lineup is None:
                continue
            
            # Check for duplicates
            if self.is_duplicate_lineup(lineup, lineups, duplicate_threshold):
                continue
            
            # Calculate score
            score = self.calculate_lineup_score(lineup)
            lineup['score'] = score
            
            lineups.append(lineup)
            
            if len(lineups) % 5 == 0:
                print(f"   Generated {len(lineups)}/{num_lineups} lineups...")
        
        if len(lineups) < num_lineups:
            print(f"⚠️  Only generated {len(lineups)} unique lineups (requested {num_lineups})")
        else:
            print(f"✅ Successfully generated {len(lineups)} unique lineups")
        
        # Clean up names in all lineups
        for lineup in lineups:
            lineup['name'] = lineup['name'].astype(str).str.strip()
        
        return lineups
    
    def get_player_usage_stats(self, lineups: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate player exposure across lineups
        
        Args:
            lineups: List of lineups
            
        Returns:
            Dictionary of player names to exposure percentages
        """
        player_counts = {}
        
        for lineup in lineups:
            for _, player in lineup.iterrows():
                name = str(player['name']).strip()
                player_counts[name] = player_counts.get(name, 0) + 1
        
        total_lineups = len(lineups)
        exposures = {
            name: (count / total_lineups) * 100
            for name, count in player_counts.items()
        }
        
        return exposures
    
    def analyze_portfolio(self, lineups: List[pd.DataFrame]) -> Dict:
        """
        Analyze lineup portfolio for quality metrics
        
        Args:
            lineups: List of lineups
            
        Returns:
            Dictionary of portfolio metrics
        """
        if not lineups:
            return {}
        
        metrics = {
            'num_lineups': len(lineups),
            'avg_projection': np.mean([lu['projection'].sum() for lu in lineups]),
            'avg_salary': np.mean([lu['salary'].sum() for lu in lineups]),
            'avg_ownership': np.mean([lu.get('ownership', pd.Series([15.0])).mean() for lu in lineups]),
            'salary_efficiency': [],
            'unique_players': set()
        }
        
        # Calculate additional metrics
        for lineup in lineups:
            total_salary = lineup['salary'].sum()
            metrics['salary_efficiency'].append(total_salary / self.salary_cap)
            metrics['unique_players'].update(lineup['name'].values)
        
        metrics['unique_players'] = len(metrics['unique_players'])
        metrics['avg_salary_efficiency'] = np.mean(metrics['salary_efficiency']) * 100
        
        # Get player exposure stats safely
        exposures = {}
        for lineup in lineups:
            for _, player in lineup.iterrows():
                # Get player data safely
                player_data = self._get_player_data_safe(player['name'])
                if player_data is not None:
                    name = str(player['name']).strip()
                    if name not in exposures:
                        exposures[name] = {
                            'count': 0,
                            'ownership': float(player_data.get('ownership', 15.0)),
                            'leverage': float(player_data.get('leverage_score', 100))
                        }
                    exposures[name]['count'] += 1
        
        # Calculate exposure percentages
        for name in exposures:
            exposures[name]['exposure'] = (exposures[name]['count'] / len(lineups)) * 100
        
        # Most/least exposed (if we have data)
        if exposures:
            sorted_exposure = sorted(exposures.items(), key=lambda x: x[1]['exposure'], reverse=True)
            metrics['most_exposed'] = sorted_exposure[:5] if len(sorted_exposure) >= 5 else sorted_exposure
            metrics['least_exposed'] = sorted_exposure[-5:] if len(sorted_exposure) >= 5 else []
        
        return metrics
