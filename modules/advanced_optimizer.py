"""
Module 2: Advanced Optimizer
Unified interface for advanced lineup generation with stacking and genetic algorithms
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from modules.stacking_engine import StackingEngine
from modules.genetic_optimizer import GeneticOptimizer

logger = logging.getLogger(__name__)


class AdvancedOptimizer:
    """
    Advanced optimizer that integrates stacking and genetic algorithms.
    
    This is the main interface for Module 2 features.
    """
    
    def __init__(
        self,
        players_df: pd.DataFrame,
        opponent_model,
        salary_cap: int = 50000
    ):
        """
        Initialize advanced optimizer.
        
        Args:
            players_df: DataFrame with player data
            opponent_model: OpponentModel instance from Phase 1
            salary_cap: Maximum salary allowed
        """
        self.players_df = players_df.copy()
        self.opponent_model = opponent_model
        self.salary_cap = salary_cap
        
        # Initialize sub-components
        self.stacking_engine = StackingEngine(players_df)
        self.genetic_optimizer = GeneticOptimizer(
            players_df,
            opponent_model,
            self.stacking_engine,
            salary_cap
        )
        
        logger.info("AdvancedOptimizer initialized with all sub-components")
    
    def generate_with_stacking(
        self,
        num_lineups: int = 20,
        mode: str = 'GENETIC_GPP',
        enforce_stacks: bool = True,
        max_ownership: float = None,
        generations: int = 100,
        population_size: int = 200
    ) -> List[Dict]:
        """
        Generate lineups using genetic algorithm with stacking.
        
        This is the primary method for generating optimized lineups.
        
        Args:
            num_lineups: Number of lineups to generate
            mode: Optimization mode (GENETIC_GPP, GENETIC_CASH, GENETIC_CONTRARIAN, LEVERAGE_FIRST)
            enforce_stacks: Require QB stacks
            max_ownership: Maximum total ownership threshold
            generations: Number of GA generations
            population_size: GA population size
        
        Returns:
            List of lineup dictionaries
        """
        logger.info(f"Generating {num_lineups} lineups with mode={mode}")
        
        # Handle leverage-first mode separately
        if mode == 'LEVERAGE_FIRST':
            return self.generate_leverage_first(
                num_lineups=num_lineups,
                enforce_stacks=enforce_stacks,
                min_leverage=2.0
            )
        
        # Configure genetic optimizer
        self.genetic_optimizer.generations = generations
        self.genetic_optimizer.population_size = population_size
        
        # Extract target mode (remove GENETIC_ prefix if present)
        target_mode = mode.replace('GENETIC_', '') if 'GENETIC_' in mode else mode
        
        # Run genetic evolution
        lineups = self.genetic_optimizer.evolve(
            target_mode=target_mode,
            enforce_stack=enforce_stacks,
            max_ownership=max_ownership,
            num_lineups=num_lineups
        )
        
        # Add stacking analysis to each lineup
        for lineup in lineups:
            stacking_metrics = self.stacking_engine.get_stacking_metrics(lineup['players'])
            lineup['stacking_metrics'] = stacking_metrics
        
        logger.info(f"Successfully generated {len(lineups)} lineups")
        
        return lineups
    
    def generate_leverage_first(
        self,
        num_lineups: int = 20,
        enforce_stacks: bool = True,
        min_leverage: float = 2.0,
        target_correlation: float = 60.0
    ) -> List[Dict]:
        """
        Generate lineups using leverage-first strategy.
        
        Strategy:
        1. Identify high-leverage players (ceiling/ownership > threshold)
        2. Build stacks around these players
        3. Fill remaining spots optimizing for correlation
        4. Ensure diversity across lineups
        
        Args:
            num_lineups: Number of lineups to generate
            enforce_stacks: Require QB stacks
            min_leverage: Minimum leverage threshold
            target_correlation: Target correlation score
        
        Returns:
            List of lineup dictionaries
        """
        logger.info(f"Generating {num_lineups} leverage-first lineups")
        
        # Calculate leverage scores
        self.players_df['leverage'] = self.players_df['ceiling'] / self.players_df['ownership'].clip(lower=1.0)
        
        # Get high-leverage players
        leverage_players = self.players_df[
            self.players_df['leverage'] >= min_leverage
        ].sort_values('leverage', ascending=False)
        
        logger.info(f"Found {len(leverage_players)} high-leverage players")
        
        lineups = []
        used_captains = set()
        
        for i in range(num_lineups):
            # Select leverage captain (avoid repeats)
            available_captains = leverage_players[
                ~leverage_players['name'].isin(used_captains)
            ]
            
            if available_captains.empty:
                # Reset and allow repeats
                used_captains.clear()
                available_captains = leverage_players
            
            captain = available_captains.iloc[i % len(available_captains)]
            used_captains.add(captain['name'])
            
            # Build lineup around captain
            lineup = self._build_leverage_lineup(
                captain=captain,
                enforce_stack=enforce_stacks,
                target_correlation=target_correlation
            )
            
            if lineup:
                lineups.append(lineup)
        
        logger.info(f"Generated {len(lineups)} leverage-first lineups")
        
        return lineups
    
    def _build_leverage_lineup(
        self,
        captain: pd.Series,
        enforce_stack: bool,
        target_correlation: float
    ) -> Optional[Dict]:
        """Build a single leverage-focused lineup around a captain"""
        lineup_players = [captain['name']]
        remaining_salary = self.salary_cap - (captain['salary'] * 1.5)
        
        # If captain is QB and enforce_stack, prioritize pass-catchers from same team
        if enforce_stack and captain['position'] == 'QB':
            pass_catchers = self.players_df[
                (self.players_df['team'] == captain['team']) &
                (self.players_df['position'].isin(['WR', 'TE'])) &
                (self.players_df['salary'] <= remaining_salary)
            ].sort_values('leverage', ascending=False)
            
            # Add top 2 pass-catchers if possible
            for _, catcher in pass_catchers.head(2).iterrows():
                if catcher['salary'] <= remaining_salary:
                    lineup_players.append(catcher['name'])
                    remaining_salary -= catcher['salary']
        
        # Fill remaining spots
        while len(lineup_players) < 6:
            # Available players
            available = self.players_df[
                (~self.players_df['name'].isin(lineup_players)) &
                (self.players_df['salary'] <= remaining_salary)
            ]
            
            if available.empty:
                break
            
            # Score by leverage + correlation
            scores = []
            for _, player in available.iterrows():
                test_lineup = lineup_players + [player['name']]
                correlation = self.stacking_engine.calculate_lineup_correlation(test_lineup)
                
                # Combined score
                leverage_score = player['leverage'] / 3.0  # Normalize
                correlation_score = correlation / 100.0
                combined_score = 0.6 * leverage_score + 0.4 * correlation_score
                
                scores.append(combined_score)
            
            # Select best player
            best_idx = np.argmax(scores)
            best_player = available.iloc[best_idx]
            
            lineup_players.append(best_player['name'])
            remaining_salary -= best_player['salary']
        
        # Validate lineup
        if len(lineup_players) != 6:
            return None
        
        # Check stack requirement
        if enforce_stack and not self.stacking_engine._has_qb_stack(lineup_players, min_size=2):
            return None
        
        # Calculate metrics
        lineup_df = self.players_df[self.players_df['name'].isin(lineup_players)]
        
        captain_name = lineup_players[0]
        captain_data = lineup_df[lineup_df['name'] == captain_name].iloc[0]
        
        captain_proj = captain_data['projection'] * 1.5
        flex_proj = lineup_df[lineup_df['name'] != captain_name]['projection'].sum()
        total_proj = captain_proj + flex_proj
        
        captain_ceil = captain_data['ceiling'] * 1.5
        flex_ceil = lineup_df[lineup_df['name'] != captain_name]['ceiling'].sum()
        total_ceil = captain_ceil + flex_ceil
        
        total_ownership = lineup_df['ownership'].sum()
        
        captain_salary = captain_data['salary'] * 1.5
        flex_salary = lineup_df[lineup_df['name'] != captain_name]['salary'].sum()
        total_salary = captain_salary + flex_salary
        
        correlation = self.stacking_engine.calculate_lineup_correlation(lineup_players)
        
        avg_leverage = lineup_df['leverage'].mean()
        
        stacking_metrics = self.stacking_engine.get_stacking_metrics(lineup_players)
        
        return {
            'players': lineup_players,
            'captain': captain_name,
            'metrics': {
                'total_projection': float(total_proj),
                'total_ceiling': float(total_ceil),
                'total_ownership': float(total_ownership),
                'total_salary': int(total_salary),
                'correlation': float(correlation),
                'avg_leverage': float(avg_leverage)
            },
            'stacking_metrics': stacking_metrics
        }
    
    def optimize_for_contest(
        self,
        contest_type: str,
        num_lineups: int
    ) -> List[Dict]:
        """
        Use preset configurations for specific contest types.
        
        Contest Types:
        - GPP: Large-field tournaments (Milly Maker)
        - CASH: Cash games and double-ups
        - SINGLE_ENTRY: Single-entry GPPs
        - SATELLITE: Qualifier tournaments
        
        Args:
            contest_type: Type of contest
            num_lineups: Number of lineups
        
        Returns:
            List of optimized lineups
        """
        presets = {
            'GPP': {
                'mode': 'GENETIC_GPP',
                'enforce_stacks': True,
                'max_ownership': None,
                'generations': 100,
                'population_size': 200
            },
            'MILLY_MAKER': {
                'mode': 'LEVERAGE_FIRST',
                'enforce_stacks': True,
                'max_ownership': None,
                'generations': 150,
                'population_size': 300
            },
            'CASH': {
                'mode': 'GENETIC_CASH',
                'enforce_stacks': True,
                'max_ownership': 200.0,  # Lower ownership for cash
                'generations': 80,
                'population_size': 150
            },
            'SINGLE_ENTRY': {
                'mode': 'GENETIC_CONTRARIAN',
                'enforce_stacks': True,
                'max_ownership': 150.0,
                'generations': 120,
                'population_size': 250
            },
            'SATELLITE': {
                'mode': 'GENETIC_GPP',
                'enforce_stacks': True,
                'max_ownership': 180.0,
                'generations': 100,
                'population_size': 200
            }
        }
        
        preset = presets.get(contest_type.upper(), presets['GPP'])
        
        logger.info(f"Using {contest_type} preset configuration")
        
        return self.generate_with_stacking(
            num_lineups=num_lineups,
            mode=preset['mode'],
            enforce_stacks=preset['enforce_stacks'],
            max_ownership=preset['max_ownership'],
            generations=preset['generations'],
            population_size=preset['population_size']
        )
    
    def analyze_portfolio(self, lineups: List[Dict]) -> Dict:
        """
        Analyze a portfolio of lineups.
        
        Provides metrics on:
        - Player exposure
        - Correlation distribution
        - Leverage distribution
        - Stack coverage
        
        Args:
            lineups: List of lineup dictionaries
        
        Returns:
            Portfolio analysis dictionary
        """
        if not lineups:
            return {}
        
        # Player exposure
        player_counts = {}
        for lineup in lineups:
            for player in lineup['players']:
                player_counts[player] = player_counts.get(player, 0) + 1
        
        # Sort by exposure
        sorted_exposure = sorted(
            player_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate exposure percentages
        exposure_pct = {
            player: (count / len(lineups)) * 100
            for player, count in sorted_exposure
        }
        
        # Correlation stats
        correlations = [lineup['metrics']['correlation'] for lineup in lineups]
        
        # Leverage stats
        leverages = []
        for lineup in lineups:
            lineup_df = self.players_df[self.players_df['name'].isin(lineup['players'])]
            avg_lev = lineup_df['leverage'].mean()
            leverages.append(avg_lev)
        
        # Stack coverage
        qb_stack_count = sum(
            1 for lineup in lineups
            if lineup['stacking_metrics']['qb_stacks'] > 0
        )
        
        # Unique captains
        unique_captains = len(set(lineup['captain'] for lineup in lineups))
        
        return {
            'num_lineups': len(lineups),
            'player_exposure': {
                'top_10': sorted_exposure[:10],
                'exposure_pct': exposure_pct
            },
            'correlation_stats': {
                'mean': float(np.mean(correlations)),
                'median': float(np.median(correlations)),
                'min': float(np.min(correlations)),
                'max': float(np.max(correlations)),
                'std': float(np.std(correlations))
            },
            'leverage_stats': {
                'mean': float(np.mean(leverages)),
                'median': float(np.median(leverages)),
                'min': float(np.min(leverages)),
                'max': float(np.max(leverages))
            },
            'stacking_coverage': {
                'qb_stack_pct': (qb_stack_count / len(lineups)) * 100,
                'unique_captains': unique_captains,
                'captain_diversity': (unique_captains / len(lineups)) * 100
            },
            'projection_stats': {
                'mean': float(np.mean([l['metrics']['total_projection'] for l in lineups])),
                'max': float(np.max([l['metrics']['total_projection'] for l in lineups]))
            },
            'ownership_stats': {
                'mean': float(np.mean([l['metrics']['total_ownership'] for l in lineups])),
                'min': float(np.min([l['metrics']['total_ownership'] for l in lineups])),
                'max': float(np.max([l['metrics']['total_ownership'] for l in lineups]))
            }
        }
    
    def get_stack_recommendations(
        self,
        min_correlation: float = 0.5,
        max_stacks: int = 10
    ) -> List[Dict]:
        """
        Get top stack recommendations.
        
        Args:
            min_correlation: Minimum correlation threshold
            max_stacks: Maximum recommendations
        
        Returns:
            List of stack dictionaries
        """
        recommendations = self.stacking_engine.get_stack_recommendations(
            min_correlation=min_correlation,
            max_stacks=max_stacks
        )
        
        # Convert to dictionaries
        return [
            {
                'primary_player': rec.primary_player,
                'stack_players': rec.stack_players,
                'correlation_score': rec.correlation_score,
                'stack_type': rec.stack_type,
                'total_salary': rec.total_salary,
                'combined_projection': rec.combined_projection,
                'combined_ceiling': rec.combined_ceiling
            }
            for rec in recommendations
        ]
    
    def compare_to_phase1(
        self,
        phase1_lineups: List[Dict],
        module2_lineups: List[Dict]
    ) -> Dict:
        """
        Compare Module 2 lineups to Phase 1 lineups.
        
        Args:
            phase1_lineups: Lineups from Phase 1 optimizer
            module2_lineups: Lineups from Module 2 optimizer
        
        Returns:
            Comparison dictionary with improvements
        """
        # Phase 1 metrics
        phase1_correlations = []
        phase1_qb_stacks = 0
        
        for lineup in phase1_lineups:
            players = lineup.get('players', [])
            corr = self.stacking_engine.calculate_lineup_correlation(players)
            phase1_correlations.append(corr)
            
            if self.stacking_engine._has_qb_stack(players, min_size=2):
                phase1_qb_stacks += 1
        
        # Module 2 metrics
        module2_correlations = [l['metrics']['correlation'] for l in module2_lineups]
        module2_qb_stacks = sum(
            1 for l in module2_lineups
            if l['stacking_metrics']['qb_stacks'] > 0
        )
        
        # Calculate improvements
        corr_improvement = (
            (np.mean(module2_correlations) / np.mean(phase1_correlations) - 1) * 100
            if phase1_correlations else 0
        )
        
        stack_improvement = (
            ((module2_qb_stacks / len(module2_lineups)) / 
             (phase1_qb_stacks / len(phase1_lineups) if phase1_qb_stacks > 0 else 0.01) - 1) * 100
            if phase1_lineups else 0
        )
        
        return {
            'phase1': {
                'avg_correlation': float(np.mean(phase1_correlations)) if phase1_correlations else 0,
                'qb_stack_pct': (phase1_qb_stacks / len(phase1_lineups) * 100) if phase1_lineups else 0
            },
            'module2': {
                'avg_correlation': float(np.mean(module2_correlations)),
                'qb_stack_pct': (module2_qb_stacks / len(module2_lineups) * 100)
            },
            'improvement': {
                'correlation_pct': float(corr_improvement),
                'stacking_pct': float(stack_improvement)
            }
        }
    
    def get_evolution_stats(self) -> Dict:
        """Get genetic algorithm evolution statistics"""
        return self.genetic_optimizer.get_evolution_stats()
