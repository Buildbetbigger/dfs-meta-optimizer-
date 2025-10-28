"""
Module 2: Advanced Optimizer
Unified interface for genetic algorithm and stacking engine

Provides:
- Simple API for lineup generation
- Contest-specific presets (GPP, Cash, Milly Maker, etc.)
- Portfolio analysis
- Performance comparison tools
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from .stacking_engine import StackingEngine
from .genetic_optimizer import GeneticOptimizer
import logging

logger = logging.getLogger(__name__)


class AdvancedOptimizer:
    """
    Unified interface for Module 2 advanced lineup generation
    
    Combines genetic algorithm and stacking engine into a simple API
    with contest-specific presets and portfolio analysis
    """
    
    # Contest presets with optimal settings
    CONTEST_PRESETS = {
        'GPP': {
            'mode': 'GPP',
            'enforce_stacks': True,
            'max_ownership': 45.0,
            'generations': 100,
            'population_size': 200
        },
        'MILLY_MAKER': {
            'mode': 'GPP',
            'enforce_stacks': True,
            'max_ownership': 35.0,  # More contrarian for large field
            'generations': 150,      # More thorough
            'population_size': 300
        },
        'CASH': {
            'mode': 'CASH',
            'enforce_stacks': False,  # Less variance needed
            'max_ownership': 60.0,    # Can play chalk
            'generations': 75,
            'population_size': 150
        },
        'SINGLE_ENTRY': {
            'mode': 'GPP',
            'enforce_stacks': True,
            'max_ownership': 40.0,
            'generations': 120,
            'population_size': 250
        },
        'DOUBLE_UP': {
            'mode': 'CASH',
            'enforce_stacks': False,
            'max_ownership': 55.0,
            'generations': 75,
            'population_size': 150
        },
        'H2H': {
            'mode': 'BALANCED',
            'enforce_stacks': True,
            'max_ownership': 50.0,
            'generations': 80,
            'population_size': 150
        },
        'CONTRARIAN': {
            'mode': 'CONTRARIAN',
            'enforce_stacks': True,
            'max_ownership': 25.0,  # Very low ownership
            'generations': 120,
            'population_size': 200
        },
        'BALANCED': {
            'mode': 'BALANCED',
            'enforce_stacks': True,
            'max_ownership': 45.0,
            'generations': 100,
            'population_size': 200
        }
    }
    
    def __init__(self,
                 players_df: pd.DataFrame,
                 opponent_model=None,
                 salary_cap: int = 50000,
                 roster_size: int = 9):
        """
        Initialize advanced optimizer
        
        Args:
            players_df: DataFrame with player data
            opponent_model: OpponentModeler instance (optional)
            salary_cap: Maximum salary
            roster_size: Number of players in lineup
        """
        self.players_df = players_df
        self.opponent_model = opponent_model
        self.salary_cap = salary_cap
        self.roster_size = roster_size
        
        # Initialize engines
        self.stacking_engine = StackingEngine(players_df)
        self.genetic_optimizer = GeneticOptimizer(
            players_df,
            opponent_model,
            self.stacking_engine,
            salary_cap,
            roster_size
        )
        
        logger.info(f"AdvancedOptimizer initialized with {len(players_df)} players")
    
    def generate_with_stacking(self,
                               num_lineups: int = 20,
                               mode: str = 'GENETIC_GPP',
                               enforce_stacks: bool = True,
                               max_ownership: float = 45.0,
                               **kwargs) -> List[Dict]:
        """
        Generate lineups using genetic algorithm with stacking
        
        Args:
            num_lineups: Number of lineups to generate
            mode: Optimization mode (GENETIC_GPP, GENETIC_CASH, etc.)
            enforce_stacks: Require QB stacks
            max_ownership: Maximum average ownership
            **kwargs: Additional parameters to override
            
        Returns:
            List of lineup dictionaries
        """
        # Parse mode
        if mode.startswith('GENETIC_'):
            target_mode = mode.replace('GENETIC_', '')
        else:
            target_mode = mode
        
        # Apply any kwargs overrides
        generations = kwargs.get('generations', self.genetic_optimizer.generations)
        population_size = kwargs.get('population_size', self.genetic_optimizer.population_size)
        
        self.genetic_optimizer.generations = generations
        self.genetic_optimizer.population_size = population_size
        
        logger.info(f"Generating {num_lineups} lineups with mode={target_mode}, "
                   f"enforce_stacks={enforce_stacks}")
        
        # Run genetic algorithm
        lineups = self.genetic_optimizer.evolve(
            target_mode=target_mode,
            enforce_stacks=enforce_stacks,
            num_lineups=num_lineups,
            max_ownership=max_ownership
        )
        
        return lineups
    
    def generate_from_preset(self,
                            contest_type: str,
                            num_lineups: int = 20) -> List[Dict]:
        """
        Generate lineups using contest-specific preset
        
        Args:
            contest_type: Contest type (GPP, MILLY_MAKER, CASH, etc.)
            num_lineups: Number of lineups to generate
            
        Returns:
            List of lineup dictionaries
        """
        preset = self.CONTEST_PRESETS.get(contest_type.upper())
        
        if not preset:
            logger.warning(f"Unknown contest type '{contest_type}', using GPP")
            preset = self.CONTEST_PRESETS['GPP']
        
        logger.info(f"Using preset for {contest_type}: {preset}")
        
        # Apply preset settings
        self.genetic_optimizer.generations = preset['generations']
        self.genetic_optimizer.population_size = preset['population_size']
        
        # Generate lineups
        return self.generate_with_stacking(
            num_lineups=num_lineups,
            mode=preset['mode'],
            enforce_stacks=preset['enforce_stacks'],
            max_ownership=preset['max_ownership']
        )
    
    def optimize_leverage_first(self,
                                num_lineups: int = 20,
                                min_leverage: float = 2.0) -> List[Dict]:
        """
        Leverage-first optimization: Build around high-leverage plays
        
        Args:
            num_lineups: Number of lineups to generate
            min_leverage: Minimum leverage threshold
            
        Returns:
            List of lineup dictionaries
        """
        logger.info(f"Running leverage-first optimization, min_leverage={min_leverage}")
        
        # Adjust weights to heavily favor leverage
        original_weights = self.genetic_optimizer.weights.copy()
        
        self.genetic_optimizer.weights = {
            'projection': 0.15,
            'ceiling': 0.25,
            'leverage': 0.45,  # Heavy leverage weighting
            'correlation': 0.10,
            'ownership': 0.05
        }
        
        # Generate lineups
        lineups = self.genetic_optimizer.evolve(
            target_mode='GPP',
            enforce_stacks=True,
            num_lineups=num_lineups,
            max_ownership=50.0
        )
        
        # Restore original weights
        self.genetic_optimizer.weights = original_weights
        
        return lineups
    
    def get_stacking_recommendations(self,
                                    stack_type: str = 'qb_stack',
                                    min_correlation: float = 0.7,
                                    top_n: int = 10) -> List[Dict]:
        """
        Get top stacking recommendations
        
        Args:
            stack_type: Type of stack ('qb_stack', 'game_stack')
            min_correlation: Minimum correlation threshold
            top_n: Number of recommendations to return
            
        Returns:
            List of stack recommendations
        """
        logger.info(f"Getting {stack_type} recommendations")
        
        if stack_type == 'qb_stack':
            stacks = self.stacking_engine.get_qb_stacks(min_correlation)
            return [vars(s) for s in stacks[:top_n]]
        
        return []
    
    def analyze_portfolio(self, lineups: List[Dict]) -> Dict:
        """
        Analyze portfolio of lineups for exposure and correlation
        
        Args:
            lineups: List of lineup dictionaries
            
        Returns:
            Portfolio analysis dictionary
        """
        if not lineups:
            return {}
        
        logger.info(f"Analyzing portfolio of {len(lineups)} lineups")
        
        # Extract all players
        all_players = []
        for lineup in lineups:
            all_players.extend([p['player_name'] for p in lineup['players']])
        
        # Calculate exposure
        from collections import Counter
        exposure = Counter(all_players)
        total_lineups = len(lineups)
        
        exposure_pct = {
            player: (count / total_lineups) * 100 
            for player, count in exposure.items()
        }
        
        # Sort by exposure
        top_exposure = sorted(exposure_pct.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate portfolio metrics
        total_proj = sum(lu['total_projection'] for lu in lineups)
        total_salary = sum(lu['total_salary'] for lu in lineups)
        total_ownership = sum(lu['avg_ownership'] for lu in lineups)
        
        # Correlation stats
        correlation_scores = [lu.get('correlation_score', 0) for lu in lineups]
        
        # Uniqueness (how different are lineups from each other)
        unique_players = len(set(all_players))
        total_slots = total_lineups * self.roster_size
        uniqueness_pct = (unique_players / min(total_slots, len(self.players_df))) * 100
        
        return {
            'num_lineups': total_lineups,
            'total_players_used': unique_players,
            'uniqueness_pct': uniqueness_pct,
            'avg_projection': total_proj / total_lineups,
            'avg_salary': total_salary / total_lineups,
            'avg_ownership': total_ownership / total_lineups,
            'avg_correlation': np.mean(correlation_scores),
            'top_exposure': top_exposure[:15],
            'exposure_distribution': exposure_pct,
            'correlation_range': {
                'min': min(correlation_scores) if correlation_scores else 0,
                'max': max(correlation_scores) if correlation_scores else 0,
                'std': np.std(correlation_scores) if correlation_scores else 0
            }
        }
    
    def compare_to_phase1(self,
                         phase1_lineups: List[Dict],
                         module2_lineups: List[Dict]) -> Dict:
        """
        Compare Phase 1 lineups to Module 2 lineups
        
        Args:
            phase1_lineups: Lineups from Phase 1 optimizer
            module2_lineups: Lineups from Module 2 (this)
            
        Returns:
            Comparison dictionary
        """
        logger.info("Comparing Phase 1 vs Module 2")
        
        # Analyze both portfolios
        phase1_analysis = self.analyze_portfolio(phase1_lineups)
        module2_analysis = self.analyze_portfolio(module2_lineups)
        
        # Calculate improvements
        correlation_improvement = (
            (module2_analysis['avg_correlation'] - phase1_analysis.get('avg_correlation', 0)) /
            max(phase1_analysis.get('avg_correlation', 1), 1) * 100
        )
        
        uniqueness_improvement = (
            (module2_analysis['uniqueness_pct'] - phase1_analysis.get('uniqueness_pct', 0)) /
            max(phase1_analysis.get('uniqueness_pct', 1), 1) * 100
        )
        
        return {
            'phase1': phase1_analysis,
            'module2': module2_analysis,
            'improvement': {
                'correlation_pct': correlation_improvement,
                'uniqueness_pct': uniqueness_improvement,
                'projection_diff': module2_analysis['avg_projection'] - phase1_analysis.get('avg_projection', 0),
                'ownership_diff': module2_analysis['avg_ownership'] - phase1_analysis.get('avg_ownership', 15)
            }
        }
    
    def get_evolution_stats(self) -> Dict:
        """Get statistics from last evolution run"""
        return self.genetic_optimizer.get_evolution_stats()
    
    def set_custom_weights(self, weights: Dict):
        """
        Set custom fitness weights
        
        Args:
            weights: Dictionary with keys: projection, ceiling, leverage, correlation, ownership
        """
        self.genetic_optimizer.weights = weights
        logger.info(f"Custom weights set: {weights}")
    
    def get_available_presets(self) -> List[str]:
        """Get list of available contest presets"""
        return list(self.CONTEST_PRESETS.keys())
    
    def print_preset_details(self, contest_type: str):
        """
        Print detailed preset configuration
        
        Args:
            contest_type: Contest type to show details for
        """
        preset = self.CONTEST_PRESETS.get(contest_type.upper())
        
        if not preset:
            print(f"Unknown contest type: {contest_type}")
            return
        
        print(f"\n{'='*60}")
        print(f"Preset: {contest_type.upper()}")
        print(f"{'='*60}")
        for key, value in preset.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
    
    def save_lineups_csv(self, lineups: List[Dict], filename: str):
        """
        Save lineups to CSV file
        
        Args:
            lineups: List of lineup dictionaries
            filename: Output filename
        """
        if not lineups:
            logger.warning("No lineups to save")
            return
        
        # Flatten to rows
        rows = []
        for i, lineup in enumerate(lineups, 1):
            for player in lineup['players']:
                rows.append({
                    'lineup_num': i,
                    'player_name': player['player_name'],
                    'position': player['position'],
                    'team': player['team'],
                    'salary': player['salary'],
                    'projection': player['projection'],
                    'ownership': player.get('ownership', 0)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(lineups)} lineups to {filename}")


# Convenience functions
def create_optimizer(players_df: pd.DataFrame,
                    opponent_model=None,
                    salary_cap: int = 50000,
                    roster_size: int = 9) -> AdvancedOptimizer:
    """Factory function to create optimizer"""
    return AdvancedOptimizer(players_df, opponent_model, salary_cap, roster_size)


def quick_generate(players_df: pd.DataFrame,
                  contest_type: str = 'GPP',
                  num_lineups: int = 20) -> List[Dict]:
    """Quick lineup generation with preset"""
    optimizer = AdvancedOptimizer(players_df)
    return optimizer.generate_from_preset(contest_type, num_lineups)
