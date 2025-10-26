"""
Module 3: Portfolio Optimizer
Multi-entry portfolio optimization with exposure management
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from copy import deepcopy

from modules.exposure_manager import ExposureManager
from modules.advanced_optimizer import AdvancedOptimizer

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimizes portfolios of lineups for multi-entry contests.
    
    Features:
    - Exposure-aware lineup generation
    - Diversity optimization
    - Captain distribution management
    - Multi-objective portfolio scoring
    - Iterative portfolio building
    """
    
    def __init__(
        self,
        players_df: pd.DataFrame,
        opponent_model,
        salary_cap: int = 50000
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            players_df: DataFrame with player data
            opponent_model: OpponentModel instance
            salary_cap: Maximum salary allowed
        """
        self.players_df = players_df.copy()
        self.opponent_model = opponent_model
        self.salary_cap = salary_cap
        
        # Initialize sub-components
        self.exposure_manager = ExposureManager(players_df)
        self.advanced_optimizer = AdvancedOptimizer(
            players_df,
            opponent_model,
            salary_cap
        )
        
        logger.info("PortfolioOptimizer initialized")
    
    def generate_portfolio(
        self,
        num_lineups: int,
        mode: str = 'GENETIC_GPP',
        enforce_stacks: bool = True,
        max_player_exposure: float = 40.0,
        min_unique_players_per_lineup: int = 3,
        target_captain_diversity: float = 0.30,
        generations_per_batch: int = 100,
        batch_size: int = 50
    ) -> List[Dict]:
        """
        Generate a portfolio of lineups with exposure controls.
        
        Strategy:
        1. Set exposure limits
        2. Generate lineups in batches
        3. Check exposure after each batch
        4. Adjust future batches based on current exposure
        5. Ensure diversity across portfolio
        
        Args:
            num_lineups: Total number of lineups to generate
            mode: Optimization mode (GENETIC_GPP, GENETIC_CASH, etc.)
            enforce_stacks: Require QB stacks
            max_player_exposure: Maximum exposure % for any player
            min_unique_players_per_lineup: Min different players between lineups
            target_captain_diversity: Target % of unique captains
            generations_per_batch: GA generations per batch
            batch_size: Lineups per batch
        
        Returns:
            List of lineup dictionaries with exposure controls
        """
        logger.info(f"Generating portfolio of {num_lineups} lineups")
        
        # Set global exposure limit
        self.exposure_manager.set_global_max_exposure(max_player_exposure)
        
        portfolio = []
        batches = int(np.ceil(num_lineups / batch_size))
        
        for batch_num in range(batches):
            remaining_lineups = num_lineups - len(portfolio)
            current_batch_size = min(batch_size, remaining_lineups)
            
            logger.info(f"Batch {batch_num + 1}/{batches}: Generating {current_batch_size} lineups")
            
            # Generate batch with exposure awareness
            batch_lineups = self._generate_batch_with_exposure(
                num_lineups=current_batch_size,
                existing_portfolio=portfolio,
                total_target_lineups=num_lineups,
                mode=mode,
                enforce_stacks=enforce_stacks,
                min_unique_players=min_unique_players_per_lineup,
                target_captain_diversity=target_captain_diversity,
                generations=generations_per_batch
            )
            
            portfolio.extend(batch_lineups)
            
            # Check exposure after batch
            compliance = self.exposure_manager.check_exposure_compliance(portfolio)
            
            if compliance['total_violations'] > 0:
                logger.warning(f"Batch {batch_num + 1} has {compliance['total_violations']} exposure violations")
            
            logger.info(f"Portfolio size: {len(portfolio)}/{num_lineups}")
        
        # Final compliance check
        final_compliance = self.exposure_manager.check_exposure_compliance(portfolio)
        
        if final_compliance['compliant']:
            logger.info(f"✓ Portfolio generated with full exposure compliance")
        else:
            logger.warning(f"Portfolio has {final_compliance['total_violations']} violations")
        
        return portfolio
    
    def _generate_batch_with_exposure(
        self,
        num_lineups: int,
        existing_portfolio: List[Dict],
        total_target_lineups: int,
        mode: str,
        enforce_stacks: bool,
        min_unique_players: int,
        target_captain_diversity: float,
        generations: int
    ) -> List[Dict]:
        """Generate a batch of lineups with exposure checking"""
        
        # Generate larger pool to select from
        pool_size = num_lineups * 3
        
        # Use advanced optimizer to generate candidate lineups
        candidate_pool = self.advanced_optimizer.generate_with_stacking(
            num_lineups=pool_size,
            mode=mode,
            enforce_stacks=enforce_stacks,
            generations=generations,
            population_size=200
        )
        
        # Select lineups that meet exposure constraints
        selected_lineups = []
        
        # Calculate current exposures
        current_exposure = self.exposure_manager.calculate_current_exposure(existing_portfolio)
        
        # Track captain usage
        captain_counts = {}
        for lineup in existing_portfolio:
            captain = lineup['captain']
            captain_counts[captain] = captain_counts.get(captain, 0) + 1
        
        # Try to add lineups from pool
        for candidate in candidate_pool:
            if len(selected_lineups) >= num_lineups:
                break
            
            # Check exposure compliance
            if not self.exposure_manager.enforce_exposure_on_lineup(
                candidate,
                existing_portfolio + selected_lineups,
                total_target_lineups
            ):
                continue
            
            # Check diversity vs existing portfolio
            if not self._check_lineup_diversity(
                candidate,
                existing_portfolio + selected_lineups,
                min_unique_players
            ):
                continue
            
            # Check captain diversity
            candidate_captain = candidate['captain']
            current_captain_count = captain_counts.get(candidate_captain, 0)
            total_lineups_so_far = len(existing_portfolio) + len(selected_lineups)
            
            if total_lineups_so_far > 0:
                captain_exposure = (current_captain_count + 1) / (total_lineups_so_far + 1)
                
                # Don't exceed captain diversity target
                if captain_exposure > (1.0 / max(1, total_target_lineups * target_captain_diversity)):
                    # Skip if this captain is overused, unless we're running out of options
                    if len(candidate_pool) - candidate_pool.index(candidate) > num_lineups:
                        continue
            
            # Add lineup
            selected_lineups.append(candidate)
            captain_counts[candidate_captain] = captain_counts.get(candidate_captain, 0) + 1
        
        # If we didn't get enough lineups, fill with best available
        if len(selected_lineups) < num_lineups:
            logger.warning(f"Only generated {len(selected_lineups)}/{num_lineups} with strict constraints")
            
            # Relax constraints and add more
            for candidate in candidate_pool:
                if len(selected_lineups) >= num_lineups:
                    break
                
                if candidate in selected_lineups:
                    continue
                
                # Only check diversity (relaxed exposure)
                if self._check_lineup_diversity(
                    candidate,
                    existing_portfolio + selected_lineups,
                    min_unique_players
                ):
                    selected_lineups.append(candidate)
        
        logger.info(f"Selected {len(selected_lineups)} lineups for batch")
        
        return selected_lineups
    
    def _check_lineup_diversity(
        self,
        candidate: Dict,
        existing_lineups: List[Dict],
        min_unique_players: int
    ) -> bool:
        """Check if lineup has enough unique players vs existing lineups"""
        
        if not existing_lineups:
            return True
        
        candidate_players = set(candidate['players'])
        
        # Check against all existing lineups
        for existing in existing_lineups:
            existing_players = set(existing['players'])
            
            # Count overlapping players
            overlap = len(candidate_players & existing_players)
            unique_players = 6 - overlap
            
            if unique_players < min_unique_players:
                return False
        
        return True
    
    def optimize_existing_portfolio(
        self,
        lineups: List[Dict],
        target_exposure: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Optimize an existing portfolio by swapping players.
        
        Args:
            lineups: Current lineup portfolio
            target_exposure: Optional dict of {player_name: target_exposure_%}
        
        Returns:
            Optimized portfolio
        """
        optimized = deepcopy(lineups)
        
        # Get current exposure
        current_exposure = self.exposure_manager.calculate_current_exposure(optimized)
        
        # Identify overexposed players
        overexposed = []
        for player, exp in current_exposure.items():
            if exp > self.exposure_manager.global_max_exposure:
                overexposed.append(player)
        
        if not overexposed:
            logger.info("Portfolio already compliant")
            return optimized
        
        logger.info(f"Optimizing portfolio: {len(overexposed)} overexposed players")
        
        # For each overexposed player, try to replace in some lineups
        for player in overexposed:
            target_exp = self.exposure_manager.global_max_exposure
            current_exp = current_exposure[player]
            
            # How many lineups to remove player from
            lineups_to_modify = int(np.ceil(((current_exp - target_exp) / 100) * len(optimized)))
            
            logger.info(f"Attempting to remove {player} from {lineups_to_modify} lineups")
            
            # Find lineups containing this player
            lineups_with_player = [
                (i, lineup) for i, lineup in enumerate(optimized)
                if player in lineup['players']
            ]
            
            # Sort by impact (remove from lowest projection lineups first)
            lineups_with_player.sort(key=lambda x: x[1]['metrics']['total_projection'])
            
            modified_count = 0
            
            for idx, lineup in lineups_with_player[:lineups_to_modify]:
                # Try to replace player
                replacement = self._find_replacement_player(
                    lineup,
                    player,
                    optimized
                )
                
                if replacement:
                    # Create modified lineup
                    new_players = [p if p != player else replacement for p in lineup['players']]
                    
                    # Recalculate metrics
                    modified_lineup = self._recalculate_lineup_metrics(new_players)
                    
                    optimized[idx] = modified_lineup
                    modified_count += 1
            
            logger.info(f"Modified {modified_count} lineups for {player}")
        
        return optimized
    
    def _find_replacement_player(
        self,
        lineup: Dict,
        player_to_replace: str,
        portfolio: List[Dict]
    ) -> Optional[str]:
        """Find a suitable replacement player for lineup"""
        
        # Get player to replace data
        player_data = self.players_df[self.players_df['name'] == player_to_replace]
        
        if player_data.empty:
            return None
        
        player_info = player_data.iloc[0]
        
        # Calculate available salary
        lineup_df = self.players_df[self.players_df['name'].isin(lineup['players'])]
        current_salary = lineup_df['salary'].sum()
        
        # Account for captain multiplier
        if player_to_replace == lineup['captain']:
            available_salary = player_info['salary'] * 1.5
        else:
            available_salary = player_info['salary']
        
        # Find candidates (same position, affordable, not in lineup, underexposed)
        current_exposure = self.exposure_manager.calculate_current_exposure(portfolio)
        
        candidates = self.players_df[
            (self.players_df['position'] == player_info['position']) &
            (self.players_df['salary'] <= available_salary) &
            (~self.players_df['name'].isin(lineup['players']))
        ].copy()
        
        if candidates.empty:
            return None
        
        # Add exposure data
        candidates['current_exposure'] = candidates['name'].map(
            lambda x: current_exposure.get(x, 0.0)
        )
        
        # Prefer underexposed players with good projections
        candidates['score'] = (
            candidates['projection'] * 0.6 +
            (100 - candidates['current_exposure']) * 0.4
        )
        
        # Select best candidate
        best_candidate = candidates.nlargest(1, 'score')
        
        if not best_candidate.empty:
            return best_candidate.iloc[0]['name']
        
        return None
    
    def _recalculate_lineup_metrics(self, players: List[str]) -> Dict:
        """Recalculate metrics for a modified lineup"""
        
        lineup_df = self.players_df[self.players_df['name'].isin(players)]
        
        if len(lineup_df) != 6:
            return None
        
        captain = players[0]
        captain_data = lineup_df[lineup_df['name'] == captain].iloc[0]
        
        # Calculate metrics
        captain_proj = captain_data['projection'] * 1.5
        flex_proj = lineup_df[lineup_df['name'] != captain]['projection'].sum()
        total_proj = captain_proj + flex_proj
        
        captain_ceil = captain_data['ceiling'] * 1.5
        flex_ceil = lineup_df[lineup_df['name'] != captain]['ceiling'].sum()
        total_ceil = captain_ceil + flex_ceil
        
        total_ownership = lineup_df['ownership'].sum()
        
        captain_salary = captain_data['salary'] * 1.5
        flex_salary = lineup_df[lineup_df['name'] != captain]['salary'].sum()
        total_salary = captain_salary + flex_salary
        
        correlation = self.advanced_optimizer.stacking_engine.calculate_lineup_correlation(players)
        
        stacking_metrics = self.advanced_optimizer.stacking_engine.get_stacking_metrics(players)
        
        return {
            'players': players,
            'captain': captain,
            'metrics': {
                'total_projection': float(total_proj),
                'total_ceiling': float(total_ceil),
                'total_ownership': float(total_ownership),
                'total_salary': int(total_salary),
                'correlation': float(correlation)
            },
            'stacking_metrics': stacking_metrics
        }
    
    def generate_tiered_portfolio(
        self,
        num_lineups: int,
        tier_distribution: Dict[str, float] = None,
        mode: str = 'GENETIC_GPP'
    ) -> Dict[str, List[Dict]]:
        """
        Generate portfolio with different optimization tiers.
        
        Example tiers:
        - Safe (40%): High projection, lower variance
        - Balanced (40%): Standard optimization
        - Contrarian (20%): Low ownership, high upside
        
        Args:
            num_lineups: Total number of lineups
            tier_distribution: Dict of {tier_name: percentage}
            mode: Base optimization mode
        
        Returns:
            Dictionary of {tier_name: [lineups]}
        """
        if tier_distribution is None:
            tier_distribution = {
                'safe': 0.30,
                'balanced': 0.50,
                'contrarian': 0.20
            }
        
        logger.info(f"Generating tiered portfolio: {num_lineups} lineups")
        
        tiers = {}
        
        for tier_name, percentage in tier_distribution.items():
            tier_size = int(num_lineups * percentage)
            
            if tier_size == 0:
                continue
            
            logger.info(f"Generating {tier_name} tier: {tier_size} lineups")
            
            # Adjust mode based on tier
            if tier_name == 'safe':
                tier_mode = 'GENETIC_CASH'
                max_exposure = 50.0
            elif tier_name == 'contrarian':
                tier_mode = 'GENETIC_CONTRARIAN'
                max_exposure = 30.0
            else:  # balanced
                tier_mode = mode
                max_exposure = 40.0
            
            # Generate tier lineups
            tier_lineups = self.generate_portfolio(
                num_lineups=tier_size,
                mode=tier_mode,
                max_player_exposure=max_exposure,
                batch_size=min(20, tier_size)
            )
            
            tiers[tier_name] = tier_lineups
        
        return tiers
    
    def get_portfolio_metrics(self, lineups: List[Dict]) -> Dict:
        """
        Get comprehensive portfolio metrics.
        
        Args:
            lineups: Portfolio of lineups
        
        Returns:
            Dictionary with portfolio-level metrics
        """
        if not lineups:
            return {}
        
        # Use advanced optimizer's portfolio analysis
        base_metrics = self.advanced_optimizer.analyze_portfolio(lineups)
        
        # Add exposure analysis
        exposure_report = self.exposure_manager.get_exposure_report(lineups)
        compliance = self.exposure_manager.check_exposure_compliance(lineups)
        
        # Calculate portfolio diversity
        unique_players = len(set(
            player
            for lineup in lineups
            for player in lineup['players']
        ))
        
        total_player_slots = len(lineups) * 6
        portfolio_diversity = (unique_players / len(self.players_df)) * 100
        
        return {
            **base_metrics,
            'exposure_compliance': {
                'compliant': compliance['compliant'],
                'violations': compliance['total_violations'],
                'warnings': compliance['total_warnings']
            },
            'portfolio_diversity': {
                'unique_players': unique_players,
                'total_players_available': len(self.players_df),
                'diversity_pct': portfolio_diversity,
                'avg_players_per_lineup': unique_players / len(lineups) if lineups else 0
            },
            'top_exposures': exposure_report.head(10).to_dict('records') if not exposure_report.empty else []
        }
