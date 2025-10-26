"""
Module 3: Lineup Filter
Deduplication and filtering tools for lineup portfolios
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class LineupFilter:
    """
    Filters and deduplicates lineup portfolios.
    
    Features:
    - Exact duplicate detection
    - Similarity-based filtering
    - Quality-based filtering
    - Correlation-based filtering
    - Captain-based filtering
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize lineup filter.
        
        Args:
            players_df: DataFrame with player data
        """
        self.players_df = players_df.copy()
        logger.info("LineupFilter initialized")
    
    def remove_exact_duplicates(self, lineups: List[Dict]) -> List[Dict]:
        """
        Remove exact duplicate lineups (same players, same captain).
        
        Args:
            lineups: List of lineup dictionaries
        
        Returns:
            Deduplicated list of lineups
        """
        seen = set()
        unique_lineups = []
        duplicates_removed = 0
        
        for lineup in lineups:
            # Create hashable representation
            players_tuple = tuple(sorted(lineup['players']))
            captain = lineup['captain']
            lineup_hash = (players_tuple, captain)
            
            if lineup_hash not in seen:
                seen.add(lineup_hash)
                unique_lineups.append(lineup)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} exact duplicates")
        
        return unique_lineups
    
    def remove_similar_lineups(
        self,
        lineups: List[Dict],
        min_unique_players: int = 2
    ) -> List[Dict]:
        """
        Remove lineups that are too similar (differ by fewer than N players).
        
        Args:
            lineups: List of lineup dictionaries
            min_unique_players: Minimum number of different players required
        
        Returns:
            Filtered list of lineups
        """
        if min_unique_players <= 0:
            return lineups
        
        filtered_lineups = []
        removed_count = 0
        
        for candidate in lineups:
            candidate_players = set(candidate['players'])
            
            # Check against all filtered lineups
            is_unique_enough = True
            
            for existing in filtered_lineups:
                existing_players = set(existing['players'])
                
                # Count different players
                different_players = len(candidate_players ^ existing_players)
                
                if different_players < min_unique_players:
                    is_unique_enough = False
                    break
            
            if is_unique_enough:
                filtered_lineups.append(candidate)
            else:
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} similar lineups (min_unique={min_unique_players})")
        
        return filtered_lineups
    
    def filter_by_projection(
        self,
        lineups: List[Dict],
        min_projection: Optional[float] = None,
        max_projection: Optional[float] = None,
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Filter lineups by projection thresholds.
        
        Args:
            lineups: List of lineup dictionaries
            min_projection: Minimum total projection
            max_projection: Maximum total projection
            top_n: Keep only top N by projection
        
        Returns:
            Filtered lineups
        """
        filtered = lineups.copy()
        
        # Apply min threshold
        if min_projection is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['total_projection'] >= min_projection
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with projection >= {min_projection}")
        
        # Apply max threshold
        if max_projection is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['total_projection'] <= max_projection
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with projection <= {max_projection}")
        
        # Take top N
        if top_n is not None and len(filtered) > top_n:
            filtered.sort(key=lambda x: x['metrics']['total_projection'], reverse=True)
            filtered = filtered[:top_n]
            logger.info(f"Kept top {top_n} lineups by projection")
        
        return filtered
    
    def filter_by_ceiling(
        self,
        lineups: List[Dict],
        min_ceiling: Optional[float] = None,
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Filter lineups by ceiling.
        
        Args:
            lineups: List of lineup dictionaries
            min_ceiling: Minimum total ceiling
            top_n: Keep only top N by ceiling
        
        Returns:
            Filtered lineups
        """
        filtered = lineups.copy()
        
        # Apply min threshold
        if min_ceiling is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['total_ceiling'] >= min_ceiling
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with ceiling >= {min_ceiling}")
        
        # Take top N
        if top_n is not None and len(filtered) > top_n:
            filtered.sort(key=lambda x: x['metrics']['total_ceiling'], reverse=True)
            filtered = filtered[:top_n]
            logger.info(f"Kept top {top_n} lineups by ceiling")
        
        return filtered
    
    def filter_by_ownership(
        self,
        lineups: List[Dict],
        min_ownership: Optional[float] = None,
        max_ownership: Optional[float] = None
    ) -> List[Dict]:
        """
        Filter lineups by total ownership.
        
        Args:
            lineups: List of lineup dictionaries
            min_ownership: Minimum total ownership
            max_ownership: Maximum total ownership
        
        Returns:
            Filtered lineups
        """
        filtered = lineups.copy()
        
        # Apply min threshold
        if min_ownership is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['total_ownership'] >= min_ownership
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with ownership >= {min_ownership}")
        
        # Apply max threshold
        if max_ownership is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['total_ownership'] <= max_ownership
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with ownership <= {max_ownership}")
        
        return filtered
    
    def filter_by_correlation(
        self,
        lineups: List[Dict],
        min_correlation: Optional[float] = None,
        max_correlation: Optional[float] = None
    ) -> List[Dict]:
        """
        Filter lineups by correlation score.
        
        Args:
            lineups: List of lineup dictionaries
            min_correlation: Minimum correlation
            max_correlation: Maximum correlation
        
        Returns:
            Filtered lineups
        """
        filtered = lineups.copy()
        
        # Apply min threshold
        if min_correlation is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['correlation'] >= min_correlation
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with correlation >= {min_correlation}")
        
        # Apply max threshold
        if max_correlation is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['correlation'] <= max_correlation
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with correlation <= {max_correlation}")
        
        return filtered
    
    def filter_by_stacks(
        self,
        lineups: List[Dict],
        require_qb_stack: bool = False,
        require_game_stack: bool = False,
        min_stacks: int = 0
    ) -> List[Dict]:
        """
        Filter lineups by stacking requirements.
        
        Args:
            lineups: List of lineup dictionaries
            require_qb_stack: Require QB stacks
            require_game_stack: Require game stacks
            min_stacks: Minimum total stacks
        
        Returns:
            Filtered lineups
        """
        filtered = []
        
        for lineup in lineups:
            stacking_metrics = lineup.get('stacking_metrics', {})
            
            # Check QB stack requirement
            if require_qb_stack and stacking_metrics.get('qb_stacks', 0) == 0:
                continue
            
            # Check game stack requirement
            if require_game_stack and stacking_metrics.get('game_stacks', 0) == 0:
                continue
            
            # Check minimum stacks
            total_stacks = stacking_metrics.get('qb_stacks', 0) + stacking_metrics.get('game_stacks', 0)
            if total_stacks < min_stacks:
                continue
            
            filtered.append(lineup)
        
        logger.info(f"Filtered to {len(filtered)} lineups meeting stack requirements")
        
        return filtered
    
    def filter_by_captain(
        self,
        lineups: List[Dict],
        allowed_captains: Optional[List[str]] = None,
        banned_captains: Optional[List[str]] = None,
        max_captain_exposure: Optional[float] = None
    ) -> List[Dict]:
        """
        Filter lineups by captain constraints.
        
        Args:
            lineups: List of lineup dictionaries
            allowed_captains: Only keep lineups with these captains
            banned_captains: Remove lineups with these captains
            max_captain_exposure: Maximum exposure % for any single captain
        
        Returns:
            Filtered lineups
        """
        filtered = lineups.copy()
        
        # Filter by allowed captains
        if allowed_captains is not None:
            filtered = [
                l for l in filtered
                if l['captain'] in allowed_captains
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with allowed captains")
        
        # Filter out banned captains
        if banned_captains is not None:
            filtered = [
                l for l in filtered
                if l['captain'] not in banned_captains
            ]
            logger.info(f"Filtered to {len(filtered)} lineups (removed banned captains)")
        
        # Enforce captain exposure limit
        if max_captain_exposure is not None and filtered:
            captain_counts = Counter(l['captain'] for l in filtered)
            max_count = int(np.ceil((max_captain_exposure / 100) * len(filtered)))
            
            # Keep lineups, respecting captain limits
            balanced_lineups = []
            current_captain_counts = Counter()
            
            for lineup in filtered:
                captain = lineup['captain']
                
                if current_captain_counts[captain] < max_count:
                    balanced_lineups.append(lineup)
                    current_captain_counts[captain] += 1
            
            filtered = balanced_lineups
            logger.info(f"Balanced to {len(filtered)} lineups with max captain exposure {max_captain_exposure}%")
        
        return filtered
    
    def filter_by_salary(
        self,
        lineups: List[Dict],
        min_salary: Optional[int] = None,
        max_salary: Optional[int] = None
    ) -> List[Dict]:
        """
        Filter lineups by total salary.
        
        Args:
            lineups: List of lineup dictionaries
            min_salary: Minimum total salary
            max_salary: Maximum total salary
        
        Returns:
            Filtered lineups
        """
        filtered = lineups.copy()
        
        # Apply min threshold
        if min_salary is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['total_salary'] >= min_salary
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with salary >= ${min_salary:,}")
        
        # Apply max threshold
        if max_salary is not None:
            filtered = [
                l for l in filtered
                if l['metrics']['total_salary'] <= max_salary
            ]
            logger.info(f"Filtered to {len(filtered)} lineups with salary <= ${max_salary:,}")
        
        return filtered
    
    def diversify_portfolio(
        self,
        lineups: List[Dict],
        target_size: int,
        diversity_weight: float = 0.5,
        quality_weight: float = 0.5
    ) -> List[Dict]:
        """
        Select most diverse subset of lineups while maintaining quality.
        
        Args:
            lineups: List of lineup dictionaries
            target_size: Desired portfolio size
            diversity_weight: Weight for diversity (0-1)
            quality_weight: Weight for quality/projection (0-1)
        
        Returns:
            Diversified portfolio
        """
        if len(lineups) <= target_size:
            return lineups
        
        logger.info(f"Diversifying portfolio from {len(lineups)} to {target_size} lineups")
        
        # Normalize weights
        total_weight = diversity_weight + quality_weight
        diversity_weight /= total_weight
        quality_weight /= total_weight
        
        # Start with highest projection lineup
        lineups_sorted = sorted(
            lineups,
            key=lambda x: x['metrics']['total_projection'],
            reverse=True
        )
        
        selected = [lineups_sorted[0]]
        remaining = lineups_sorted[1:]
        
        # Iteratively add most diverse + high quality lineups
        while len(selected) < target_size and remaining:
            best_score = -1
            best_lineup = None
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Calculate diversity score (avg unique players vs selected)
                diversity_scores = []
                candidate_players = set(candidate['players'])
                
                for existing in selected:
                    existing_players = set(existing['players'])
                    unique_players = len(candidate_players ^ existing_players)
                    diversity_scores.append(unique_players / 6.0)  # Normalize to 0-1
                
                avg_diversity = np.mean(diversity_scores)
                
                # Normalize quality (projection)
                max_proj = max(l['metrics']['total_projection'] for l in remaining)
                min_proj = min(l['metrics']['total_projection'] for l in remaining)
                
                if max_proj > min_proj:
                    quality_score = (candidate['metrics']['total_projection'] - min_proj) / (max_proj - min_proj)
                else:
                    quality_score = 1.0
                
                # Combined score
                combined_score = (
                    diversity_weight * avg_diversity +
                    quality_weight * quality_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_lineup = candidate
                    best_idx = idx
            
            if best_lineup:
                selected.append(best_lineup)
                remaining.pop(best_idx)
        
        logger.info(f"Diversified portfolio to {len(selected)} lineups")
        
        return selected
    
    def get_lineup_similarity_matrix(self, lineups: List[Dict]) -> pd.DataFrame:
        """
        Calculate similarity matrix for all lineup pairs.
        
        Args:
            lineups: List of lineup dictionaries
        
        Returns:
            DataFrame with pairwise similarity scores (0-100)
        """
        n = len(lineups)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 100.0
                else:
                    # Calculate similarity (% of shared players)
                    players_i = set(lineups[i]['players'])
                    players_j = set(lineups[j]['players'])
                    
                    shared = len(players_i & players_j)
                    similarity = (shared / 6.0) * 100
                    
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # Create DataFrame
        lineup_names = [f"Lineup {i+1}" for i in range(n)]
        df = pd.DataFrame(
            similarity_matrix,
            index=lineup_names,
            columns=lineup_names
        )
        
        return df
    
    def find_most_unique_lineups(
        self,
        lineups: List[Dict],
        n: int = 10
    ) -> List[Dict]:
        """
        Find N most unique lineups in portfolio.
        
        Args:
            lineups: List of lineup dictionaries
            n: Number of unique lineups to return
        
        Returns:
            List of most unique lineups
        """
        if len(lineups) <= n:
            return lineups
        
        # Calculate uniqueness score for each lineup
        uniqueness_scores = []
        
        for i, candidate in enumerate(lineups):
            candidate_players = set(candidate['players'])
            
            # Sum of differences from all other lineups
            total_uniqueness = 0
            
            for j, other in enumerate(lineups):
                if i == j:
                    continue
                
                other_players = set(other['players'])
                unique_count = len(candidate_players ^ other_players)
                total_uniqueness += unique_count
            
            avg_uniqueness = total_uniqueness / (len(lineups) - 1)
            uniqueness_scores.append((i, avg_uniqueness))
        
        # Sort by uniqueness
        uniqueness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        top_indices = [idx for idx, _ in uniqueness_scores[:n]]
        
        return [lineups[idx] for idx in top_indices]
    
    def batch_filter(
        self,
        lineups: List[Dict],
        filters: List[Dict]
    ) -> List[Dict]:
        """
        Apply multiple filters in sequence.
        
        Args:
            lineups: List of lineup dictionaries
            filters: List of filter configurations, e.g.:
                [
                    {'type': 'projection', 'min': 120.0},
                    {'type': 'correlation', 'min': 60.0},
                    {'type': 'ownership', 'max': 250.0}
                ]
        
        Returns:
            Filtered lineups
        """
        filtered = lineups.copy()
        
        for filter_config in filters:
            filter_type = filter_config.get('type')
            
            if filter_type == 'projection':
                filtered = self.filter_by_projection(
                    filtered,
                    min_projection=filter_config.get('min'),
                    max_projection=filter_config.get('max'),
                    top_n=filter_config.get('top_n')
                )
            
            elif filter_type == 'ceiling':
                filtered = self.filter_by_ceiling(
                    filtered,
                    min_ceiling=filter_config.get('min'),
                    top_n=filter_config.get('top_n')
                )
            
            elif filter_type == 'ownership':
                filtered = self.filter_by_ownership(
                    filtered,
                    min_ownership=filter_config.get('min'),
                    max_ownership=filter_config.get('max')
                )
            
            elif filter_type == 'correlation':
                filtered = self.filter_by_correlation(
                    filtered,
                    min_correlation=filter_config.get('min'),
                    max_correlation=filter_config.get('max')
                )
            
            elif filter_type == 'stacks':
                filtered = self.filter_by_stacks(
                    filtered,
                    require_qb_stack=filter_config.get('require_qb_stack', False),
                    require_game_stack=filter_config.get('require_game_stack', False),
                    min_stacks=filter_config.get('min_stacks', 0)
                )
            
            elif filter_type == 'captain':
                filtered = self.filter_by_captain(
                    filtered,
                    allowed_captains=filter_config.get('allowed_captains'),
                    banned_captains=filter_config.get('banned_captains'),
                    max_captain_exposure=filter_config.get('max_captain_exposure')
                )
            
            elif filter_type == 'similarity':
                filtered = self.remove_similar_lineups(
                    filtered,
                    min_unique_players=filter_config.get('min_unique_players', 2)
                )
            
            elif filter_type == 'duplicates':
                filtered = self.remove_exact_duplicates(filtered)
            
            if not filtered:
                logger.warning(f"All lineups filtered out at filter: {filter_type}")
                break
        
        logger.info(f"Batch filtering: {len(lineups)} → {len(filtered)} lineups")
        
        return filtered
