"""
Module 5: Lineup Evaluator
Evaluates and ranks lineups across multiple dimensions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LineupEvaluator:
    """
    Evaluates lineups across multiple scoring dimensions.
    
    Features:
    - Multi-dimensional scoring
    - Lineup ranking
    - Best lineup selection
    - Risk/reward profiling
    - Submission recommendations
    """
    
    def __init__(self):
        """Initialize lineup evaluator"""
        logger.info("LineupEvaluator initialized")
    
    def _calculate_projection_score(self, lineup: pd.DataFrame) -> float:
        """Calculate total projection score"""
        return lineup['projection'].sum()
    
    def _calculate_ceiling_score(self, lineup: pd.DataFrame) -> float:
        """Calculate total ceiling score"""
        if 'ceiling' in lineup.columns:
            return lineup['ceiling'].sum()
        else:
            # Estimate ceiling as projection * 1.4
            return lineup['projection'].sum() * 1.4
    
    def _calculate_floor_score(self, lineup: pd.DataFrame) -> float:
        """Calculate total floor score"""
        if 'floor' in lineup.columns:
            return lineup['floor'].sum()
        else:
            # Estimate floor as projection * 0.6
            return lineup['projection'].sum() * 0.6
    
    def _calculate_leverage_score(self, lineup: pd.DataFrame) -> float:
        """
        Calculate leverage score.
        
        Leverage = Sum(ceiling / ownership) for all players
        """
        if 'ceiling' not in lineup.columns:
            lineup = lineup.copy()
            lineup['ceiling'] = lineup['projection'] * 1.4
        
        if 'ownership' not in lineup.columns:
            # Can't calculate leverage without ownership
            return 0.0
        
        leverage_scores = lineup['ceiling'] / (lineup['ownership'] + 0.1)
        return leverage_scores.sum()
    
    def _calculate_correlation_score(self, lineup: pd.DataFrame) -> float:
        """
        Calculate correlation score.
        
        Higher score = more correlation (stacks)
        """
        score = 0.0
        
        # QB + pass-catcher correlation
        qbs = lineup[lineup['position'] == 'QB']
        pass_catchers = lineup[lineup['position'].isin(['WR', 'TE'])]
        
        for _, qb in qbs.iterrows():
            qb_team = qb.get('team', '')
            same_team_catchers = pass_catchers[pass_catchers['team'] == qb_team]
            
            # Award points for each same-team pass catcher
            score += len(same_team_catchers) * 20
        
        # RB + DST correlation (rare but powerful)
        rbs = lineup[lineup['position'] == 'RB']
        dsts = lineup[lineup['position'] == 'DST']
        
        for _, rb in rbs.iterrows():
            rb_team = rb.get('team', '')
            same_team_dst = dsts[dsts['team'] == rb_team]
            score += len(same_team_dst) * 15
        
        # Game stacks (multiple players from same game)
        if 'opponent' in lineup.columns:
            for _, player in lineup.iterrows():
                team = player.get('team', '')
                opp = player.get('opponent', '')
                
                # Count players from this game
                game_players = lineup[
                    (lineup['team'] == team) | 
                    (lineup['team'] == opp) |
                    (lineup['opponent'] == team) |
                    (lineup['opponent'] == opp)
                ]
                
                if len(game_players) >= 4:
                    score += 10  # Bonus for game stacks
        
        return score
    
    def _calculate_uniqueness_score(self, lineup: pd.DataFrame) -> float:
        """
        Calculate uniqueness score.
        
        Lower total ownership = more unique lineup
        """
        if 'ownership' not in lineup.columns:
            return 50.0  # Default mid-range
        
        avg_ownership = lineup['ownership'].mean()
        
        # Invert so lower ownership = higher score
        uniqueness = 100 - avg_ownership
        
        return max(0, uniqueness)
    
    def _calculate_salary_efficiency(self, lineup: pd.DataFrame, salary_cap: int = 50000) -> float:
        """
        Calculate salary efficiency.
        
        Score = (Total projection / Total salary) * 1000
        """
        total_salary = lineup['salary'].sum()
        total_projection = lineup['projection'].sum()
        
        if total_salary == 0:
            return 0.0
        
        efficiency = (total_projection / total_salary) * 1000
        
        # Bonus for using close to full cap
        cap_usage = total_salary / salary_cap
        if cap_usage >= 0.98:
            efficiency *= 1.05  # 5% bonus for near-full usage
        
        return efficiency
    
    def _calculate_variance_score(self, lineup: pd.DataFrame) -> float:
        """
        Calculate variance/boom-or-bust score.
        
        Higher score = more variance = boom-or-bust
        """
        if 'ceiling' in lineup.columns and 'floor' in lineup.columns:
            ceiling_sum = lineup['ceiling'].sum()
            floor_sum = lineup['floor'].sum()
            
            variance = ceiling_sum - floor_sum
        else:
            # Estimate from projections
            variance = lineup['projection'].std() * len(lineup)
        
        return variance
    
    def evaluate_lineup(
        self,
        lineup: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Evaluate a single lineup across all dimensions.
        
        Args:
            lineup: Lineup DataFrame
            weights: Optional custom weights for scoring dimensions
        
        Returns:
            Dictionary with all scores
        """
        # Default weights
        if weights is None:
            weights = {
                'projection': 1.0,
                'ceiling': 1.0,
                'leverage': 1.0,
                'correlation': 1.0,
                'uniqueness': 0.5,
                'efficiency': 0.5
            }
        
        scores = {
            'projection': self._calculate_projection_score(lineup),
            'ceiling': self._calculate_ceiling_score(lineup),
            'floor': self._calculate_floor_score(lineup),
            'leverage': self._calculate_leverage_score(lineup),
            'correlation': self._calculate_correlation_score(lineup),
            'uniqueness': self._calculate_uniqueness_score(lineup),
            'efficiency': self._calculate_salary_efficiency(lineup),
            'variance': self._calculate_variance_score(lineup)
        }
        
        # Calculate composite score
        composite = sum(
            scores.get(dim, 0) * weights.get(dim, 0)
            for dim in weights.keys()
        )
        
        scores['composite'] = composite
        
        return scores
    
    def evaluate_portfolio(
        self,
        lineups: List[pd.DataFrame],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Evaluate all lineups in portfolio.
        
        Args:
            lineups: List of lineup DataFrames
            weights: Optional custom weights
        
        Returns:
            DataFrame with rankings
        """
        results = []
        
        for idx, lineup in enumerate(lineups):
            scores = self.evaluate_lineup(lineup, weights)
            
            result = {
                'lineup_id': idx,
                **scores
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Add percentile ranks
        for col in ['projection', 'ceiling', 'leverage', 'correlation', 'composite']:
            if col in df.columns:
                df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
        
        logger.info(f"Evaluated {len(lineups)} lineups")
        
        return df
    
    def get_best_lineups(
        self,
        evaluation_df: pd.DataFrame,
        criterion: str = 'composite',
        n: int = 20
    ) -> List[int]:
        """
        Get best N lineups by criterion.
        
        Args:
            evaluation_df: Results from evaluate_portfolio
            criterion: Which metric to use ('composite', 'ceiling', 'leverage', etc.)
            n: Number of lineups to return
        
        Returns:
            List of lineup IDs
        """
        if criterion not in evaluation_df.columns:
            logger.warning(f"Criterion '{criterion}' not found, using 'composite'")
            criterion = 'composite'
        
        top_lineups = evaluation_df.nlargest(n, criterion)
        
        return top_lineups['lineup_id'].tolist()
    
    def get_balanced_selection(
        self,
        evaluation_df: pd.DataFrame,
        n: int = 20
    ) -> List[int]:
        """
        Get balanced selection of lineups.
        
        Mix of:
        - High ceiling (tournament upside)
        - High leverage (ownership edge)
        - High correlation (stacking power)
        - Safe plays (high floor)
        
        Args:
            evaluation_df: Results from evaluate_portfolio
            n: Number of lineups to select
        
        Returns:
            List of lineup IDs
        """
        selected = []
        
        # 40% highest ceiling (boom potential)
        ceiling_count = int(n * 0.4)
        ceiling_lineups = evaluation_df.nlargest(ceiling_count, 'ceiling')['lineup_id'].tolist()
        selected.extend(ceiling_lineups)
        
        # 30% highest leverage (ownership edge)
        leverage_count = int(n * 0.3)
        leverage_lineups = evaluation_df[
            ~evaluation_df['lineup_id'].isin(selected)
        ].nlargest(leverage_count, 'leverage')['lineup_id'].tolist()
        selected.extend(leverage_lineups)
        
        # 20% highest correlation (stack power)
        corr_count = int(n * 0.2)
        corr_lineups = evaluation_df[
            ~evaluation_df['lineup_id'].isin(selected)
        ].nlargest(corr_count, 'correlation')['lineup_id'].tolist()
        selected.extend(corr_lineups)
        
        # 10% safest (high floor)
        safe_count = n - len(selected)
        safe_lineups = evaluation_df[
            ~evaluation_df['lineup_id'].isin(selected)
        ].nlargest(safe_count, 'floor')['lineup_id'].tolist()
        selected.extend(safe_lineups)
        
        logger.info(f"Balanced selection: {ceiling_count} ceiling, {leverage_count} leverage, "
                   f"{corr_count} correlation, {safe_count} safe")
        
        return selected[:n]
    
    def categorize_lineups(
        self,
        evaluation_df: pd.DataFrame
    ) -> Dict[str, List[int]]:
        """
        Categorize lineups by risk profile.
        
        Returns:
            Dictionary with categories:
            - high_ceiling: Tournament boom potential
            - high_leverage: Ownership edge
            - high_correlation: Stacking power
            - safe: High floor plays
            - balanced: Mix of metrics
        """
        categories = {
            'high_ceiling': [],
            'high_leverage': [],
            'high_correlation': [],
            'safe': [],
            'balanced': []
        }
        
        # Use percentiles to categorize
        for _, row in evaluation_df.iterrows():
            lineup_id = row['lineup_id']
            
            # High ceiling: top 25% in ceiling
            if row.get('ceiling_percentile', 0) >= 75:
                categories['high_ceiling'].append(lineup_id)
            
            # High leverage: top 25% in leverage
            if row.get('leverage_percentile', 0) >= 75:
                categories['high_leverage'].append(lineup_id)
            
            # High correlation: top 25% in correlation
            if row.get('correlation_percentile', 0) >= 75:
                categories['high_correlation'].append(lineup_id)
            
            # Safe: top 25% in floor
            if row['floor'] >= evaluation_df['floor'].quantile(0.75):
                categories['safe'].append(lineup_id)
            
            # Balanced: middle 50% in most metrics
            ceiling_pct = row.get('ceiling_percentile', 50)
            leverage_pct = row.get('leverage_percentile', 50)
            
            if 25 <= ceiling_pct <= 75 and 25 <= leverage_pct <= 75:
                categories['balanced'].append(lineup_id)
        
        return categories
    
    def get_submission_recommendations(
        self,
        evaluation_df: pd.DataFrame,
        contest_type: str = 'GPP',
        n: int = 20
    ) -> Dict:
        """
        Get recommendations for which lineups to submit.
        
        Args:
            evaluation_df: Results from evaluate_portfolio
            contest_type: Contest type ('GPP', 'CASH', 'SATELLITE')
            n: Number of lineups to submit
        
        Returns:
            Dictionary with recommendations
        """
        if contest_type == 'GPP':
            # GPP: prioritize ceiling and leverage
            selected = self.get_balanced_selection(evaluation_df, n)
            strategy = "Balanced mix: 40% high ceiling, 30% leverage, 20% correlation, 10% safe"
        
        elif contest_type == 'CASH':
            # Cash: prioritize floor and consistency
            selected = evaluation_df.nlargest(n, 'floor')['lineup_id'].tolist()
            strategy = "Safe plays: highest floor lineups selected"
        
        elif contest_type == 'SATELLITE':
            # Satellite: similar to cash but allow some upside
            # 70% safe, 30% upside
            safe_count = int(n * 0.7)
            upside_count = n - safe_count
            
            safe = evaluation_df.nlargest(safe_count, 'floor')['lineup_id'].tolist()
            upside = evaluation_df[
                ~evaluation_df['lineup_id'].isin(safe)
            ].nlargest(upside_count, 'ceiling')['lineup_id'].tolist()
            
            selected = safe + upside
            strategy = f"Mixed: {safe_count} safe, {upside_count} upside"
        
        else:
            # Default: composite score
            selected = self.get_best_lineups(evaluation_df, 'composite', n)
            strategy = "Composite score: best overall lineups"
        
        # Get stats on selected lineups
        selected_df = evaluation_df[evaluation_df['lineup_id'].isin(selected)]
        
        return {
            'lineup_ids': selected,
            'strategy': strategy,
            'count': len(selected),
            'avg_projection': selected_df['projection'].mean(),
            'avg_ceiling': selected_df['ceiling'].mean(),
            'avg_floor': selected_df['floor'].mean(),
            'avg_leverage': selected_df['leverage'].mean(),
            'total_salary': None  # Would need lineup data to calculate
        }
