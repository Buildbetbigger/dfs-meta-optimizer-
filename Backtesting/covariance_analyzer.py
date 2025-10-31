"""
Covariance Analyzer - Phase 2
PhD-level correlation and covariance analysis for DFS portfolios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from scipy.cluster import hierarchy
import logging

logger = logging.getLogger(__name__)

class CovarianceAnalyzer:
    """
    Advanced covariance and correlation analysis
    - Player score correlations
    - Position-level correlations
    - Game stack correlations
    - Portfolio covariance matrices
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.player_cov_matrix = None
        self.player_corr_matrix = None
        self.historical_data = None
        
    def load_historical_data(self, historical_df: pd.DataFrame):
        """Load historical player performance data"""
        self.historical_data = historical_df
        logger.info(f"Loaded {len(historical_df)} historical records")
        
    def calculate_player_covariance(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate player-level covariance and correlation matrices
        
        Returns:
            (covariance_matrix, correlation_matrix)
        """
        if self.historical_data is None:
            raise ValueError("No historical data loaded")
        
        # Pivot data: rows=weeks, columns=players, values=actual_score
        score_matrix = self.historical_data.pivot_table(
            index='week',
            columns='player_id',
            values='actual_score',
            aggfunc='mean'
        )
        
        # Calculate covariance and correlation
        self.player_cov_matrix = score_matrix.cov()
        self.player_corr_matrix = score_matrix.corr()
        
        logger.info(f"Calculated {len(self.player_cov_matrix)} x {len(self.player_cov_matrix)} covariance matrix")
        
        return self.player_cov_matrix, self.player_corr_matrix
    
    def get_position_correlations(self) -> pd.DataFrame:
        """Calculate average correlations between positions"""
        if self.player_corr_matrix is None:
            self.calculate_player_covariance()
        
        # Map player_ids to positions
        player_positions = self.historical_data[['player_id', 'position']].drop_duplicates()
        position_map = dict(zip(player_positions['player_id'], player_positions['position']))
        
        # Calculate position-level correlations
        positions = ['QB', 'RB', 'WR', 'TE', 'DST']
        pos_corr = pd.DataFrame(index=positions, columns=positions, dtype=float)
        
        for pos1 in positions:
            players1 = [p for p, pos in position_map.items() if pos == pos1]
            for pos2 in positions:
                players2 = [p for p, pos in position_map.items() if pos == pos2]
                
                # Get sub-matrix
                try:
                    sub_matrix = self.player_corr_matrix.loc[players1, players2]
                    avg_corr = sub_matrix.values.mean()
                    pos_corr.loc[pos1, pos2] = avg_corr
                except:
                    pos_corr.loc[pos1, pos2] = 0.0
        
        return pos_corr.astype(float)
    
    def calculate_lineup_variance(
        self, 
        player_ids: List[str],
        weights: List[float] = None
    ) -> float:
        """
        Calculate portfolio variance using covariance matrix
        
        Var(Portfolio) = w^T * Σ * w
        where w = weights, Σ = covariance matrix
        
        Args:
            player_ids: List of player IDs in lineup
            weights: Optional weights (default: equal weight)
        
        Returns:
            Portfolio variance
        """
        if self.player_cov_matrix is None:
            self.calculate_player_covariance()
        
        # Default to equal weights
        if weights is None:
            weights = np.ones(len(player_ids)) / len(player_ids)
        
        weights = np.array(weights)
        
        # Get relevant sub-matrix
        try:
            cov_sub = self.player_cov_matrix.loc[player_ids, player_ids]
            portfolio_var = np.dot(weights, np.dot(cov_sub.values, weights))
            return float(portfolio_var)
        except KeyError:
            # Some players not in covariance matrix
            logger.warning("Some players missing from covariance matrix")
            return 0.0
    
    def calculate_lineup_correlation(
        self,
        player_ids: List[str]
    ) -> float:
        """
        Calculate average pairwise correlation in lineup
        
        Returns:
            Average correlation coefficient
        """
        if self.player_corr_matrix is None:
            self.calculate_player_covariance()
        
        try:
            corr_sub = self.player_corr_matrix.loc[player_ids, player_ids]
            
            # Get upper triangle (exclude diagonal)
            upper_triangle = np.triu(corr_sub.values, k=1)
            correlations = upper_triangle[upper_triangle != 0]
            
            if len(correlations) > 0:
                return float(correlations.mean())
            else:
                return 0.0
        except KeyError:
            logger.warning("Some players missing from correlation matrix")
            return 0.0
    
    def find_correlated_groups(
        self,
        min_correlation: float = 0.5,
        min_group_size: int = 2
    ) -> List[List[str]]:
        """
        Identify groups of highly correlated players (stacks)
        
        Uses hierarchical clustering on correlation matrix
        
        Args:
            min_correlation: Minimum correlation for grouping
            min_group_size: Minimum players in a group
        
        Returns:
            List of player groups
        """
        if self.player_corr_matrix is None:
            self.calculate_player_covariance()
        
        # Convert correlation to distance
        distance_matrix = 1 - self.player_corr_matrix.abs()
        
        # Hierarchical clustering
        try:
            linkage = hierarchy.linkage(distance_matrix, method='average')
            clusters = hierarchy.fcluster(
                linkage, 
                t=1-min_correlation, 
                criterion='distance'
            )
            
            # Group players by cluster
            player_ids = self.player_corr_matrix.index.tolist()
            groups = {}
            for player_id, cluster_id in zip(player_ids, clusters):
                if cluster_id not in groups:
                    groups[cluster_id] = []
                groups[cluster_id].append(player_id)
            
            # Filter by size
            correlated_groups = [
                group for group in groups.values() 
                if len(group) >= min_group_size
            ]
            
            return correlated_groups
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return []
    
    def calculate_diversification_ratio(
        self,
        player_ids: List[str]
    ) -> float:
        """
        Calculate diversification ratio
        
        DR = (Sum of individual volatilities) / (Portfolio volatility)
        DR > 1 indicates diversification benefit
        
        Args:
            player_ids: Players in lineup
        
        Returns:
            Diversification ratio
        """
        if self.player_cov_matrix is None:
            self.calculate_player_covariance()
        
        try:
            # Individual standard deviations
            individual_stds = [
                np.sqrt(self.player_cov_matrix.loc[pid, pid]) 
                for pid in player_ids
            ]
            sum_individual_stds = sum(individual_stds)
            
            # Portfolio standard deviation
            portfolio_var = self.calculate_lineup_variance(player_ids)
            portfolio_std = np.sqrt(portfolio_var)
            
            if portfolio_std > 0:
                dr = sum_individual_stds / (portfolio_std * len(player_ids))
                return float(dr)
            else:
                return 1.0
        except Exception as e:
            logger.error(f"DR calculation failed: {e}")
            return 1.0
    
    def get_optimal_correlation_target(
        self,
        contest_type: str = 'GPP'
    ) -> float:
        """
        Return target correlation for contest type
        
        Args:
            contest_type: 'CASH' or 'GPP'
        
        Returns:
            Target correlation coefficient
        """
        if contest_type == 'CASH':
            # Cash games: want low correlation (diversification)
            return 0.2
        elif contest_type == 'GPP':
            # GPPs: want moderate correlation (stacking)
            return 0.4
        else:
            return 0.3
    
    def score_lineup_correlation_fit(
        self,
        player_ids: List[str],
        contest_type: str = 'GPP'
    ) -> Dict:
        """
        Score how well lineup correlation fits contest type
        
        Returns:
            Dictionary with correlation metrics and fit score
        """
        actual_corr = self.calculate_lineup_correlation(player_ids)
        target_corr = self.get_optimal_correlation_target(contest_type)
        
        # Score based on distance from target
        distance = abs(actual_corr - target_corr)
        fit_score = max(0, 100 - (distance * 200))
        
        return {
            'actual_correlation': round(actual_corr, 3),
            'target_correlation': round(target_corr, 3),
            'distance': round(distance, 3),
            'fit_score': round(fit_score, 1),
            'diversification_ratio': round(
                self.calculate_diversification_ratio(player_ids), 2
            )
        }

if __name__ == "__main__":
    # Test with mock data
    from mock_data_generator import MockDataGenerator
    
    generator = MockDataGenerator()
    data = generator.generate_full_dataset()
    
    analyzer = CovarianceAnalyzer()
    analyzer.load_historical_data(data['historical'])
    
    # Calculate matrices
    cov, corr = analyzer.calculate_player_covariance()
    
    print("=== Covariance Matrix Shape ===")
    print(f"{cov.shape}")
    
    print("\n=== Position Correlations ===")
    print(analyzer.get_position_correlations())
    
    # Test lineup
    sample_players = data['historical']['player_id'].unique()[:9]
    
    print(f"\n=== Sample Lineup Analysis ===")
    lineup_corr = analyzer.calculate_lineup_correlation(sample_players.tolist())
    print(f"Lineup correlation: {lineup_corr:.3f}")
    
    fit = analyzer.score_lineup_correlation_fit(sample_players.tolist(), 'GPP')
    print(f"GPP fit score: {fit['fit_score']:.1f}")
