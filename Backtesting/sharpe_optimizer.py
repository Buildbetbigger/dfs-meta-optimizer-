"""
Sharpe Ratio Portfolio Optimizer - Phase 2
Risk-adjusted portfolio construction using Modern Portfolio Theory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class SharpeOptimizer:
    """
    Sharpe Ratio optimization for DFS portfolios
    
    Sharpe Ratio = (Expected Return - Risk-Free Rate) / Standard Deviation
    
    In DFS context:
    - Return = Projected score / Salary (value)
    - Risk = Volatility of player scores
    - Goal: Maximize risk-adjusted returns
    """
    
    def __init__(self, covariance_analyzer):
        """
        Initialize optimizer
        
        Args:
            covariance_analyzer: CovarianceAnalyzer instance with historical data
        """
        self.cov_analyzer = covariance_analyzer
        self.risk_free_rate = 0.0  # In DFS, no risk-free return
        
    def calculate_player_returns(
        self,
        players: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate expected returns (points per $1000 salary)
        
        Args:
            players: DataFrame with player_id, projected_score, salary
        
        Returns:
            Series of returns indexed by player_id
        """
        returns = players.set_index('player_id')['projected_score'] / (players.set_index('player_id')['salary'] / 1000)
        return returns
    
    def calculate_player_volatility(
        self,
        player_ids: List[str]
    ) -> pd.Series:
        """
        Calculate player score volatility (standard deviation)
        
        Args:
            player_ids: List of player IDs
        
        Returns:
            Series of volatilities indexed by player_id
        """
        if self.cov_analyzer.player_cov_matrix is None:
            self.cov_analyzer.calculate_player_covariance()
        
        volatilities = {}
        for pid in player_ids:
            try:
                var = self.cov_analyzer.player_cov_matrix.loc[pid, pid]
                volatilities[pid] = np.sqrt(var)
            except KeyError:
                volatilities[pid] = 0.0
        
        return pd.Series(volatilities)
    
    def calculate_portfolio_sharpe(
        self,
        player_ids: List[str],
        weights: np.ndarray,
        expected_returns: pd.Series
    ) -> float:
        """
        Calculate Sharpe ratio for a portfolio
        
        Args:
            player_ids: Players in portfolio
            weights: Player weights (must sum to 1)
            expected_returns: Expected return per player
        
        Returns:
            Sharpe ratio
        """
        # Portfolio return
        portfolio_return = np.dot(weights, expected_returns.loc[player_ids].values)
        
        # Portfolio risk (standard deviation)
        portfolio_variance = self.cov_analyzer.calculate_lineup_variance(
            player_ids, 
            weights.tolist()
        )
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        if portfolio_std > 0:
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
        else:
            sharpe = 0.0
        
        return sharpe
    
    def optimize_sharpe_ratio(
        self,
        player_pool: pd.DataFrame,
        num_players: int = 9,
        position_constraints: Dict[str, int] = None
    ) -> Dict:
        """
        Find optimal portfolio that maximizes Sharpe ratio
        
        Args:
            player_pool: DataFrame with player_id, position, salary, projected_score
            num_players: Number of players in lineup
            position_constraints: Dict like {'QB': 1, 'RB': 2, ...}
        
        Returns:
            Dictionary with optimal lineup and metrics
        """
        # Calculate expected returns
        expected_returns = self.calculate_player_returns(player_pool)
        
        # Get player IDs
        player_ids = player_pool['player_id'].tolist()
        
        # Initial guess: equal weights
        initial_weights = np.ones(len(player_ids)) / len(player_ids)
        
        # Objective: minimize negative Sharpe (to maximize Sharpe)
        def objective(weights):
            sharpe = self.calculate_portfolio_sharpe(
                player_ids,
                weights,
                expected_returns
            )
            return -sharpe  # Negative because we're minimizing
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(len(player_ids))]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        optimal_weights = result.x
        
        # Select top N players by weight
        player_weights = list(zip(player_ids, optimal_weights))
        player_weights.sort(key=lambda x: x[1], reverse=True)
        
        selected_players = [pw[0] for pw in player_weights[:num_players]]
        selected_weights = [pw[1] for pw in player_weights[:num_players]]
        
        # Renormalize weights
        selected_weights = np.array(selected_weights)
        selected_weights = selected_weights / selected_weights.sum()
        
        # Calculate final metrics
        final_sharpe = self.calculate_portfolio_sharpe(
            selected_players,
            selected_weights,
            expected_returns
        )
        
        portfolio_return = np.dot(
            selected_weights,
            expected_returns.loc[selected_players].values
        )
        
        portfolio_variance = self.cov_analyzer.calculate_lineup_variance(
            selected_players,
            selected_weights.tolist()
        )
        
        portfolio_std = np.sqrt(portfolio_variance)
        
        return {
            'players': selected_players,
            'weights': selected_weights.tolist(),
            'sharpe_ratio': round(final_sharpe, 3),
            'expected_return': round(portfolio_return, 3),
            'portfolio_std': round(portfolio_std, 3),
            'diversification_ratio': round(
                self.cov_analyzer.calculate_diversification_ratio(selected_players), 2
            )
        }
    
    def generate_efficient_frontier(
        self,
        player_pool: pd.DataFrame,
        num_points: int = 20
    ) -> pd.DataFrame:
        """
        Generate efficient frontier (risk vs return tradeoff)
        
        Args:
            player_pool: Player pool
            num_points: Number of points on frontier
        
        Returns:
            DataFrame with risk, return, and sharpe for each point
        """
        expected_returns = self.calculate_player_returns(player_pool)
        player_ids = player_pool['player_id'].tolist()
        
        frontier = []
        
        # Target returns from min to max
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_points)
        
        for target_return in target_returns:
            # Objective: minimize variance
            def objective(weights):
                return self.cov_analyzer.calculate_lineup_variance(
                    player_ids,
                    weights.tolist()
                )
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns.values) - target_return}
            ]
            
            bounds = [(0, 1) for _ in range(len(player_ids))]
            initial_weights = np.ones(len(player_ids)) / len(player_ids)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                variance = result.fun
                std = np.sqrt(variance)
                sharpe = (target_return - self.risk_free_rate) / std if std > 0 else 0
                
                frontier.append({
                    'expected_return': round(target_return, 3),
                    'std_deviation': round(std, 3),
                    'sharpe_ratio': round(sharpe, 3)
                })
        
        return pd.DataFrame(frontier)
    
    def compare_lineups_sharpe(
        self,
        lineups: List[Dict],
        player_pool: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare multiple lineups by Sharpe ratio
        
        Args:
            lineups: List of lineup dicts with 'players' key
            player_pool: Player pool with projections
        
        Returns:
            DataFrame ranking lineups by Sharpe
        """
        expected_returns = self.calculate_player_returns(player_pool)
        
        results = []
        for i, lineup in enumerate(lineups):
            player_ids = lineup['players']
            
            # Equal weights for comparison
            weights = np.ones(len(player_ids)) / len(player_ids)
            
            sharpe = self.calculate_portfolio_sharpe(
                player_ids,
                weights,
                expected_returns
            )
            
            portfolio_return = np.dot(
                weights,
                expected_returns.loc[player_ids].values
            )
            
            portfolio_std = np.sqrt(
                self.cov_analyzer.calculate_lineup_variance(player_ids)
            )
            
            results.append({
                'lineup_id': i + 1,
                'sharpe_ratio': round(sharpe, 3),
                'expected_return': round(portfolio_return, 3),
                'std_deviation': round(portfolio_std, 3),
                'return_to_risk': round(portfolio_return / portfolio_std, 3) if portfolio_std > 0 else 0
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    
    def calculate_information_ratio(
        self,
        player_ids: List[str],
        benchmark_return: float,
        expected_returns: pd.Series
    ) -> float:
        """
        Calculate Information Ratio (similar to Sharpe but vs benchmark)
        
        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        
        Args:
            player_ids: Players in lineup
            benchmark_return: Benchmark return (e.g., avg lineup)
            expected_returns: Expected returns per player
        
        Returns:
            Information ratio
        """
        weights = np.ones(len(player_ids)) / len(player_ids)
        portfolio_return = np.dot(weights, expected_returns.loc[player_ids].values)
        
        excess_return = portfolio_return - benchmark_return
        
        # Tracking error = portfolio std
        tracking_error = np.sqrt(
            self.cov_analyzer.calculate_lineup_variance(player_ids)
        )
        
        if tracking_error > 0:
            ir = excess_return / tracking_error
        else:
            ir = 0.0
        
        return ir

if __name__ == "__main__":
    # Test with mock data
    from mock_data_generator import MockDataGenerator
    from covariance_analyzer import CovarianceAnalyzer
    
    generator = MockDataGenerator()
    data = generator.generate_full_dataset()
    
    # Setup covariance analyzer
    cov_analyzer = CovarianceAnalyzer()
    cov_analyzer.load_historical_data(data['historical'])
    cov_analyzer.calculate_player_covariance()
    
    # Get current week player pool
    current_pool = data['historical'][data['historical']['week'] == 8].copy()
    
    # Sharpe optimizer
    sharpe_opt = SharpeOptimizer(cov_analyzer)
    
    print("=== Sharpe Ratio Optimization ===")
    result = sharpe_opt.optimize_sharpe_ratio(current_pool, num_players=9)
    
    print(f"Optimal Sharpe Ratio: {result['sharpe_ratio']}")
    print(f"Expected Return: {result['expected_return']}")
    print(f"Portfolio Std Dev: {result['portfolio_std']}")
    print(f"Diversification Ratio: {result['diversification_ratio']}")
    print(f"\nOptimal Players: {result['players']}")
    
    print("\n=== Efficient Frontier ===")
    frontier = sharpe_opt.generate_efficient_frontier(current_pool, num_points=10)
    print(frontier.head())
