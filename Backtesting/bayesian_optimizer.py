"""
Bayesian Optimization Module - Phase 2 Bonus
Automated hyperparameter tuning for optimization strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple, Optional
from scipy.stats import norm
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """
    Bayesian Optimization for DFS strategy tuning
    
    Uses Gaussian Process regression + Expected Improvement acquisition
    to find optimal hyperparameters (weights, thresholds, etc.)
    
    Example use cases:
    - Optimize ownership_weight, leverage_weight, etc.
    - Find optimal stack sizes
    - Tune correlation thresholds
    - Optimize Kelly fractions
    """
    
    def __init__(self, n_initial: int = 5, n_iterations: int = 20):
        """
        Initialize Bayesian optimizer
        
        Args:
            n_initial: Number of random initial samples
            n_iterations: Total optimization iterations
        """
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.X_observed = []  # Parameter configurations tried
        self.y_observed = []  # Observed returns/scores
        
    def gaussian_process_predict(
        self,
        X_new: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        noise: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gaussian Process prediction (simplified)
        
        Returns:
            (mean_predictions, std_predictions)
        """
        # RBF kernel
        def kernel(x1, x2, length_scale=1.0):
            dist = np.sum((x1 - x2) ** 2)
            return np.exp(-dist / (2 * length_scale ** 2))
        
        # Build kernel matrices
        K = np.zeros((len(X_train), len(X_train)))
        for i in range(len(X_train)):
            for j in range(len(X_train)):
                K[i, j] = kernel(X_train[i], X_train[j])
        
        K += noise * np.eye(len(X_train))  # Add noise
        
        # K_star: covariance between new points and training
        K_star = np.zeros((len(X_new), len(X_train)))
        for i in range(len(X_new)):
            for j in range(len(X_train)):
                K_star[i, j] = kernel(X_new[i], X_train[j])
        
        # K_star_star: covariance of new points
        K_star_star = np.array([[kernel(x, x) for x in X_new]])
        
        # GP prediction
        try:
            K_inv = np.linalg.inv(K)
            mean = K_star @ K_inv @ y_train
            cov = K_star_star - K_star @ K_inv @ K_star.T
            std = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            mean = np.mean(y_train) * np.ones(len(X_new))
            std = np.std(y_train) * np.ones(len(X_new))
        
        return mean, std
    
    def expected_improvement(
        self,
        X_new: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        Expected Improvement acquisition function
        
        Args:
            X_new: Candidate points
            X_train: Observed points
            y_train: Observed values
            xi: Exploration parameter
        
        Returns:
            EI values for each candidate
        """
        mean, std = self.gaussian_process_predict(X_new, X_train, y_train)
        
        y_best = np.max(y_train)
        
        # Expected improvement
        with np.errstate(divide='warn'):
            imp = mean - y_best - xi
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0
        
        return ei
    
    def propose_next_sample(
        self,
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Propose next hyperparameter configuration to try
        
        Args:
            bounds: List of (min, max) for each parameter
        
        Returns:
            Next parameter configuration
        """
        if len(self.X_observed) < self.n_initial:
            # Random exploration
            return np.array([
                np.random.uniform(low, high) 
                for low, high in bounds
            ])
        
        # Use Expected Improvement
        X_train = np.array(self.X_observed)
        y_train = np.array(self.y_observed)
        
        # Generate candidate points
        n_candidates = 1000
        X_candidates = np.array([
            [np.random.uniform(low, high) for low, high in bounds]
            for _ in range(n_candidates)
        ])
        
        # Calculate EI
        ei_values = self.expected_improvement(X_candidates, X_train, y_train)
        
        # Return best candidate
        best_idx = np.argmax(ei_values)
        return X_candidates[best_idx]
    
    def optimize(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        param_names: List[str]
    ) -> Dict:
        """
        Run Bayesian optimization
        
        Args:
            objective_func: Function to maximize (returns score/ROI)
            bounds: Parameter bounds
            param_names: Names of parameters
        
        Returns:
            Dict with best parameters and optimization history
        """
        logger.info(f"Starting Bayesian optimization: {self.n_iterations} iterations")
        
        history = []
        
        for iteration in range(self.n_iterations):
            # Propose next sample
            X_next = self.propose_next_sample(bounds)
            
            # Evaluate objective
            y_next = objective_func(X_next)
            
            # Store observation
            self.X_observed.append(X_next)
            self.y_observed.append(y_next)
            
            # Track history
            params = {name: float(val) for name, val in zip(param_names, X_next)}
            history.append({
                'iteration': iteration + 1,
                'parameters': params,
                'score': float(y_next)
            })
            
            logger.info(f"Iteration {iteration+1}/{self.n_iterations}: score={y_next:.3f}")
        
        # Find best
        best_idx = np.argmax(self.y_observed)
        best_params = {
            name: float(val) 
            for name, val in zip(param_names, self.X_observed[best_idx])
        }
        best_score = float(self.y_observed[best_idx])
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'history': history,
            'num_iterations': self.n_iterations
        }

class StrategyTuner:
    """
    Automated strategy tuning using Bayesian optimization
    """
    
    def __init__(self, historical_results: pd.DataFrame):
        """
        Initialize tuner with historical results
        
        Args:
            historical_results: DataFrame with lineup results (score, ROI, etc.)
        """
        self.results = historical_results
        
    def tune_weight_parameters(
        self,
        lineup_generator: Callable,
        validation_weeks: List[int]
    ) -> Dict:
        """
        Tune optimization weights (projection, ownership, leverage, etc.)
        
        Args:
            lineup_generator: Function that takes weights dict and generates lineups
            validation_weeks: Weeks to validate on
        
        Returns:
            Optimal weight configuration
        """
        # Define parameters to tune
        param_names = [
            'projection_weight',
            'ownership_weight', 
            'leverage_weight',
            'correlation_weight'
        ]
        
        bounds = [
            (0.0, 2.0),  # projection_weight
            (0.0, 1.0),  # ownership_weight
            (0.0, 1.0),  # leverage_weight
            (0.0, 1.0)   # correlation_weight
        ]
        
        # Objective: average ROI on validation weeks
        def objective(params):
            weights = {name: val for name, val in zip(param_names, params)}
            
            rois = []
            for week in validation_weeks:
                lineups = lineup_generator(weights, week)
                # Simulate results (mock)
                week_roi = np.random.normal(10, 20)  # Placeholder
                rois.append(week_roi)
            
            return np.mean(rois)
        
        # Run optimization
        optimizer = BayesianOptimizer(n_initial=5, n_iterations=20)
        result = optimizer.optimize(objective, bounds, param_names)
        
        return result
    
    def tune_kelly_fraction(
        self,
        bankroll: float,
        contest_results: pd.DataFrame
    ) -> float:
        """
        Find optimal Kelly fraction based on historical results
        
        Args:
            bankroll: Starting bankroll
            contest_results: Historical contest outcomes
        
        Returns:
            Optimal Kelly fraction
        """
        param_names = ['kelly_fraction']
        bounds = [(0.05, 1.0)]
        
        def objective(params):
            kelly_frac = params[0]
            
            # Simulate bankroll growth
            br = bankroll
            for _, result in contest_results.iterrows():
                entry_fee = result['entry_fee']
                roi = result['roi'] / 100
                
                # Bet size
                bet = br * kelly_frac
                bet = max(bet, entry_fee)
                
                # Outcome
                br += bet * roi
                
                if br <= 0:
                    return -1e6  # Bust penalty
            
            # Return final bankroll
            return br
        
        optimizer = BayesianOptimizer(n_initial=5, n_iterations=15)
        result = optimizer.optimize(objective, bounds, param_names)
        
        return result['best_parameters']['kelly_fraction']
    
    def tune_stack_size(
        self,
        stack_range: Tuple[int, int] = (2, 7)
    ) -> int:
        """
        Find optimal stack size for contest type
        
        Args:
            stack_range: (min_stack, max_stack)
        
        Returns:
            Optimal stack size
        """
        param_names = ['stack_size']
        bounds = [(stack_range[0], stack_range[1])]
        
        def objective(params):
            stack_size = int(params[0])
            
            # Score based on historical performance
            # Placeholder: in reality, would test lineups with this stack size
            score = np.random.normal(0, 1)  # Mock
            
            return score
        
        optimizer = BayesianOptimizer(n_initial=3, n_iterations=10)
        result = optimizer.optimize(objective, bounds, param_names)
        
        return int(result['best_parameters']['stack_size'])

if __name__ == "__main__":
    # Test Bayesian optimizer
    print("=== Bayesian Optimization Test ===")
    
    # Simple test function: maximize negative parabola
    def test_objective(x):
        return -(x[0] - 3) ** 2 - (x[1] + 2) ** 2 + 10
    
    optimizer = BayesianOptimizer(n_initial=3, n_iterations=10)
    result = optimizer.optimize(
        test_objective,
        bounds=[(-5, 5), (-5, 5)],
        param_names=['x', 'y']
    )
    
    print(f"\nBest parameters: {result['best_parameters']}")
    print(f"Best score: {result['best_score']:.3f}")
    print(f"True optimum: x=3, y=-2, score=10")
