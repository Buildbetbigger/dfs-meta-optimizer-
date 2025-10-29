"""
DFS Meta-Optimizer v7.0.0 - Strategy Optimizer Module
Automated parameter tuning through simulation

Features:
- Single/multi-parameter optimization
- A/B testing framework
- Grid search
- Sensitivity analysis
- Contest-specific tuning
- Automated strategy improvement
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Callable
import logging
from itertools import product

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizes strategy parameters through simulation.
    
    NEW v7.0.0 Features:
    - Automated parameter tuning
    - Scientific A/B testing
    - Multi-parameter grid search
    - Contest-specific optimization
    - Self-improving intelligence
    """
    
    def __init__(self, optimizer_function: Callable, simulator_function: Callable):
        """
        Initialize strategy optimizer.
        
        Args:
            optimizer_function: Function that generates lineups
            simulator_function: Function that simulates contests
        """
        self.optimizer_function = optimizer_function
        self.simulator_function = simulator_function
        
        logger.info("StrategyOptimizer v7.0.0 initialized")
    
    def test_single_parameter(
        self,
        parameter_name: str,
        values: List[Any],
        base_config: Dict,
        num_simulations: int = 5000,
        objective: str = 'expected_roi'
    ) -> pd.DataFrame:
        """
        Test different values for a single parameter.
        
        Args:
            parameter_name: Name of parameter to test
            values: List of values to test
            base_config: Base configuration for other parameters
            num_simulations: Simulations per test
            objective: What to optimize ('expected_roi', 'win_rate', 'cash_rate')
        
        Returns:
            DataFrame with test results
        """
        logger.info(f"Testing parameter '{parameter_name}' with {len(values)} values")
        
        results = []
        
        for value in values:
            # Create config with this parameter value
            config = base_config.copy()
            config[parameter_name] = value
            
            try:
                # Generate lineups with this config
                lineups = self.optimizer_function(**config)
                
                # Simulate performance
                sim_result = self.simulator_function(lineups, num_simulations=num_simulations)
                
                results.append({
                    parameter_name: value,
                    'expected_roi': sim_result.get('expected_roi', 0),
                    'win_rate': sim_result.get('win_rate', 0),
                    'cash_rate': sim_result.get('cash_rate', 0),
                    'top_10_rate': sim_result.get('top_10_rate', 0),
                    'avg_score': sim_result.get('avg_score', 0)
                })
                
                logger.info(f"{parameter_name}={value}: ROI={sim_result.get('expected_roi', 0):.1f}%")
            
            except Exception as e:
                logger.error(f"Error testing {parameter_name}={value}: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Find optimal value
            optimal_idx = df[objective].idxmax()
            optimal_value = df.loc[optimal_idx, parameter_name]
            optimal_score = df.loc[optimal_idx, objective]
            
            logger.info(f"Optimal {parameter_name}: {optimal_value} ({objective}={optimal_score:.2f})")
        
        return df
    
    def test_multiple_parameters(
        self,
        parameters: Dict[str, List[Any]],
        base_config: Dict,
        num_simulations: int = 3000,
        objective: str = 'expected_roi',
        max_combinations: int = 50
    ) -> pd.DataFrame:
        """
        Test multiple parameters via grid search.
        
        NEW v7.0.0: Find optimal parameter combinations!
        
        Args:
            parameters: Dict of {parameter_name: [values_to_test]}
            base_config: Base configuration
            num_simulations: Simulations per combination
            objective: What to optimize
            max_combinations: Maximum combinations to test
        
        Returns:
            DataFrame with all test results
        """
        # Generate all combinations
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        combinations = list(product(*param_values))
        
        # Limit combinations if too many
        if len(combinations) > max_combinations:
            logger.warning(f"Too many combinations ({len(combinations)}), "
                          f"sampling {max_combinations}")
            import random
            combinations = random.sample(combinations, max_combinations)
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        
        for combo in combinations:
            # Create config with these parameter values
            config = base_config.copy()
            for param_name, value in zip(param_names, combo):
                config[param_name] = value
            
            try:
                # Generate lineups
                lineups = self.optimizer_function(**config)
                
                # Simulate
                sim_result = self.simulator_function(lineups, num_simulations=num_simulations)
                
                result = {
                    **{name: val for name, val in zip(param_names, combo)},
                    'expected_roi': sim_result.get('expected_roi', 0),
                    'win_rate': sim_result.get('win_rate', 0),
                    'cash_rate': sim_result.get('cash_rate', 0),
                    'top_10_rate': sim_result.get('top_10_rate', 0)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error testing combination {combo}: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # Find optimal combination
        if not df.empty:
            optimal_idx = df[objective].idxmax()
            optimal_combo = df.loc[optimal_idx]
            
            logger.info(f"Optimal configuration found:")
            for param_name in param_names:
                logger.info(f"  {param_name}: {optimal_combo[param_name]}")
            logger.info(f"  {objective}: {optimal_combo[objective]:.2f}")
        
        return df
    
    def ab_test(
        self,
        config_a: Dict,
        config_b: Dict,
        num_simulations: int = 10000,
        test_name_a: str = "Strategy A",
        test_name_b: str = "Strategy B"
    ) -> Dict:
        """
        A/B test two strategy configurations.
        
        NEW v7.0.0: Scientific strategy comparison!
        
        Args:
            config_a: First configuration
            config_b: Second configuration
            num_simulations: Number of simulations
            test_name_a: Name for first strategy
            test_name_b: Name for second strategy
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"A/B testing: {test_name_a} vs {test_name_b}")
        
        try:
            # Test Strategy A
            lineups_a = self.optimizer_function(**config_a)
            sim_a = self.simulator_function(lineups_a, num_simulations=num_simulations)
            
            # Test Strategy B
            lineups_b = self.optimizer_function(**config_b)
            sim_b = self.simulator_function(lineups_b, num_simulations=num_simulations)
            
            # Compare
            comparison = {
                'strategy_a': {
                    'name': test_name_a,
                    'config': config_a,
                    'expected_roi': sim_a.get('expected_roi', 0),
                    'win_rate': sim_a.get('win_rate', 0),
                    'cash_rate': sim_a.get('cash_rate', 0),
                    'top_10_rate': sim_a.get('top_10_rate', 0),
                    'avg_score': sim_a.get('avg_score', 0)
                },
                'strategy_b': {
                    'name': test_name_b,
                    'config': config_b,
                    'expected_roi': sim_b.get('expected_roi', 0),
                    'win_rate': sim_b.get('win_rate', 0),
                    'cash_rate': sim_b.get('cash_rate', 0),
                    'top_10_rate': sim_b.get('top_10_rate', 0),
                    'avg_score': sim_b.get('avg_score', 0)
                },
                'differences': {
                    'roi_diff': sim_b.get('expected_roi', 0) - sim_a.get('expected_roi', 0),
                    'win_rate_diff': sim_b.get('win_rate', 0) - sim_a.get('win_rate', 0),
                    'cash_rate_diff': sim_b.get('cash_rate', 0) - sim_a.get('cash_rate', 0)
                }
            }
            
            # Determine winner
            roi_a = sim_a.get('expected_roi', 0)
            roi_b = sim_b.get('expected_roi', 0)
            
            if roi_b > roi_a:
                comparison['winner'] = test_name_b
                comparison['winner_advantage'] = roi_b - roi_a
            else:
                comparison['winner'] = test_name_a
                comparison['winner_advantage'] = roi_a - roi_b
            
            logger.info(f"Winner: {comparison['winner']} "
                       f"(+{comparison['winner_advantage']:.1f}% ROI)")
            
            return comparison
        
        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            return {}
    
    def optimize_exposure(
        self,
        exposure_range: List[float],
        base_config: Dict,
        num_simulations: int = 5000
    ) -> Dict:
        """
        Find optimal max player exposure.
        
        Args:
            exposure_range: List of exposure values to test (e.g., [20, 25, 30, 35])
            base_config: Base configuration
            num_simulations: Simulations per test
        
        Returns:
            Dictionary with optimal exposure and results
        """
        results = self.test_single_parameter(
            parameter_name='max_player_exposure',
            values=exposure_range,
            base_config=base_config,
            num_simulations=num_simulations,
            objective='expected_roi'
        )
        
        if results.empty:
            return {}
        
        optimal_idx = results['expected_roi'].idxmax()
        optimal_exposure = results.loc[optimal_idx, 'max_player_exposure']
        optimal_roi = results.loc[optimal_idx, 'expected_roi']
        
        return {
            'optimal_exposure': optimal_exposure,
            'optimal_roi': optimal_roi,
            'results': results
        }
    
    def optimize_for_contest_type(
        self,
        contest_type: str,
        parameter_ranges: Dict[str, List[Any]],
        base_config: Dict,
        num_simulations: int = 5000
    ) -> Dict:
        """
        Optimize parameters for specific contest type.
        
        NEW v7.0.0: Contest-specific auto-tuning!
        
        Args:
            contest_type: 'MILLY_MAKER', 'GPP', 'CASH', etc.
            parameter_ranges: Parameters to optimize
            base_config: Base configuration
            num_simulations: Simulations per test
        
        Returns:
            Dictionary with optimal configuration
        """
        logger.info(f"Optimizing for contest type: {contest_type}")
        
        # Adjust objective based on contest type
        if contest_type == 'CASH':
            objective = 'cash_rate'
        elif contest_type in ['MILLY_MAKER', 'GPP_LARGE']:
            objective = 'win_rate'
        else:
            objective = 'expected_roi'
        
        # Run grid search
        results = self.test_multiple_parameters(
            parameters=parameter_ranges,
            base_config=base_config,
            num_simulations=num_simulations,
            objective=objective
        )
        
        if results.empty:
            return {}
        
        # Get optimal configuration
        optimal_idx = results[objective].idxmax()
        optimal_config = results.loc[optimal_idx].to_dict()
        
        return {
            'contest_type': contest_type,
            'objective': objective,
            'optimal_config': optimal_config,
            'all_results': results
        }
    
    def find_optimal_portfolio_size(
        self,
        sizes: List[int],
        base_config: Dict,
        num_simulations: int = 5000,
        entry_fee: float = 20.0
    ) -> Dict:
        """
        Find optimal number of lineups to submit.
        
        Args:
            sizes: Portfolio sizes to test (e.g., [20, 50, 100, 150])
            base_config: Base configuration
            num_simulations: Simulations per size
            entry_fee: Entry fee per lineup
        
        Returns:
            Dictionary with optimal size and ROI curve
        """
        results = self.test_single_parameter(
            parameter_name='num_lineups',
            values=sizes,
            base_config=base_config,
            num_simulations=num_simulations,
            objective='expected_roi'
        )
        
        if results.empty:
            return {}
        
        # Calculate total expected profit for each size
        results['total_entries'] = results['num_lineups'] * entry_fee
        results['expected_profit'] = (
            results['total_entries'] * (results['expected_roi'] / 100)
        )
        
        # Find size with best profit
        optimal_idx = results['expected_profit'].idxmax()
        optimal_size = results.loc[optimal_idx, 'num_lineups']
        optimal_profit = results.loc[optimal_idx, 'expected_profit']
        
        return {
            'optimal_size': optimal_size,
            'expected_profit': optimal_profit,
            'results': results
        }
    
    def sensitivity_analysis(
        self,
        parameter_name: str,
        base_value: Any,
        variation_range: float,
        num_tests: int,
        base_config: Dict,
        num_simulations: int = 3000
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a parameter.
        
        Tests how sensitive results are to parameter changes.
        
        Args:
            parameter_name: Parameter to test
            base_value: Base value of parameter
            variation_range: How much to vary (e.g., 0.2 = ±20%)
            num_tests: Number of test points
            base_config: Base configuration
            num_simulations: Simulations per test
        
        Returns:
            DataFrame with sensitivity results
        """
        # Generate test values around base value
        if isinstance(base_value, (int, float)):
            min_val = base_value * (1 - variation_range)
            max_val = base_value * (1 + variation_range)
            test_values = np.linspace(min_val, max_val, num_tests)
        else:
            logger.error(f"Sensitivity analysis only supports numeric parameters")
            return pd.DataFrame()
        
        return self.test_single_parameter(
            parameter_name=parameter_name,
            values=test_values.tolist(),
            base_config=base_config,
            num_simulations=num_simulations,
            objective='expected_roi'
        )
    
    def generate_optimization_report(
        self,
        test_results: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Generate text report from optimization results.
        
        Args:
            test_results: Dictionary of test name -> results DataFrame
        
        Returns:
            Formatted report string
        """
        report = "=" * 70 + "\n"
        report += "STRATEGY OPTIMIZATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        for test_name, results in test_results.items():
            report += f"\n{test_name}:\n"
            report += "-" * 70 + "\n"
            
            if results.empty:
                report += "No results\n"
                continue
            
            # Show top 5 results
            top_5 = results.nlargest(5, 'expected_roi')
            
            for idx, row in top_5.iterrows():
                report += f"  Rank {idx + 1}:\n"
                for col in results.columns:
                    if col not in ['expected_roi', 'win_rate', 'cash_rate']:
                        report += f"    {col}: {row[col]}\n"
                report += f"    Expected ROI: {row['expected_roi']:.1f}%\n"
                report += f"    Win Rate: {row['win_rate']:.2f}%\n"
                report += f"    Cash Rate: {row['cash_rate']:.1f}%\n"
                report += "\n"
        
        report += "=" * 70 + "\n"
        
        return report
