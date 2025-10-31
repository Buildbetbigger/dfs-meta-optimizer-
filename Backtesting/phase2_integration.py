"""
Phase 2 Integration Module
Combines all advanced math components into unified pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Import Phase 2 modules
from mock_data_generator import MockDataGenerator
from covariance_analyzer import CovarianceAnalyzer
from sharpe_optimizer import SharpeOptimizer
from advanced_kelly import AdvancedKellyCriterion
from bayesian_optimizer import BayesianOptimizer, StrategyTuner
from mcts_captain import MCTSCaptainSelector, ShowdownOptimizer

logger = logging.getLogger(__name__)

class Phase2Pipeline:
    """
    Complete Phase 2 pipeline integrating:
    1. Historical data analysis (covariance)
    2. Risk-adjusted optimization (Sharpe)
    3. Advanced bankroll management (Kelly)
    4. Hyperparameter tuning (Bayesian)
    5. Captain selection (MCTS)
    """
    
    def __init__(self):
        """Initialize all Phase 2 components"""
        logger.info("Initializing Phase 2 Advanced Math Pipeline")
        
        # Data
        self.data_generator = MockDataGenerator()
        self.historical_data = None
        
        # Analysis
        self.cov_analyzer = CovarianceAnalyzer()
        self.sharpe_optimizer = None
        
        # Optimization
        self.kelly_calculator = AdvancedKellyCriterion()
        self.bayesian_opt = None
        self.mcts_selector = MCTSCaptainSelector()
        
        # Results
        self.optimization_history = []
        
    def load_data(self, use_mock: bool = True, custom_data: Optional[Dict] = None):
        """
        Load historical data
        
        Args:
            use_mock: Use mock data generator
            custom_data: Optional custom data dict
        """
        if use_mock:
            logger.info("Generating mock historical data")
            data = self.data_generator.generate_full_dataset()
            self.historical_data = data
        else:
            self.historical_data = custom_data
        
        # Load into covariance analyzer
        self.cov_analyzer.load_historical_data(self.historical_data['historical'])
        self.cov_analyzer.calculate_player_covariance()
        
        logger.info("Data loaded and covariance calculated")
        
    def optimize_portfolio_risk_adjusted(
        self,
        player_pool: pd.DataFrame,
        num_lineups: int = 10
    ) -> List[Dict]:
        """
        Generate risk-adjusted optimal lineups using Sharpe ratio
        
        Args:
            player_pool: Current week player pool
            num_lineups: Number of lineups to generate
        
        Returns:
            List of optimized lineups
        """
        logger.info(f"Generating {num_lineups} Sharpe-optimized lineups")
        
        # Initialize Sharpe optimizer with covariance data
        self.sharpe_optimizer = SharpeOptimizer(self.cov_analyzer)
        
        lineups = []
        for i in range(num_lineups):
            # Add randomness for diversity
            player_sample = player_pool.sample(min(30, len(player_pool)))
            
            result = self.sharpe_optimizer.optimize_sharpe_ratio(
                player_sample,
                num_players=9
            )
            
            lineups.append({
                'lineup_id': i + 1,
                'players': result['players'],
                'sharpe_ratio': result['sharpe_ratio'],
                'expected_return': result['expected_return'],
                'risk': result['portfolio_std']
            })
        
        return lineups
    
    def calculate_kelly_sizing(
        self,
        lineups: List[Dict],
        contest: Dict,
        bankroll: float
    ) -> Dict:
        """
        Calculate Kelly-optimal entry sizing
        
        Args:
            lineups: Generated lineups
            contest: Contest details
            bankroll: Available bankroll
        
        Returns:
            Kelly recommendations
        """
        logger.info("Calculating Kelly-optimal sizing")
        
        # Link Kelly calculator to covariance analyzer
        self.kelly_calculator.cov_analyzer = self.cov_analyzer
        
        # Calculate optimal entries
        num_entries = self.kelly_calculator.calculate_kelly_for_lineup_portfolio(
            lineups,
            contest,
            bankroll,
            kelly_fraction=0.25
        )
        
        # Simulate outcomes
        sim_results = self.kelly_calculator.simulate_kelly_outcomes(
            bankroll=bankroll,
            kelly_fraction=0.25,
            win_prob=0.15,
            prize_multiple=contest.get('prize_multiple', 10),
            entry_fee=contest.get('entry_fee', 10),
            num_bets=50,
            num_simulations=500
        )
        
        return {
            'recommended_entries': num_entries,
            'total_investment': num_entries * contest.get('entry_fee', 10),
            'bankroll_pct': (num_entries * contest.get('entry_fee', 10)) / bankroll,
            'simulation': sim_results
        }
    
    def tune_strategy_parameters(
        self,
        validation_weeks: List[int] = None
    ) -> Dict:
        """
        Use Bayesian optimization to tune strategy parameters
        
        Args:
            validation_weeks: Weeks to validate on
        
        Returns:
            Optimal parameters
        """
        if validation_weeks is None:
            validation_weeks = [1, 2, 3]
        
        logger.info("Running Bayesian optimization for parameter tuning")
        
        # Initialize tuner
        tuner = StrategyTuner(self.historical_data.get('contest_results'))
        
        # Define objective function
        def lineup_generator(weights: Dict, week: int) -> List[Dict]:
            # Mock lineup generation with weights
            return [{'projected_points': np.random.normal(140, 20)} for _ in range(20)]
        
        # Tune weights
        result = tuner.tune_weight_parameters(lineup_generator, validation_weeks)
        
        logger.info(f"Optimal parameters: {result['best_parameters']}")
        
        return result
    
    def optimize_showdown_with_mcts(
        self,
        players: pd.DataFrame,
        num_lineups: int = 5
    ) -> List[Dict]:
        """
        Generate showdown lineups with MCTS captain selection
        
        Args:
            players: Showdown player pool
            num_lineups: Number of lineups
        
        Returns:
            Optimized showdown lineups
        """
        logger.info(f"Generating {num_lineups} showdown lineups with MCTS")
        
        optimizer = ShowdownOptimizer(mcts_iterations=500)
        lineups = optimizer.optimize_showdown_lineup(players, num_lineups)
        
        return lineups
    
    def run_full_pipeline(
        self,
        player_pool: pd.DataFrame,
        contest: Dict,
        bankroll: float,
        mode: str = 'classic'
    ) -> Dict:
        """
        Run complete Phase 2 pipeline
        
        Args:
            player_pool: Current player pool
            contest: Contest details
            bankroll: Available bankroll
            mode: 'classic' or 'showdown'
        
        Returns:
            Complete optimization results
        """
        logger.info("=" * 70)
        logger.info("PHASE 2 ADVANCED MATH PIPELINE")
        logger.info("=" * 70)
        
        results = {}
        
        # Step 1: Load data if not already loaded
        if self.historical_data is None:
            self.load_data(use_mock=True)
        
        # Step 2: Generate lineups
        if mode == 'showdown':
            lineups = self.optimize_showdown_with_mcts(player_pool, num_lineups=10)
        else:
            lineups = self.optimize_portfolio_risk_adjusted(player_pool, num_lineups=20)
        
        results['lineups'] = lineups
        results['num_lineups'] = len(lineups)
        
        # Step 3: Calculate Kelly sizing
        kelly_results = self.calculate_kelly_sizing(lineups, contest, bankroll)
        results['kelly'] = kelly_results
        
        # Step 4: Risk metrics
        if mode == 'classic' and lineups:
            sharpe_comparison = self.sharpe_optimizer.compare_lineups_sharpe(
                lineups,
                player_pool
            )
            results['sharpe_rankings'] = sharpe_comparison.to_dict('records')
        
        # Step 5: Strategy tuning (optional, can be slow)
        # tuning_results = self.tune_strategy_parameters()
        # results['tuned_parameters'] = tuning_results
        
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Generated {results['num_lineups']} lineups")
        logger.info(f"Kelly recommends {kelly_results['recommended_entries']} entries")
        logger.info(f"Investment: ${kelly_results['total_investment']:.2f} ({kelly_results['bankroll_pct']:.1%})")
        logger.info("=" * 70)
        
        return results
    
    def generate_phase2_report(self, results: Dict) -> str:
        """
        Generate comprehensive Phase 2 report
        
        Args:
            results: Results from run_full_pipeline
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("PHASE 2 ADVANCED MATH - OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Lineup summary
        report.append("LINEUP GENERATION:")
        report.append(f"  Total Lineups: {results['num_lineups']}")
        
        if 'sharpe_rankings' in results:
            report.append("\n  Top 5 Lineups by Sharpe Ratio:")
            for i, lineup in enumerate(results['sharpe_rankings'][:5]):
                report.append(f"    #{i+1}: Sharpe={lineup['sharpe_ratio']:.3f}, Return={lineup['expected_return']:.3f}")
        
        # Kelly recommendations
        kelly = results.get('kelly', {})
        report.append("\nKELLY CRITERION BANKROLL MANAGEMENT:")
        report.append(f"  Recommended Entries: {kelly.get('recommended_entries', 0)}")
        report.append(f"  Total Investment: ${kelly.get('total_investment', 0):.2f}")
        report.append(f"  Bankroll %: {kelly.get('bankroll_pct', 0):.1%}")
        
        sim = kelly.get('simulation', {})
        if sim:
            report.append("\n  Monte Carlo Simulation (500 runs):")
            report.append(f"    Mean Final Bankroll: ${sim.get('mean_final_bankroll', 0):.2f}")
            report.append(f"    Bust Rate: {sim.get('bust_rate', 0):.1%}")
            report.append(f"    95th Percentile: ${sim.get('95th_percentile', 0):.2f}")
        
        # Correlation analysis
        report.append("\nCORRELATION ANALYSIS:")
        pos_corr = self.cov_analyzer.get_position_correlations()
        report.append("  Position Correlations:")
        report.append(f"    QB-WR: {pos_corr.loc['QB', 'WR']:.3f}")
        report.append(f"    RB-RB: {pos_corr.loc['RB', 'RB']:.3f}")
        report.append(f"    QB-DST: {pos_corr.loc['QB', 'DST']:.3f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test full pipeline
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Phase 2 Pipeline...\n")
    
    # Initialize pipeline
    pipeline = Phase2Pipeline()
    
    # Load data
    pipeline.load_data(use_mock=True)
    
    # Mock player pool
    player_pool = pipeline.historical_data['historical'][
        pipeline.historical_data['historical']['week'] == 8
    ].copy()
    
    # Mock contest
    contest = {
        'name': 'Test GPP',
        'size': 10000,
        'entry_fee': 10,
        'prize_pool': 80000,
        'top_payout': 10000,
        'prize_multiple': 10
    }
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        player_pool,
        contest,
        bankroll=1000,
        mode='classic'
    )
    
    # Generate report
    report = pipeline.generate_phase2_report(results)
    print("\n" + report)
