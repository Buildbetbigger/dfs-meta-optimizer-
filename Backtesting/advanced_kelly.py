"""
Advanced Kelly Criterion - Phase 2
Variance-aware bankroll management with correlation considerations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)

class AdvancedKellyCriterion:
    """
    Advanced Kelly Criterion for DFS bankroll management
    
    Standard Kelly: f* = (bp - q) / b
    where b = odds, p = win prob, q = 1-p
    
    Advanced Kelly considers:
    1. Variance of outcomes (not just expected value)
    2. Correlation between entries
    3. Fractional Kelly for risk management
    4. Multi-contest portfolio optimization
    """
    
    def __init__(self, covariance_analyzer=None):
        """
        Initialize Kelly calculator
        
        Args:
            covariance_analyzer: Optional CovarianceAnalyzer for correlation
        """
        self.cov_analyzer = covariance_analyzer
        
    def calculate_simple_kelly(
        self,
        win_prob: float,
        prize_multiple: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Basic Kelly formula
        
        Args:
            win_prob: Probability of winning (0-1)
            prize_multiple: Prize / Entry (e.g., $100 prize / $10 entry = 10x)
            kelly_fraction: Fraction of Kelly to use (conservative: 0.25)
        
        Returns:
            Fraction of bankroll to wager
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
        
        lose_prob = 1 - win_prob
        
        # Kelly formula: f* = (bp - q) / b
        # where b = net odds (prize_multiple - 1)
        net_odds = prize_multiple - 1
        
        kelly_pct = (net_odds * win_prob - lose_prob) / net_odds
        
        # Apply fraction
        kelly_pct = max(0, kelly_pct * kelly_fraction)
        
        return kelly_pct
    
    def calculate_kelly_with_variance(
        self,
        expected_value: float,
        variance: float,
        entry_fee: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Kelly criterion adjusted for variance
        
        When variance is high, Kelly recommends smaller bet size
        
        Args:
            expected_value: Expected profit (not EV%)
            variance: Variance of outcomes
            entry_fee: Cost to enter
            kelly_fraction: Fraction of Kelly
        
        Returns:
            Fraction of bankroll to wager
        """
        if expected_value <= 0:
            return 0.0
        
        # Variance-adjusted Kelly
        # f* = μ / σ²
        # where μ = expected value, σ² = variance
        
        if variance > 0:
            kelly_pct = expected_value / variance
        else:
            # No variance = use simple Kelly
            win_prob = 0.5  # Default assumption
            prize_multiple = (expected_value + entry_fee) / entry_fee
            return self.calculate_simple_kelly(win_prob, prize_multiple, kelly_fraction)
        
        # Apply fraction and cap
        kelly_pct = kelly_pct * kelly_fraction
        kelly_pct = min(kelly_pct, 0.25)  # Never risk more than 25% even with full Kelly
        
        return max(0, kelly_pct)
    
    def calculate_multi_contest_kelly(
        self,
        contests: List[Dict],
        bankroll: float,
        correlation_matrix: Optional[np.ndarray] = None,
        kelly_fraction: float = 0.25
    ) -> Dict[str, float]:
        """
        Optimal bankroll allocation across multiple contests
        
        Considers correlation between contests
        
        Args:
            contests: List of contest dicts with:
                - 'name': contest name
                - 'expected_value': Expected profit
                - 'variance': Outcome variance
                - 'entry_fee': Entry cost
            bankroll: Total bankroll
            correlation_matrix: Optional correlation between contests
            kelly_fraction: Fraction of Kelly
        
        Returns:
            Dict mapping contest name to number of entries
        """
        n = len(contests)
        
        # Expected values vector
        ev_vector = np.array([c['expected_value'] for c in contests])
        
        # Variance-covariance matrix
        if correlation_matrix is not None:
            # Build covariance from correlation and variances
            stds = np.sqrt([c['variance'] for c in contests])
            cov_matrix = np.outer(stds, stds) * correlation_matrix
        else:
            # Assume independent
            cov_matrix = np.diag([c['variance'] for c in contests])
        
        # Kelly weights: w = Σ^-1 * μ
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            kelly_weights = np.dot(inv_cov, ev_vector)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using independent Kelly")
            kelly_weights = np.array([
                self.calculate_kelly_with_variance(
                    c['expected_value'],
                    c['variance'],
                    c['entry_fee'],
                    kelly_fraction=1.0
                ) for c in contests
            ])
        
        # Apply Kelly fraction
        kelly_weights = kelly_weights * kelly_fraction
        
        # Ensure non-negative
        kelly_weights = np.maximum(kelly_weights, 0)
        
        # Convert to entries
        allocations = {}
        for i, contest in enumerate(contests):
            allocation_amount = bankroll * kelly_weights[i]
            num_entries = int(allocation_amount / contest['entry_fee'])
            allocations[contest['name']] = max(0, num_entries)
        
        return allocations
    
    def calculate_kelly_growth_rate(
        self,
        kelly_fraction: float,
        win_prob: float,
        prize_multiple: float
    ) -> float:
        """
        Calculate expected growth rate of bankroll
        
        G(f) = p * log(1 + fb) + q * log(1 - f)
        where f = bet fraction, b = net odds
        
        Args:
            kelly_fraction: Fraction of Kelly being used
            win_prob: Win probability
            prize_multiple: Prize / Entry
        
        Returns:
            Expected growth rate per bet
        """
        f = self.calculate_simple_kelly(win_prob, prize_multiple, kelly_fraction)
        
        if f <= 0 or f >= 1:
            return 0.0
        
        net_odds = prize_multiple - 1
        lose_prob = 1 - win_prob
        
        # Growth rate
        growth_rate = (
            win_prob * np.log(1 + f * net_odds) +
            lose_prob * np.log(1 - f)
        )
        
        return growth_rate
    
    def find_optimal_kelly_fraction(
        self,
        win_prob: float,
        prize_multiple: float,
        risk_tolerance: float = 0.5
    ) -> float:
        """
        Find optimal Kelly fraction balancing growth and risk
        
        Args:
            win_prob: Win probability
            prize_multiple: Prize / Entry
            risk_tolerance: 0 (very conservative) to 1 (full Kelly)
        
        Returns:
            Optimal Kelly fraction
        """
        # Objective: maximize growth rate - risk penalty
        def objective(kelly_frac):
            growth = self.calculate_kelly_growth_rate(
                kelly_frac,
                win_prob,
                prize_multiple
            )
            
            # Risk penalty (higher fraction = higher risk)
            risk_penalty = (1 - risk_tolerance) * kelly_frac
            
            return -(growth - risk_penalty)  # Negative for minimization
        
        result = minimize_scalar(
            objective,
            bounds=(0.05, 1.0),
            method='bounded'
        )
        
        return result.x
    
    def simulate_kelly_outcomes(
        self,
        bankroll: float,
        kelly_fraction: float,
        win_prob: float,
        prize_multiple: float,
        entry_fee: float,
        num_bets: int = 100,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Monte Carlo simulation of Kelly betting outcomes
        
        Args:
            bankroll: Starting bankroll
            kelly_fraction: Kelly fraction to use
            win_prob: Win probability
            prize_multiple: Prize / Entry
            entry_fee: Entry cost
            num_bets: Number of bets to simulate
            num_simulations: Number of simulation runs
        
        Returns:
            Dictionary with simulation results
        """
        f = self.calculate_simple_kelly(win_prob, prize_multiple, kelly_fraction)
        
        final_bankrolls = []
        
        for _ in range(num_simulations):
            br = bankroll
            
            for _ in range(num_bets):
                if br <= entry_fee:
                    break  # Bankroll depleted
                
                # Bet size
                bet = min(br * f, br)  # Never bet more than bankroll
                bet = max(bet, entry_fee)  # At least one entry
                
                # Outcome
                if np.random.random() < win_prob:
                    # Win
                    br += bet * (prize_multiple - 1)
                else:
                    # Lose
                    br -= bet
            
            final_bankrolls.append(br)
        
        final_bankrolls = np.array(final_bankrolls)
        
        return {
            'mean_final_bankroll': np.mean(final_bankrolls),
            'median_final_bankroll': np.median(final_bankrolls),
            'std_final_bankroll': np.std(final_bankrolls),
            'min_final_bankroll': np.min(final_bankrolls),
            'max_final_bankroll': np.max(final_bankrolls),
            'bust_rate': np.sum(final_bankrolls < entry_fee) / num_simulations,
            '95th_percentile': np.percentile(final_bankrolls, 95),
            '5th_percentile': np.percentile(final_bankrolls, 5)
        }
    
    def calculate_kelly_for_lineup_portfolio(
        self,
        lineups: List[Dict],
        contest: Dict,
        bankroll: float,
        kelly_fraction: float = 0.25
    ) -> int:
        """
        Calculate Kelly-optimal entries for a lineup portfolio
        
        Considers correlation between lineups
        
        Args:
            lineups: List of lineup dicts with expected scores
            contest: Contest dict with structure
            bankroll: Total bankroll
            kelly_fraction: Fraction of Kelly
        
        Returns:
            Recommended number of total entries
        """
        # Estimate EV and variance for portfolio
        expected_scores = [l.get('projected_points', 0) for l in lineups]
        
        if not expected_scores:
            return 0
        
        mean_score = np.mean(expected_scores)
        std_score = np.std(expected_scores)
        
        # Estimate win probability (simplified)
        contest_size = contest.get('size', 1000)
        win_threshold = 180  # Typical GPP-winning score
        
        # Z-score
        if std_score > 0:
            z = (win_threshold - mean_score) / std_score
            # Approximate win probability
            from scipy.stats import norm
            win_prob = 1 - norm.cdf(z)
        else:
            win_prob = 0.01
        
        # Expected value
        entry_fee = contest.get('entry_fee', 10)
        prize_pool = contest.get('prize_pool', entry_fee * contest_size * 0.9)
        top_prize = contest.get('top_payout', prize_pool * 0.2)
        
        ev = win_prob * top_prize - (1 - win_prob) * entry_fee
        variance = (top_prize ** 2) * win_prob * (1 - win_prob)
        
        # Kelly calculation
        kelly_pct = self.calculate_kelly_with_variance(
            ev,
            variance,
            entry_fee,
            kelly_fraction
        )
        
        # Number of entries
        total_investment = bankroll * kelly_pct
        num_entries = int(total_investment / entry_fee)
        
        # Cap at portfolio size
        num_entries = min(num_entries, len(lineups))
        
        return max(0, num_entries)

if __name__ == "__main__":
    # Test Kelly calculations
    kelly = AdvancedKellyCriterion()
    
    print("=== Simple Kelly ===")
    win_prob = 0.15
    prize_mult = 10
    kelly_frac = 0.25
    
    f = kelly.calculate_simple_kelly(win_prob, prize_mult, kelly_frac)
    print(f"Win Prob: {win_prob}, Prize: {prize_mult}x")
    print(f"Kelly fraction (0.25): {f:.2%}")
    
    print("\n=== Variance-Adjusted Kelly ===")
    ev = 5  # Expected $5 profit
    variance = 100
    entry = 10
    
    f = kelly.calculate_kelly_with_variance(ev, variance, entry, kelly_frac)
    print(f"EV: ${ev}, Variance: {variance}, Entry: ${entry}")
    print(f"Kelly fraction: {f:.2%}")
    
    print("\n=== Optimal Kelly Fraction ===")
    optimal_frac = kelly.find_optimal_kelly_fraction(win_prob, prize_mult, risk_tolerance=0.5)
    print(f"Optimal fraction: {optimal_frac:.3f}")
    
    print("\n=== Kelly Simulation ===")
    sim_results = kelly.simulate_kelly_outcomes(
        bankroll=1000,
        kelly_fraction=0.25,
        win_prob=0.15,
        prize_multiple=10,
        entry_fee=10,
        num_bets=50,
        num_simulations=1000
    )
    
    print(f"Mean final bankroll: ${sim_results['mean_final_bankroll']:.2f}")
    print(f"Median final bankroll: ${sim_results['median_final_bankroll']:.2f}")
    print(f"Bust rate: {sim_results['bust_rate']:.1%}")
    print(f"95th percentile: ${sim_results['95th_percentile']:.2f}")
