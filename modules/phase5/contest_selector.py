"""
DFS Meta-Optimizer v7.0.0 - Contest Selector Module
Intelligent contest selection and EV analysis

Features:
- Field strength estimation
- Contest EV calculation
- Multi-contest comparison
- Risk-adjusted recommendations
- Portfolio fit analysis
- Kelly Criterion bankroll management
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Contest:
    """Represents a DFS contest"""
    name: str
    size: int  # Total entries
    entry_fee: float
    prize_pool: float
    top_payout: float
    contest_type: str  # 'GPP', 'CASH', 'SATELLITE', 'H2H'
    
    # Prize structure (optional)
    places_paid: Optional[int] = None
    min_cash: Optional[float] = None


class ContestSelector:
    """
    Analyzes and recommends contests based on lineup portfolio.
    
    NEW v7.0.0 Features:
    - Automated contest selection
    - Expected value optimization
    - Kelly Criterion bankroll sizing
    - Risk-adjusted recommendations
    - Portfolio-contest fit analysis
    """
    
    def __init__(self):
        """Initialize contest selector"""
        logger.info("ContestSelector v7.0.0 initialized")
    
    def estimate_field_strength(self, contest: Contest) -> float:
        """
        Estimate field strength (0-100).
        
        Larger contests = stronger fields
        Higher stakes = stronger fields
        
        Args:
            contest: Contest to evaluate
        
        Returns:
            Field strength score (0-100)
        """
        # Base strength from size
        if contest.size >= 100000:
            size_strength = 85
        elif contest.size >= 50000:
            size_strength = 80
        elif contest.size >= 10000:
            size_strength = 70
        elif contest.size >= 1000:
            size_strength = 60
        else:
            size_strength = 50
        
        # Adjust for entry fee
        if contest.entry_fee >= 100:
            fee_adjust = 10
        elif contest.entry_fee >= 50:
            fee_adjust = 5
        else:
            fee_adjust = 0
        
        field_strength = min(100, size_strength + fee_adjust)
        
        return field_strength
    
    def estimate_win_probability(
        self,
        portfolio_strength: float,
        field_strength: float,
        contest_size: int
    ) -> float:
        """
        Estimate win probability based on skill edge.
        
        Args:
            portfolio_strength: Your portfolio quality (0-100)
            field_strength: Field quality (0-100)
            contest_size: Contest entries
        
        Returns:
            Win probability (0-1)
        """
        # Base win rate = 1 / size
        base_rate = 1.0 / contest_size
        
        # Adjust for skill edge
        skill_edge = (portfolio_strength - field_strength) / 100
        
        # Multiply base rate by skill multiplier
        # +10 skill edge = 1.5x win rate
        # -10 skill edge = 0.5x win rate
        multiplier = 1 + (skill_edge * 5)
        multiplier = max(0.1, min(10, multiplier))  # Cap at 0.1x to 10x
        
        win_prob = base_rate * multiplier
        
        # Cap at reasonable maximum
        win_prob = min(0.05, win_prob)  # Max 5% win rate
        
        return win_prob
    
    def estimate_cash_probability(
        self,
        portfolio_strength: float,
        field_strength: float,
        contest_type: str
    ) -> float:
        """
        Estimate cash probability.
        
        Args:
            portfolio_strength: Your portfolio quality
            field_strength: Field quality
            contest_type: Contest type
        
        Returns:
            Cash probability (0-1)
        """
        skill_edge = portfolio_strength - field_strength
        
        if contest_type == 'CASH':
            # Cash games: ~45-55% depending on skill
            base_cash = 0.50
            cash_prob = base_cash + (skill_edge / 200)
        elif contest_type == 'GPP':
            # GPP: ~15-25% cash (top 20%)
            base_cash = 0.20
            cash_prob = base_cash + (skill_edge / 250)
        elif contest_type == 'SATELLITE':
            # Satellite: similar to cash
            base_cash = 0.45
            cash_prob = base_cash + (skill_edge / 200)
        else:
            base_cash = 0.50
            cash_prob = base_cash
        
        # Bounds
        cash_prob = max(0.05, min(0.95, cash_prob))
        
        return cash_prob
    
    def calculate_contest_ev(
        self,
        contest: Contest,
        portfolio_strength: float = 75.0,
        num_entries: int = 1
    ) -> Dict:
        """
        Calculate expected value for a contest.
        
        Args:
            contest: Contest details
            portfolio_strength: Your portfolio quality (0-100)
            num_entries: Number of entries you're submitting
        
        Returns:
            Dictionary with EV metrics
        """
        field_strength = self.estimate_field_strength(contest)
        
        # Estimate probabilities
        win_prob = self.estimate_win_probability(
            portfolio_strength,
            field_strength,
            contest.size
        )
        
        cash_prob = self.estimate_cash_probability(
            portfolio_strength,
            field_strength,
            contest.contest_type
        )
        
        # Calculate EV
        total_entry = contest.entry_fee * num_entries
        
        # Simplified EV calculation
        if contest.contest_type == 'CASH':
            # Cash = double your money if you win
            expected_return = cash_prob * (total_entry * 1.8)
        else:
            # GPP = win big or lose
            expected_return = (
                win_prob * contest.top_payout +
                cash_prob * (total_entry * 2)  # Small cashes
            )
        
        ev = expected_return - total_entry
        roi = (ev / total_entry) * 100 if total_entry > 0 else 0
        
        return {
            'contest_name': contest.name,
            'entry_fee': contest.entry_fee,
            'num_entries': num_entries,
            'total_cost': total_entry,
            'win_probability': win_prob,
            'cash_probability': cash_prob,
            'expected_return': expected_return,
            'expected_value': ev,
            'roi': roi,
            'field_strength': field_strength
        }
    
    def compare_contests(
        self,
        contests: List[Contest],
        portfolio_strength: float = 75.0,
        num_entries: int = 20
    ) -> pd.DataFrame:
        """
        Compare multiple contests by EV.
        
        Args:
            contests: List of contests to compare
            portfolio_strength: Your portfolio quality
            num_entries: Entries per contest
        
        Returns:
            DataFrame with comparison (sorted by ROI)
        """
        results = []
        
        for contest in contests:
            ev_data = self.calculate_contest_ev(
                contest,
                portfolio_strength,
                num_entries
            )
            
            results.append(ev_data)
        
        df = pd.DataFrame(results)
        
        # Sort by ROI
        df = df.sort_values('roi', ascending=False)
        
        logger.info(f"Compared {len(contests)} contests")
        
        return df
    
    def recommend_best_contest(
        self,
        contests: List[Contest],
        portfolio_strength: float = 75.0,
        num_entries: int = 20,
        risk_tolerance: str = 'balanced'
    ) -> Dict:
        """
        Recommend best contest for your portfolio.
        
        NEW v7.0.0: Automated contest selection!
        
        Args:
            contests: Available contests
            portfolio_strength: Your portfolio quality
            num_entries: Number of entries
            risk_tolerance: 'conservative', 'balanced', 'aggressive'
        
        Returns:
            Dictionary with recommendation
        """
        comparison = self.compare_contests(contests, portfolio_strength, num_entries)
        
        if comparison.empty:
            return {}
        
        if risk_tolerance == 'conservative':
            # Prefer high cash rate
            best_idx = comparison['cash_probability'].idxmax()
        elif risk_tolerance == 'aggressive':
            # Prefer high win rate (big prizes)
            best_idx = comparison['win_probability'].idxmax()
        else:
            # Balanced: best ROI
            best_idx = comparison['roi'].idxmax()
        
        recommendation = comparison.loc[best_idx].to_dict()
        recommendation['reason'] = self._explain_recommendation(
            recommendation,
            risk_tolerance
        )
        
        logger.info(f"Recommended: {recommendation['contest_name']} "
                   f"(ROI: {recommendation['roi']:.1f}%)")
        
        return recommendation
    
    def _explain_recommendation(
        self,
        contest_ev: Dict,
        risk_tolerance: str
    ) -> str:
        """Generate explanation for recommendation"""
        if risk_tolerance == 'conservative':
            return (
                f"Best for conservative play with {contest_ev['cash_probability']*100:.1f}% "
                f"cash rate and {contest_ev['roi']:.1f}% ROI"
            )
        elif risk_tolerance == 'aggressive':
            return (
                f"Best upside potential with {contest_ev['win_probability']*100:.2f}% "
                f"win rate and {contest_ev['roi']:.1f}% ROI"
            )
        else:
            return (
                f"Best overall value with {contest_ev['roi']:.1f}% ROI "
                f"and {contest_ev['cash_probability']*100:.1f}% cash rate"
            )
    
    def analyze_portfolio_fit(
        self,
        contest: Contest,
        portfolio_stats: Dict
    ) -> Dict:
        """
        Analyze how well portfolio fits contest.
        
        Args:
            contest: Contest to analyze
            portfolio_stats: Portfolio statistics from VarianceAnalyzer
        
        Returns:
            Dictionary with fit analysis
        """
        # Extract portfolio stats
        variance_profile = portfolio_stats.get('risk_classification', 'balanced')
        
        # Determine fit
        if contest.contest_type == 'CASH':
            # Cash = want low variance
            if variance_profile == 'conservative':
                fit_score = 90
                fit_desc = "Excellent fit - conservative portfolio for cash game"
            elif variance_profile == 'balanced':
                fit_score = 75
                fit_desc = "Good fit - balanced portfolio can cash consistently"
            else:
                fit_score = 50
                fit_desc = "Poor fit - aggressive portfolio risky for cash"
        
        elif contest.contest_type == 'GPP':
            # GPP = want high variance/ceiling
            if variance_profile == 'aggressive':
                fit_score = 90
                fit_desc = "Excellent fit - aggressive portfolio for GPP upside"
            elif variance_profile == 'balanced':
                fit_score = 75
                fit_desc = "Good fit - balanced portfolio has GPP potential"
            else:
                fit_score = 50
                fit_desc = "Poor fit - conservative portfolio lacks GPP upside"
        
        else:
            fit_score = 70
            fit_desc = "Moderate fit for this contest type"
        
        return {
            'contest_name': contest.name,
            'contest_type': contest.contest_type,
            'portfolio_variance': variance_profile,
            'fit_score': fit_score,
            'fit_description': fit_desc
        }
    
    def calculate_bankroll_kelly(
        self,
        contest_ev: Dict,
        bankroll: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate optimal entry size using Kelly Criterion.
        
        NEW v7.0.0: Professional bankroll management!
        
        Args:
            contest_ev: Contest EV from calculate_contest_ev
            bankroll: Your total bankroll
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        
        Returns:
            Recommended number of entries
        """
        roi = contest_ev['roi'] / 100
        entry_fee = contest_ev['entry_fee']
        
        if roi <= 0:
            return 0  # No edge, don't play
        
        # Simplified Kelly: f = edge / odds
        kelly_pct = roi * kelly_fraction
        
        # Recommended amount
        recommended = bankroll * kelly_pct
        
        # Convert to number of entries
        num_entries = int(recommended / entry_fee)
        
        # Cap at reasonable maximum (10% of bankroll)
        max_entries = int((bankroll * 0.1) / entry_fee)
        num_entries = min(num_entries, max_entries)
        
        logger.info(f"Kelly recommends {num_entries} entries for ${bankroll:.2f} bankroll")
        
        return num_entries
    
    def generate_contest_report(
        self,
        comparison_df: pd.DataFrame
    ) -> str:
        """
        Generate contest comparison report.
        
        Args:
            comparison_df: Results from compare_contests
        
        Returns:
            Formatted report string
        """
        report = "=" * 70 + "\n"
        report += "CONTEST COMPARISON REPORT\n"
        report += "=" * 70 + "\n\n"
        
        for idx, row in comparison_df.iterrows():
            rank = idx + 1
            
            report += f"#{rank}: {row['contest_name']}\n"
            report += "-" * 70 + "\n"
            report += f"  Entry Fee: ${row['entry_fee']:.2f} x {row['num_entries']} entries "
            report += f"= ${row['total_cost']:.2f}\n"
            report += f"  Expected ROI: {row['roi']:.1f}%\n"
            report += f"  Expected Value: ${row['expected_value']:.2f}\n"
            report += f"  Win Probability: {row['win_probability']*100:.2f}%\n"
            report += f"  Cash Probability: {row['cash_probability']*100:.1f}%\n"
            report += f"  Field Strength: {row['field_strength']:.0f}/100\n"
            report += "\n"
        
        # Recommendation
        if not comparison_df.empty:
            best = comparison_df.iloc[0]
            report += "RECOMMENDATION:\n"
            report += f"  Best Contest: {best['contest_name']}\n"
            report += f"  Expected ROI: {best['roi']:.1f}%\n"
            report += f"  Total Investment: ${best['total_cost']:.2f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report
