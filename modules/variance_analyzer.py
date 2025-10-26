"""
Module 5: Variance Analyzer
Analyzes lineup and portfolio variance/risk profiles
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VarianceAnalyzer:
    """
    Analyzes variance and risk in lineups and portfolios.
    
    Features:
    - Lineup variance calculation
    - Portfolio risk profiling
    - Boom-or-bust identification
    - Risk/reward scoring
    - Distribution analysis
    """
    
    def __init__(self):
        """Initialize variance analyzer"""
        logger.info("VarianceAnalyzer initialized")
    
    def _calculate_lineup_variance(self, lineup: pd.DataFrame) -> Dict:
        """
        Calculate variance metrics for a single lineup.
        
        Args:
            lineup: Lineup DataFrame with projection, ceiling, floor
        
        Returns:
            Dictionary with variance metrics
        """
        # Get ceiling and floor
        if 'ceiling' in lineup.columns and 'floor' in lineup.columns:
            ceiling = lineup['ceiling'].sum()
            floor = lineup['floor'].sum()
        else:
            # Estimate from projection
            projection = lineup['projection'].sum()
            ceiling = projection * 1.4
            floor = projection * 0.6
        
        projection = lineup['projection'].sum()
        
        # Calculate metrics
        variance_range = ceiling - floor
        upside = ceiling - projection
        downside = projection - floor
        upside_ratio = upside / projection if projection > 0 else 0
        
        # Coefficient of variation (std dev / mean)
        std_estimate = variance_range / 4  # Rough estimate
        cv = std_estimate / projection if projection > 0 else 0
        
        return {
            'projection': projection,
            'ceiling': ceiling,
            'floor': floor,
            'variance_range': variance_range,
            'upside': upside,
            'downside': downside,
            'upside_ratio': upside_ratio,
            'coefficient_variation': cv
        }
    
    def analyze_lineup(
        self,
        lineup: pd.DataFrame,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Comprehensive variance analysis for single lineup.
        
        Args:
            lineup: Lineup DataFrame
            num_simulations: Number of Monte Carlo simulations
        
        Returns:
            Dictionary with complete variance profile
        """
        base_metrics = self._calculate_lineup_variance(lineup)
        
        # Simulate lineup scores
        simulated_scores = []
        
        for _ in range(num_simulations):
            lineup_score = 0
            
            for _, player in lineup.iterrows():
                projection = player['projection']
                
                # Estimate std dev
                if 'ceiling' in player and 'floor' in player:
                    std_dev = (player['ceiling'] - player['floor']) / 4
                else:
                    std_dev = projection * 0.35  # Default 35% variance
                
                # Sample score
                score = np.random.normal(projection, std_dev)
                score = max(0, score)  # Floor at 0
                
                lineup_score += score
            
            simulated_scores.append(lineup_score)
        
        simulated_scores = np.array(simulated_scores)
        
        # Calculate distribution metrics
        distribution_metrics = {
            'mean': np.mean(simulated_scores),
            'median': np.median(simulated_scores),
            'std_dev': np.std(simulated_scores),
            'percentile_5': np.percentile(simulated_scores, 5),
            'percentile_25': np.percentile(simulated_scores, 25),
            'percentile_75': np.percentile(simulated_scores, 75),
            'percentile_95': np.percentile(simulated_scores, 95),
            'skewness': self._calculate_skewness(simulated_scores)
        }
        
        # Classify risk profile
        cv = base_metrics['coefficient_variation']
        
        if cv > 0.4:
            risk_profile = 'high_variance'
            description = "Boom-or-bust lineup with extreme upside and downside"
        elif cv > 0.25:
            risk_profile = 'medium_variance'
            description = "Moderate variance with good upside potential"
        else:
            risk_profile = 'low_variance'
            description = "Safe lineup with consistent scoring expectation"
        
        return {
            **base_metrics,
            **distribution_metrics,
            'risk_profile': risk_profile,
            'description': description,
            'simulated_scores': simulated_scores
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        
        return skewness
    
    def analyze_portfolio(
        self,
        lineups: List[pd.DataFrame]
    ) -> Dict:
        """
        Analyze variance across entire portfolio.
        
        Args:
            lineups: List of lineup DataFrames
        
        Returns:
            Dictionary with portfolio variance profile
        """
        logger.info(f"Analyzing variance for {len(lineups)} lineups")
        
        lineup_variances = []
        
        for lineup in lineups:
            variance = self._calculate_lineup_variance(lineup)
            lineup_variances.append(variance)
        
        df = pd.DataFrame(lineup_variances)
        
        # Categorize lineups by variance
        high_var = df[df['coefficient_variation'] > 0.4]
        med_var = df[(df['coefficient_variation'] > 0.25) & (df['coefficient_variation'] <= 0.4)]
        low_var = df[df['coefficient_variation'] <= 0.25]
        
        # Portfolio-level metrics
        portfolio_metrics = {
            'total_lineups': len(lineups),
            'high_variance_count': len(high_var),
            'medium_variance_count': len(med_var),
            'low_variance_count': len(low_var),
            'high_variance_pct': (len(high_var) / len(lineups)) * 100,
            'medium_variance_pct': (len(med_var) / len(lineups)) * 100,
            'low_variance_pct': (len(low_var) / len(lineups)) * 100,
            'avg_projection': df['projection'].mean(),
            'avg_ceiling': df['ceiling'].mean(),
            'avg_floor': df['floor'].mean(),
            'avg_variance_range': df['variance_range'].mean(),
            'avg_upside_ratio': df['upside_ratio'].mean()
        }
        
        # Determine overall portfolio risk
        high_var_pct = portfolio_metrics['high_variance_pct']
        
        if high_var_pct > 60:
            portfolio_risk = 'aggressive'
            recommendation = "Highly aggressive portfolio - consider adding safer plays"
        elif high_var_pct > 40:
            portfolio_risk = 'balanced_aggressive'
            recommendation = "Good balance with slight aggressive tilt"
        elif high_var_pct > 20:
            portfolio_risk = 'balanced'
            recommendation = "Well-balanced portfolio across risk levels"
        else:
            portfolio_risk = 'conservative'
            recommendation = "Conservative portfolio - consider adding upside plays"
        
        portfolio_metrics['risk_classification'] = portfolio_risk
        portfolio_metrics['recommendation'] = recommendation
        
        return portfolio_metrics
    
    def identify_boom_or_bust(
        self,
        lineups: List[pd.DataFrame],
        threshold: float = 0.4
    ) -> List[int]:
        """
        Identify boom-or-bust lineups (high variance).
        
        Args:
            lineups: List of lineup DataFrames
            threshold: Coefficient of variation threshold
        
        Returns:
            List of lineup indices with high variance
        """
        boom_or_bust = []
        
        for idx, lineup in enumerate(lineups):
            variance = self._calculate_lineup_variance(lineup)
            
            if variance['coefficient_variation'] > threshold:
                boom_or_bust.append(idx)
        
        logger.info(f"Identified {len(boom_or_bust)} boom-or-bust lineups")
        
        return boom_or_bust
    
    def identify_safe_plays(
        self,
        lineups: List[pd.DataFrame],
        threshold: float = 0.25
    ) -> List[int]:
        """
        Identify safe lineups (low variance).
        
        Args:
            lineups: List of lineup DataFrames
            threshold: Coefficient of variation threshold
        
        Returns:
            List of lineup indices with low variance
        """
        safe_plays = []
        
        for idx, lineup in enumerate(lineups):
            variance = self._calculate_lineup_variance(lineup)
            
            if variance['coefficient_variation'] < threshold:
                safe_plays.append(idx)
        
        logger.info(f"Identified {len(safe_plays)} safe lineups")
        
        return safe_plays
    
    def compare_variance_profiles(
        self,
        lineups: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compare variance profiles across lineups.
        
        Args:
            lineups: List of lineup DataFrames
        
        Returns:
            DataFrame with variance comparison
        """
        results = []
        
        for idx, lineup in enumerate(lineups):
            variance = self._calculate_lineup_variance(lineup)
            
            results.append({
                'lineup_id': idx,
                **variance
            })
        
        df = pd.DataFrame(results)
        
        # Sort by variance (highest to lowest)
        df = df.sort_values('variance_range', ascending=False)
        
        return df
    
    def calculate_risk_reward_score(
        self,
        lineup: pd.DataFrame,
        risk_tolerance: float = 0.5
    ) -> float:
        """
        Calculate risk/reward score for lineup.
        
        Higher score = better risk-adjusted returns
        
        Args:
            lineup: Lineup DataFrame
            risk_tolerance: 0 (risk-averse) to 1 (risk-seeking)
        
        Returns:
            Risk/reward score
        """
        variance = self._calculate_lineup_variance(lineup)
        
        # Base score = projection
        base_score = variance['projection']
        
        # Adjust for upside (reward)
        upside_bonus = variance['upside'] * risk_tolerance
        
        # Penalize for downside (risk)
        downside_penalty = variance['downside'] * (1 - risk_tolerance)
        
        # Final score
        risk_reward = base_score + upside_bonus - downside_penalty
        
        return risk_reward
    
    def recommend_portfolio_mix(
        self,
        num_lineups: int,
        risk_profile: str = 'balanced'
    ) -> Dict:
        """
        Recommend portfolio variance mix.
        
        Args:
            num_lineups: Total lineups in portfolio
            risk_profile: 'conservative', 'balanced', 'aggressive'
        
        Returns:
            Dictionary with recommended distribution
        """
        if risk_profile == 'conservative':
            high_var_pct = 0.20
            med_var_pct = 0.40
            low_var_pct = 0.40
        elif risk_profile == 'balanced':
            high_var_pct = 0.40
            med_var_pct = 0.40
            low_var_pct = 0.20
        elif risk_profile == 'aggressive':
            high_var_pct = 0.60
            med_var_pct = 0.30
            low_var_pct = 0.10
        else:
            # Default to balanced
            high_var_pct = 0.40
            med_var_pct = 0.40
            low_var_pct = 0.20
        
        return {
            'total_lineups': num_lineups,
            'high_variance': int(num_lineups * high_var_pct),
            'medium_variance': int(num_lineups * med_var_pct),
            'low_variance': int(num_lineups * low_var_pct),
            'risk_profile': risk_profile
        }
    
    def generate_variance_report(
        self,
        portfolio_analysis: Dict
    ) -> str:
        """
        Generate text report of portfolio variance.
        
        Args:
            portfolio_analysis: Results from analyze_portfolio
        
        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "PORTFOLIO VARIANCE ANALYSIS\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Total Lineups: {portfolio_analysis['total_lineups']}\n\n"
        
        report += "Variance Distribution:\n"
        report += f"  High Variance (Boom-or-Bust): {portfolio_analysis['high_variance_count']} "
        report += f"({portfolio_analysis['high_variance_pct']:.1f}%)\n"
        report += f"  Medium Variance: {portfolio_analysis['medium_variance_count']} "
        report += f"({portfolio_analysis['medium_variance_pct']:.1f}%)\n"
        report += f"  Low Variance (Safe): {portfolio_analysis['low_variance_count']} "
        report += f"({portfolio_analysis['low_variance_pct']:.1f}%)\n\n"
        
        report += "Portfolio Metrics:\n"
        report += f"  Average Projection: {portfolio_analysis['avg_projection']:.1f}\n"
        report += f"  Average Ceiling: {portfolio_analysis['avg_ceiling']:.1f}\n"
        report += f"  Average Floor: {portfolio_analysis['avg_floor']:.1f}\n"
        report += f"  Average Variance Range: {portfolio_analysis['avg_variance_range']:.1f}\n\n"
        
        report += f"Risk Classification: {portfolio_analysis['risk_classification'].upper()}\n"
        report += f"Recommendation: {portfolio_analysis['recommendation']}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report
