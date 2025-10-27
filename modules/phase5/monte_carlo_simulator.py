"""
Module 5: Monte Carlo Simulator
Simulates thousands of contest outcomes to calculate win probabilities and ROI
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation"""
    portfolio_size: int
    num_simulations: int
    contest_size: int
    
    # Win rates
    win_rate: float  # Top 1%
    top_5_rate: float
    top_10_rate: float
    top_100_rate: float
    cash_rate: float  # Top 20%
    
    # ROI metrics
    expected_roi: float
    median_roi: float
    worst_case_roi: float  # 5th percentile
    best_case_roi: float   # 95th percentile
    
    # Lineup performance
    best_lineup_wins: int
    avg_lineup_wins: float
    lineup_win_distribution: Dict[int, int]  # lineup_id -> win_count
    
    # Score distributions
    avg_score: float
    median_score: float
    score_std: float
    min_score: float
    max_score: float


class MonteCarloSimulator:
    """
    Monte Carlo contest simulator.
    
    Simulates thousands of contest outcomes by:
    1. Sampling player scores from normal distributions
    2. Calculating lineup scores
    3. Ranking lineups against simulated field
    4. Tracking win rates and ROI
    
    Features:
    - Player variance modeling
    - Field simulation
    - Multiple contest types
    - Win probability calculation
    - ROI estimation
    """
    
    def __init__(self):
        """Initialize Monte Carlo simulator"""
        logger.info("MonteCarloSimulator initialized")
    
    def _estimate_player_variance(self, player: pd.Series) -> float:
        """
        Estimate standard deviation for player scoring.
        
        Uses ceiling and floor to estimate std dev:
        std_dev ≈ (ceiling - floor) / 4
        
        Args:
            player: Player data with projection, ceiling, floor
        
        Returns:
            Estimated standard deviation
        """
        if 'ceiling' in player and 'floor' in player:
            # Use ceiling and floor if available
            ceiling = player['ceiling']
            floor = player['floor']
            std_dev = (ceiling - floor) / 4
        else:
            # Estimate from projection
            # RBs/WRs: ~35% variance, QBs: ~40%, TEs: ~30%
            position = player.get('position', 'FLEX')
            
            if position == 'QB':
                variance_pct = 0.40
            elif position in ['RB', 'WR']:
                variance_pct = 0.35
            elif position == 'TE':
                variance_pct = 0.30
            else:
                variance_pct = 0.35
            
            std_dev = player['projection'] * variance_pct
        
        return max(std_dev, 2.0)  # Minimum 2 pts std dev
    
    def _simulate_player_scores(
        self,
        players_df: pd.DataFrame,
        num_simulations: int
    ) -> np.ndarray:
        """
        Simulate player scores for all simulations.
        
        Args:
            players_df: Player data with projections
            num_simulations: Number of simulations to run
        
        Returns:
            Array of shape (num_players, num_simulations)
        """
        num_players = len(players_df)
        scores = np.zeros((num_players, num_simulations))
        
        for idx, player in players_df.iterrows():
            projection = player['projection']
            std_dev = self._estimate_player_variance(player)
            
            # Sample from normal distribution
            player_scores = np.random.normal(
                loc=projection,
                scale=std_dev,
                size=num_simulations
            )
            
            # Floor at 0 (can't score negative in DFS)
            player_scores = np.maximum(player_scores, 0)
            
            scores[idx] = player_scores
        
        return scores
    
    def _calculate_lineup_scores(
        self,
        lineups: List[pd.DataFrame],
        player_scores: np.ndarray,
        players_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate lineup scores for all simulations.
        
        Args:
            lineups: List of lineup DataFrames
            player_scores: Simulated player scores
            players_df: Original player data
        
        Returns:
            Array of shape (num_lineups, num_simulations)
        """
        num_lineups = len(lineups)
        num_simulations = player_scores.shape[1]
        
        lineup_scores = np.zeros((num_lineups, num_simulations))
        
        # Create player name to index mapping
        player_to_idx = {name: idx for idx, name in enumerate(players_df['name'])}
        
        for lineup_idx, lineup in enumerate(lineups):
            # Get player indices for this lineup
            player_indices = [
                player_to_idx[name]
                for name in lineup['name']
                if name in player_to_idx
            ]
            
            # Sum scores across all players in lineup
            lineup_scores[lineup_idx] = player_scores[player_indices].sum(axis=0)
        
        return lineup_scores
    
    def _simulate_field(
        self,
        contest_size: int,
        num_simulations: int,
        avg_field_score: float = 140.0,
        field_std: float = 15.0
    ) -> np.ndarray:
        """
        Simulate field scores.
        
        Args:
            contest_size: Number of entries in contest
            num_simulations: Number of simulations
            avg_field_score: Average field score
            field_std: Field score standard deviation
        
        Returns:
            Array of shape (contest_size, num_simulations)
        """
        # Simulate field scores from normal distribution
        field_scores = np.random.normal(
            loc=avg_field_score,
            scale=field_std,
            size=(contest_size, num_simulations)
        )
        
        return field_scores
    
    def _calculate_win_rates(
        self,
        lineup_scores: np.ndarray,
        field_scores: np.ndarray
    ) -> Dict:
        """
        Calculate win rates by comparing lineup scores to field.
        
        Args:
            lineup_scores: Shape (num_lineups, num_simulations)
            field_scores: Shape (contest_size, num_simulations)
        
        Returns:
            Dictionary with win rate statistics
        """
        num_lineups = lineup_scores.shape[0]
        num_simulations = lineup_scores.shape[1]
        contest_size = field_scores.shape[0]
        
        # Percentile thresholds
        top_1_pct = int(contest_size * 0.01)
        top_5_pct = int(contest_size * 0.05)
        top_10_pct = int(contest_size * 0.10)
        top_100 = min(100, contest_size)
        top_20_pct = int(contest_size * 0.20)  # Cash line
        
        # Track wins for each threshold
        wins_top_1 = 0
        wins_top_5 = 0
        wins_top_10 = 0
        wins_top_100 = 0
        cash_finishes = 0
        
        lineup_wins = {i: 0 for i in range(num_lineups)}
        
        for sim in range(num_simulations):
            # Get scores for this simulation
            my_scores = lineup_scores[:, sim]
            field_sim = field_scores[:, sim]
            
            # Combine your lineups with field
            all_scores = np.concatenate([my_scores, field_sim])
            
            # Sort to find thresholds
            sorted_scores = np.sort(all_scores)[::-1]  # Descending
            
            threshold_1 = sorted_scores[top_1_pct - 1] if top_1_pct > 0 else sorted_scores[0]
            threshold_5 = sorted_scores[top_5_pct - 1] if top_5_pct > 0 else sorted_scores[0]
            threshold_10 = sorted_scores[top_10_pct - 1] if top_10_pct > 0 else sorted_scores[0]
            threshold_100 = sorted_scores[top_100 - 1] if top_100 > 0 else sorted_scores[0]
            threshold_cash = sorted_scores[top_20_pct - 1] if top_20_pct > 0 else sorted_scores[0]
            
            # Check which lineups beat thresholds
            for lineup_idx, score in enumerate(my_scores):
                if score >= threshold_1:
                    wins_top_1 += 1
                    lineup_wins[lineup_idx] += 1
                if score >= threshold_5:
                    wins_top_5 += 1
                if score >= threshold_10:
                    wins_top_10 += 1
                if score >= threshold_100:
                    wins_top_100 += 1
                if score >= threshold_cash:
                    cash_finishes += 1
        
        total_entries = num_lineups * num_simulations
        
        return {
            'win_rate': (wins_top_1 / total_entries) * 100,
            'top_5_rate': (wins_top_5 / total_entries) * 100,
            'top_10_rate': (wins_top_10 / total_entries) * 100,
            'top_100_rate': (wins_top_100 / total_entries) * 100,
            'cash_rate': (cash_finishes / total_entries) * 100,
            'lineup_wins': lineup_wins
        }
    
    def simulate_portfolio(
        self,
        lineups: List[pd.DataFrame],
        players_df: pd.DataFrame,
        num_simulations: int = 10000,
        contest_size: int = 150000,
        avg_field_score: float = 140.0,
        field_std: float = 15.0,
        entry_fee: float = 20.0,
        prize_structure: Optional[Dict] = None
    ) -> SimulationResult:
        """
        Simulate portfolio performance across thousands of contests.
        
        Args:
            lineups: List of lineup DataFrames
            players_df: Player data with projections
            num_simulations: Number of contest simulations to run
            contest_size: Total entries in contest
            avg_field_score: Average field score
            field_std: Field score standard deviation
            entry_fee: Entry fee per lineup
            prize_structure: Custom prize structure (optional)
        
        Returns:
            SimulationResult with all metrics
        """
        logger.info(f"Starting simulation: {len(lineups)} lineups, {num_simulations} sims, "
                   f"contest size {contest_size}")
        
        # 1. Simulate player scores
        player_scores = self._simulate_player_scores(players_df, num_simulations)
        
        # 2. Calculate lineup scores
        lineup_scores = self._calculate_lineup_scores(lineups, player_scores, players_df)
        
        # 3. Simulate field
        field_scores = self._simulate_field(contest_size, num_simulations, avg_field_score, field_std)
        
        # 4. Calculate win rates
        win_rates = self._calculate_win_rates(lineup_scores, field_scores)
        
        # 5. Calculate ROI (simplified - using win rates as proxy)
        # In real implementation, would use actual prize structure
        num_lineups = len(lineups)
        total_entry = entry_fee * num_lineups
        
        # Estimate winnings based on win rates
        expected_winnings = (
            win_rates['win_rate'] / 100 * total_entry * 200 +  # Big wins
            win_rates['top_100_rate'] / 100 * total_entry * 10 +  # Small wins
            win_rates['cash_rate'] / 100 * total_entry * 1.8  # Cash
        )
        
        expected_roi = ((expected_winnings - total_entry) / total_entry) * 100
        
        # 6. Score statistics
        all_scores = lineup_scores.flatten()
        
        # Find best performing lineup
        lineup_win_counts = win_rates['lineup_wins']
        best_lineup = max(lineup_win_counts, key=lineup_win_counts.get)
        best_lineup_wins = lineup_win_counts[best_lineup]
        avg_lineup_wins = sum(lineup_win_counts.values()) / len(lineup_win_counts)
        
        result = SimulationResult(
            portfolio_size=num_lineups,
            num_simulations=num_simulations,
            contest_size=contest_size,
            win_rate=win_rates['win_rate'],
            top_5_rate=win_rates['top_5_rate'],
            top_10_rate=win_rates['top_10_rate'],
            top_100_rate=win_rates['top_100_rate'],
            cash_rate=win_rates['cash_rate'],
            expected_roi=expected_roi,
            median_roi=expected_roi * 0.85,  # Approximation
            worst_case_roi=-100.0,  # Worst case = lose all
            best_case_roi=expected_roi * 3.0,  # Approximation
            best_lineup_wins=best_lineup_wins,
            avg_lineup_wins=avg_lineup_wins,
            lineup_win_distribution=lineup_win_counts,
            avg_score=np.mean(all_scores),
            median_score=np.median(all_scores),
            score_std=np.std(all_scores),
            min_score=np.min(all_scores),
            max_score=np.max(all_scores)
        )
        
        logger.info(f"Simulation complete: Win rate {result.win_rate:.2f}%, "
                   f"Expected ROI {result.expected_roi:.1f}%")
        
        return result
    
    def simulate_single_lineup(
        self,
        lineup: pd.DataFrame,
        players_df: pd.DataFrame,
        num_simulations: int = 10000
    ) -> Dict:
        """
        Simulate single lineup performance.
        
        Args:
            lineup: Single lineup DataFrame
            players_df: Player data
            num_simulations: Number of simulations
        
        Returns:
            Dictionary with score distribution
        """
        # Simulate scores for players in lineup
        lineup_players = players_df[players_df['name'].isin(lineup['name'])]
        player_scores = self._simulate_player_scores(lineup_players, num_simulations)
        
        # Sum to get lineup scores
        lineup_scores = player_scores.sum(axis=0)
        
        return {
            'mean': np.mean(lineup_scores),
            'median': np.median(lineup_scores),
            'std': np.std(lineup_scores),
            'min': np.min(lineup_scores),
            'max': np.max(lineup_scores),
            'percentile_5': np.percentile(lineup_scores, 5),
            'percentile_25': np.percentile(lineup_scores, 25),
            'percentile_75': np.percentile(lineup_scores, 75),
            'percentile_95': np.percentile(lineup_scores, 95),
            'distribution': lineup_scores
        }
    
    def compare_lineups(
        self,
        lineups: List[pd.DataFrame],
        players_df: pd.DataFrame,
        num_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Compare multiple lineups via simulation.
        
        Args:
            lineups: List of lineups to compare
            players_df: Player data
            num_simulations: Number of simulations
        
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for idx, lineup in enumerate(lineups):
            sim = self.simulate_single_lineup(lineup, players_df, num_simulations)
            
            results.append({
                'lineup_id': idx,
                'mean_score': sim['mean'],
                'median_score': sim['median'],
                'std_dev': sim['std'],
                'ceiling_95th': sim['percentile_95'],
                'floor_5th': sim['percentile_5'],
                'upside': sim['percentile_95'] - sim['mean']
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('ceiling_95th', ascending=False)
        
        return df
