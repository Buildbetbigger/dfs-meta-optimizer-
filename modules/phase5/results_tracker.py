"""
DFS Meta-Optimizer v7.0.0 - Results Tracker Module
Historical performance tracking and analysis

Features:
- Contest result logging
- Performance trend analysis
- Strategy effectiveness comparison
- Ownership accuracy tracking
- Weekly/seasonal statistics
- CSV import/export
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ResultsTracker:
    """
    Tracks contest results over time for self-improvement.
    
    NEW v7.0.0 Features:
    - Historical performance database
    - Trend analysis
    - Strategy comparison
    - Ownership accuracy metrics
    - Learning from past results
    """
    
    def __init__(self):
        """Initialize results tracker"""
        self.contests: List[Dict] = []
        logger.info("ResultsTracker v7.0.0 initialized")
    
    def log_contest(
        self,
        week: int,
        contest_name: str,
        contest_type: str,
        num_lineups: int,
        entry_fee: float,
        total_winnings: float,
        finish_position: Optional[int] = None,
        contest_size: Optional[int] = None,
        actual_scores: Optional[Dict[str, float]] = None,
        actual_ownership: Optional[Dict[str, float]] = None,
        notes: str = ""
    ):
        """
        Log a contest result.
        
        Args:
            week: Week number
            contest_name: Name of contest
            contest_type: Type (GPP, CASH, etc.)
            num_lineups: Number of lineups entered
            entry_fee: Entry fee per lineup
            total_winnings: Total amount won
            finish_position: Your best finish
            contest_size: Total entries in contest
            actual_scores: Dict of {player_name: actual_score}
            actual_ownership: Dict of {player_name: actual_ownership%}
            notes: Optional notes about contest
        """
        total_entry = entry_fee * num_lineups
        profit = total_winnings - total_entry
        roi = (profit / total_entry) * 100 if total_entry > 0 else 0
        
        contest_record = {
            'timestamp': datetime.now(),
            'week': week,
            'contest_name': contest_name,
            'contest_type': contest_type,
            'num_lineups': num_lineups,
            'entry_fee': entry_fee,
            'total_entry': total_entry,
            'total_winnings': total_winnings,
            'profit': profit,
            'roi': roi,
            'finish_position': finish_position,
            'contest_size': contest_size,
            'actual_scores': actual_scores,
            'actual_ownership': actual_ownership,
            'notes': notes
        }
        
        self.contests.append(contest_record)
        
        logger.info(f"Logged contest: Week {week}, {contest_name}, "
                   f"ROI: {roi:.1f}%, Profit: ${profit:.2f}")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics across all contests.
        
        Returns:
            Dictionary with overall stats
        """
        if not self.contests:
            return {}
        
        df = pd.DataFrame(self.contests)
        
        total_contests = len(self.contests)
        total_entry = df['total_entry'].sum()
        total_winnings = df['total_winnings'].sum()
        total_profit = df['profit'].sum()
        overall_roi = (total_profit / total_entry) * 100 if total_entry > 0 else 0
        
        winning_contests = len(df[df['profit'] > 0])
        win_rate = (winning_contests / total_contests) * 100
        
        return {
            'total_contests': total_contests,
            'total_invested': total_entry,
            'total_winnings': total_winnings,
            'total_profit': total_profit,
            'overall_roi': overall_roi,
            'winning_contests': winning_contests,
            'losing_contests': total_contests - winning_contests,
            'win_rate': win_rate,
            'avg_roi_per_contest': df['roi'].mean(),
            'best_roi': df['roi'].max(),
            'worst_roi': df['roi'].min(),
            'median_roi': df['roi'].median()
        }
    
    def get_trends(self, last_n_weeks: Optional[int] = None) -> pd.DataFrame:
        """
        Get weekly performance trends.
        
        NEW v7.0.0: Track improvement over time!
        
        Args:
            last_n_weeks: Show only last N weeks (None = all)
        
        Returns:
            DataFrame with weekly stats
        """
        if not self.contests:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.contests)
        
        # Group by week
        weekly = df.groupby('week').agg({
            'total_entry': 'sum',
            'total_winnings': 'sum',
            'profit': 'sum',
            'roi': 'mean',
            'num_lineups': 'sum'
        }).reset_index()
        
        # Filter to last N weeks
        if last_n_weeks:
            weekly = weekly.tail(last_n_weeks)
        
        # Add cumulative stats
        weekly['cumulative_profit'] = weekly['profit'].cumsum()
        
        return weekly
    
    def analyze_ownership_accuracy(
        self,
        predicted_ownership: Dict[str, float]
    ) -> Dict:
        """
        Analyze ownership prediction accuracy.
        
        NEW v7.0.0: Learn from ownership predictions!
        
        Compares predicted vs actual ownership from logged contests.
        
        Args:
            predicted_ownership: Dict of {player_name: predicted_ownership%}
        
        Returns:
            Dictionary with accuracy metrics
        """
        errors = []
        abs_errors = []
        
        # Compare against all contests with actual ownership data
        for contest in self.contests:
            actual_own = contest.get('actual_ownership')
            if not actual_own:
                continue
            
            for player_name, actual in actual_own.items():
                if player_name in predicted_ownership:
                    predicted = predicted_ownership[player_name]
                    
                    error = predicted - actual
                    abs_error = abs(error)
                    
                    errors.append(error)
                    abs_errors.append(abs_error)
        
        if not errors:
            return {}
        
        return {
            'mean_error': np.mean(errors),
            'mean_absolute_error': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean([e**2 for e in errors])),
            'max_error': max(abs_errors),
            'num_predictions': len(errors)
        }
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare performance by contest type.
        
        NEW v7.0.0: Which strategy works best?
        
        Returns:
            DataFrame with strategy comparison
        """
        if not self.contests:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.contests)
        
        # Group by contest type
        comparison = df.groupby('contest_type').agg({
            'total_entry': 'sum',
            'total_winnings': 'sum',
            'profit': 'sum',
            'roi': 'mean',
            'contest_name': 'count'
        }).reset_index()
        
        comparison.rename(columns={'contest_name': 'num_contests'}, inplace=True)
        
        return comparison
    
    def get_best_weeks(self, n: int = 5) -> pd.DataFrame:
        """
        Get best performing weeks.
        
        Args:
            n: Number of weeks to return
        
        Returns:
            DataFrame with top weeks
        """
        trends = self.get_trends()
        
        if trends.empty:
            return pd.DataFrame()
        
        best = trends.nlargest(n, 'roi')
        
        return best
    
    def get_worst_weeks(self, n: int = 5) -> pd.DataFrame:
        """
        Get worst performing weeks.
        
        Args:
            n: Number of weeks to return
        
        Returns:
            DataFrame with worst weeks
        """
        trends = self.get_trends()
        
        if trends.empty:
            return pd.DataFrame()
        
        worst = trends.nsmallest(n, 'roi')
        
        return worst
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report.
        
        Returns:
            Formatted report string
        """
        if not self.contests:
            return "No contest data logged yet."
        
        stats = self.get_summary_stats()
        
        report = "=" * 70 + "\n"
        report += "DFS PERFORMANCE REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += "OVERALL STATISTICS:\n"
        report += f"  Total Contests: {stats['total_contests']}\n"
        report += f"  Total Invested: ${stats['total_invested']:.2f}\n"
        report += f"  Total Winnings: ${stats['total_winnings']:.2f}\n"
        report += f"  Total Profit: ${stats['total_profit']:.2f}\n"
        report += f"  Overall ROI: {stats['overall_roi']:.1f}%\n"
        report += f"  Winning Contests: {stats['winning_contests']} "
        report += f"({stats['win_rate']:.1f}%)\n"
        report += f"  Losing Contests: {stats['losing_contests']}\n\n"
        
        report += "PERFORMANCE METRICS:\n"
        report += f"  Average ROI: {stats['avg_roi_per_contest']:.1f}%\n"
        report += f"  Best ROI: {stats['best_roi']:.1f}%\n"
        report += f"  Worst ROI: {stats['worst_roi']:.1f}%\n"
        report += f"  Median ROI: {stats['median_roi']:.1f}%\n\n"
        
        # Weekly trends
        trends = self.get_trends(last_n_weeks=5)
        if not trends.empty:
            report += "RECENT WEEKS (Last 5):\n"
            for _, week in trends.iterrows():
                report += f"  Week {int(week['week'])}: "
                report += f"ROI {week['roi']:.1f}%, Profit ${week['profit']:.2f}\n"
            report += "\n"
        
        # Strategy comparison
        strategy_comp = self.compare_strategies()
        if not strategy_comp.empty:
            report += "BY CONTEST TYPE:\n"
            for _, strat in strategy_comp.iterrows():
                report += f"  {strat['contest_type']}: "
                report += f"{strat['num_contests']} contests, "
                report += f"ROI {strat['roi']:.1f}%\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report
    
    def export_to_csv(self, filepath: str):
        """
        Export contest data to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.contests:
            logger.warning("No data to export")
            return
        
        df = pd.DataFrame(self.contests)
        
        # Remove nested dict columns for CSV
        df = df.drop(columns=['actual_scores', 'actual_ownership'], errors='ignore')
        
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(self.contests)} contests to {filepath}")
    
    def import_from_csv(self, filepath: str):
        """
        Import contest data from CSV.
        
        Args:
            filepath: Path to CSV file
        """
        df = pd.read_csv(filepath)
        
        # Convert to contest records
        for _, row in df.iterrows():
            contest_record = row.to_dict()
            contest_record['timestamp'] = pd.to_datetime(contest_record['timestamp'])
            self.contests.append(contest_record)
        
        logger.info(f"Imported {len(df)} contests from {filepath}")
    
    def get_learning_insights(self) -> Dict:
        """
        Generate insights for self-improvement.
        
        NEW v7.0.0: System learns from past performance!
        
        Returns:
            Dictionary with actionable insights
        """
        if len(self.contests) < 5:
            return {'insight': 'Need more contest data to generate insights'}
        
        stats = self.get_summary_stats()
        trends = self.get_trends()
        strategy_comp = self.compare_strategies()
        
        insights = []
        
        # ROI trend
        if not trends.empty and len(trends) >= 3:
            recent_roi = trends.tail(3)['roi'].mean()
            overall_roi = stats['overall_roi']
            
            if recent_roi > overall_roi + 5:
                insights.append("✅ Performance improving! Recent ROI exceeds historical average")
            elif recent_roi < overall_roi - 5:
                insights.append("⚠️ Performance declining. Recent ROI below historical average")
        
        # Strategy effectiveness
        if not strategy_comp.empty:
            best_strategy = strategy_comp.nlargest(1, 'roi').iloc[0]
            insights.append(
                f"💡 Best performing strategy: {best_strategy['contest_type']} "
                f"({best_strategy['roi']:.1f}% ROI)"
            )
        
        # Win rate
        if stats['win_rate'] < 30:
            insights.append("📊 Win rate below 30% - consider more conservative plays")
        elif stats['win_rate'] > 60:
            insights.append("🎯 High win rate! Portfolio well-calibrated")
        
        return {
            'insights': insights,
            'overall_roi': stats['overall_roi'],
            'total_profit': stats['total_profit'],
            'best_strategy': best_strategy['contest_type'] if not strategy_comp.empty else 'Unknown'
        }
