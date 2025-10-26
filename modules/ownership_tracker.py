"""
Module 4: Ownership Tracker
Tracks and predicts player ownership in DFS contests
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OwnershipSnapshot:
    """Represents an ownership snapshot"""
    player_name: str
    ownership: float
    source: str  # 'predicted', 'actual', 'live'
    timestamp: datetime
    contest_type: str  # 'GPP', 'CASH', etc.


class OwnershipTracker:
    """
    Tracks and predicts player ownership percentages.
    
    Features:
    - Ownership prediction based on multiple factors
    - Live ownership tracking
    - Historical ownership data
    - Ownership trend analysis
    - Chalk vs contrarian identification
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize ownership tracker.
        
        Args:
            players_df: DataFrame with player data
        """
        self.players_df = players_df.copy()
        self.player_names = set(players_df['name'].values)
        
        # Ownership storage
        self.ownership_history: Dict[str, List[OwnershipSnapshot]] = {
            name: [] for name in self.player_names
        }
        
        # Current predicted ownership
        self.current_ownership: Dict[str, float] = {}
        
        logger.info(f"OwnershipTracker initialized with {len(self.player_names)} players")
    
    def predict_ownership(
        self,
        player_name: str,
        salary: float,
        projection: float,
        value: float,
        team_implied_total: Optional[float] = None,
        recent_performance: Optional[float] = None,
        injury_status: str = 'ACTIVE',
        news_impact: float = 0.0,
        contest_type: str = 'GPP'
    ) -> float:
        """
        Predict player ownership percentage.
        
        Factors considered:
        - Salary/projection value
        - Team implied total
        - Recent performance
        - Injury status
        - News/hype
        - Contest type
        
        Args:
            player_name: Player's name
            salary: Player's salary
            projection: Projected points
            value: Points per $1000 salary
            team_implied_total: Team's implied total from Vegas
            recent_performance: Recent average (last 3 games)
            injury_status: Current injury status
            news_impact: Impact score from recent news (0-100)
            contest_type: Contest type ('GPP' or 'CASH')
        
        Returns:
            Predicted ownership percentage (0-100)
        """
        # Base ownership from value
        if value >= 4.0:
            base_ownership = 30.0  # Elite value
        elif value >= 3.5:
            base_ownership = 20.0  # Great value
        elif value >= 3.0:
            base_ownership = 12.0  # Good value
        elif value >= 2.5:
            base_ownership = 7.0   # Decent value
        else:
            base_ownership = 3.0   # Poor value
        
        # Salary tier adjustment
        if salary >= 9000:
            salary_mult = 1.3  # Studs get more ownership
        elif salary >= 7000:
            salary_mult = 1.0
        elif salary >= 5000:
            salary_mult = 0.8
        else:
            salary_mult = 0.6  # Cheap players less owned unless value is great
        
        ownership = base_ownership * salary_mult
        
        # Team implied total boost
        if team_implied_total:
            if team_implied_total >= 28.0:
                ownership *= 1.4  # High-scoring game expected
            elif team_implied_total >= 24.0:
                ownership *= 1.2
            elif team_implied_total <= 17.0:
                ownership *= 0.7  # Low-scoring game
        
        # Recent performance boost (recency bias)
        if recent_performance:
            perf_ratio = recent_performance / projection
            if perf_ratio >= 1.3:
                ownership *= 1.3  # Hot streak = more ownership
            elif perf_ratio >= 1.1:
                ownership *= 1.15
            elif perf_ratio <= 0.7:
                ownership *= 0.8  # Slump = less ownership
        
        # Injury status impact
        if injury_status == 'QUESTIONABLE':
            ownership *= 0.6  # Questionable = less ownership
        elif injury_status == 'DOUBTFUL':
            ownership *= 0.2
        elif injury_status == 'OUT':
            ownership = 0.0
        
        # News/hype impact
        if news_impact > 0:
            hype_mult = 1 + (news_impact / 200.0)  # Up to 50% boost
            ownership *= hype_mult
        
        # Contest type adjustment
        if contest_type == 'CASH':
            # Cash games favor safer plays
            if value >= 3.0:
                ownership *= 1.5  # Value even more important in cash
        else:
            # GPP allows more variance
            ownership *= 1.0
        
        # Cap at reasonable maximum
        ownership = min(ownership, 90.0)
        
        # Floor
        ownership = max(ownership, 0.5 if injury_status == 'ACTIVE' else 0.0)
        
        return round(ownership, 1)
    
    def batch_predict_ownership(
        self,
        players_df: pd.DataFrame,
        contest_type: str = 'GPP',
        vegas_implied_totals: Optional[Dict[str, float]] = None,
        news_impacts: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Predict ownership for all players in dataframe.
        
        Args:
            players_df: Player data
            contest_type: Contest type
            vegas_implied_totals: Dict of team implied totals
            news_impacts: Dict of news impact scores by player
        
        Returns:
            DataFrame with 'ownership' column updated
        """
        updated_df = players_df.copy()
        
        vegas_implied_totals = vegas_implied_totals or {}
        news_impacts = news_impacts or {}
        
        for idx, player in updated_df.iterrows():
            player_name = player['name']
            
            # Get team implied total
            team = player.get('team', '')
            team_implied = vegas_implied_totals.get(team)
            
            # Get news impact
            news_impact = news_impacts.get(player_name, 0.0)
            
            # Calculate value if not present
            if 'value' not in player or pd.isna(player['value']):
                value = (player['projection'] / player['salary']) * 1000
            else:
                value = player['value']
            
            # Predict ownership
            ownership = self.predict_ownership(
                player_name=player_name,
                salary=player['salary'],
                projection=player['projection'],
                value=value,
                team_implied_total=team_implied,
                injury_status=player.get('injury_status', 'ACTIVE'),
                news_impact=news_impact,
                contest_type=contest_type
            )
            
            updated_df.at[idx, 'ownership'] = ownership
            
            # Store prediction
            self.update_ownership(
                player_name=player_name,
                ownership=ownership,
                source='predicted',
                contest_type=contest_type
            )
        
        logger.info(f"Predicted ownership for {len(updated_df)} players")
        
        return updated_df
    
    def update_ownership(
        self,
        player_name: str,
        ownership: float,
        source: str = 'predicted',
        contest_type: str = 'GPP',
        timestamp: Optional[datetime] = None
    ):
        """
        Update ownership for a player.
        
        Args:
            player_name: Player's name
            ownership: Ownership percentage
            source: Data source ('predicted', 'actual', 'live')
            contest_type: Contest type
            timestamp: When recorded (defaults to now)
        """
        if player_name not in self.ownership_history:
            logger.warning(f"Player not found: {player_name}")
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        snapshot = OwnershipSnapshot(
            player_name=player_name,
            ownership=ownership,
            source=source,
            timestamp=timestamp,
            contest_type=contest_type
        )
        
        self.ownership_history[player_name].append(snapshot)
        self.current_ownership[player_name] = ownership
        
        logger.debug(f"Updated {player_name} ownership: {ownership}% ({source})")
    
    def get_ownership_trends(
        self,
        player_name: str,
        hours: int = 24
    ) -> List[OwnershipSnapshot]:
        """
        Get ownership trend for a player.
        
        Args:
            player_name: Player's name
            hours: How many hours back to look
        
        Returns:
            List of OwnershipSnapshots
        """
        if player_name not in self.ownership_history:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        history = self.ownership_history[player_name]
        recent = [snap for snap in history if snap.timestamp >= cutoff]
        
        # Sort by timestamp
        recent.sort(key=lambda x: x.timestamp)
        
        return recent
    
    def identify_chalk_plays(
        self,
        ownership_threshold: float = 25.0,
        contest_type: str = 'GPP'
    ) -> List[Dict]:
        """
        Identify high-ownership (chalk) plays.
        
        Args:
            ownership_threshold: Minimum ownership % to be considered chalk
            contest_type: Contest type to filter by
        
        Returns:
            List of chalk plays with ownership data
        """
        chalk_plays = []
        
        for player_name, ownership in self.current_ownership.items():
            if ownership >= ownership_threshold:
                # Get most recent snapshot
                history = self.ownership_history[player_name]
                if history:
                    recent = history[-1]
                    
                    if recent.contest_type == contest_type:
                        chalk_plays.append({
                            'player': player_name,
                            'ownership': ownership,
                            'source': recent.source,
                            'timestamp': recent.timestamp
                        })
        
        # Sort by ownership descending
        chalk_plays.sort(key=lambda x: x['ownership'], reverse=True)
        
        return chalk_plays
    
    def identify_leverage_plays(
        self,
        players_df: pd.DataFrame,
        ownership_threshold: float = 15.0
    ) -> pd.DataFrame:
        """
        Identify low-ownership plays with high upside (leverage plays).
        
        Leverage score = ceiling / ownership
        
        Args:
            players_df: Player data with ceiling values
            ownership_threshold: Maximum ownership for leverage
        
        Returns:
            DataFrame sorted by leverage score
        """
        leverage_df = players_df.copy()
        
        # Filter to low ownership
        leverage_df = leverage_df[leverage_df['ownership'] <= ownership_threshold]
        
        # Calculate leverage score
        if 'ceiling' in leverage_df.columns:
            leverage_df['leverage_score'] = leverage_df['ceiling'] / (leverage_df['ownership'] + 0.1)
        else:
            # Use projection * 1.3 as ceiling estimate
            leverage_df['leverage_score'] = (leverage_df['projection'] * 1.3) / (leverage_df['ownership'] + 0.1)
        
        # Sort by leverage score
        leverage_df = leverage_df.sort_values('leverage_score', ascending=False)
        
        return leverage_df[['name', 'position', 'salary', 'projection', 'ownership', 'leverage_score']].head(20)
    
    def get_ownership_distribution(self) -> Dict:
        """
        Get distribution of ownership levels.
        
        Returns:
            Dictionary with ownership distribution stats
        """
        if not self.current_ownership:
            return {}
        
        ownerships = list(self.current_ownership.values())
        
        return {
            'mean': np.mean(ownerships),
            'median': np.median(ownerships),
            'std': np.std(ownerships),
            'min': min(ownerships),
            'max': max(ownerships),
            'high_owned': len([o for o in ownerships if o >= 25.0]),
            'medium_owned': len([o for o in ownerships if 10.0 <= o < 25.0]),
            'low_owned': len([o for o in ownerships if o < 10.0])
        }
    
    def get_ownership_summary(self) -> str:
        """
        Get text summary of ownership landscape.
        
        Returns:
            Formatted summary string
        """
        if not self.current_ownership:
            return "No ownership data available."
        
        summary = "Ownership Summary\n"
        summary += f"Total players: {len(self.current_ownership)}\n\n"
        
        # Distribution
        dist = self.get_ownership_distribution()
        summary += f"Distribution:\n"
        summary += f"  Mean: {dist['mean']:.1f}%\n"
        summary += f"  Median: {dist['median']:.1f}%\n"
        summary += f"  Range: {dist['min']:.1f}% - {dist['max']:.1f}%\n\n"
        
        # Ownership tiers
        summary += f"Ownership Tiers:\n"
        summary += f"  High (25%+): {dist['high_owned']} players\n"
        summary += f"  Medium (10-25%): {dist['medium_owned']} players\n"
        summary += f"  Low (<10%): {dist['low_owned']} players\n\n"
        
        # Top chalk
        chalk = self.identify_chalk_plays(ownership_threshold=20.0)
        if chalk:
            summary += "Top Chalk Plays:\n"
            for play in chalk[:10]:
                summary += f"  {play['player']}: {play['ownership']:.1f}%\n"
        
        return summary
    
    def update_with_actual_ownership(
        self,
        actual_ownership: Dict[str, float],
        source: str = 'actual'
    ):
        """
        Update with actual ownership data from a contest.
        
        Args:
            actual_ownership: Dictionary of {player_name: ownership%}
            source: Source identifier
        """
        for player_name, ownership in actual_ownership.items():
            self.update_ownership(
                player_name=player_name,
                ownership=ownership,
                source=source
            )
        
        logger.info(f"Updated {len(actual_ownership)} players with actual ownership")
    
    def calculate_prediction_accuracy(
        self,
        actual_ownership: Dict[str, float]
    ) -> Dict:
        """
        Calculate accuracy of ownership predictions vs actual.
        
        Args:
            actual_ownership: Dictionary of {player_name: actual_ownership%}
        
        Returns:
            Dictionary with accuracy metrics
        """
        errors = []
        abs_errors = []
        
        for player_name, actual_own in actual_ownership.items():
            if player_name not in self.ownership_history or not self.ownership_history[player_name]:
                continue
            
            # Get most recent prediction
            predicted_own = self.ownership_history[player_name][-1].ownership
            
            error = predicted_own - actual_own
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
    
    def clear_old_data(self, hours: int = 168):
        """
        Clear ownership data older than specified hours (default 1 week).
        
        Args:
            hours: Age threshold in hours
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        for player_name in self.ownership_history:
            self.ownership_history[player_name] = [
                snap for snap in self.ownership_history[player_name]
                if snap.timestamp >= cutoff
            ]
        
        logger.info(f"Cleared ownership data older than {hours} hours")
