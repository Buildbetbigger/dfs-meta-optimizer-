"""
Module 4: News Feed Monitor
Monitors news feeds for player updates, injuries, and lineup changes
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Represents a news item about a player"""
    player_name: str
    headline: str
    content: str
    source: str
    timestamp: datetime
    category: str  # 'injury', 'lineup', 'performance', 'other'
    severity: str  # 'critical', 'high', 'medium', 'low'
    impact_score: float  # 0-100
    tags: List[str]


class NewsFeedMonitor:
    """
    Monitors news feeds for DFS-relevant updates.
    
    Features:
    - News aggregation from multiple sources
    - Player mention detection
    - Impact severity classification
    - Injury status tracking
    - Lineup change detection
    - Real-time alerts
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize news feed monitor.
        
        Args:
            players_df: DataFrame with player data including 'name' column
        """
        self.players_df = players_df.copy()
        self.player_names = set(players_df['name'].values)
        
        # News storage
        self.news_items: List[NewsItem] = []
        self.player_news_index: Dict[str, List[NewsItem]] = {
            name: [] for name in self.player_names
        }
        
        # Injury status tracking
        self.injury_status: Dict[str, str] = {}  # player -> status
        
        # Keywords for classification
        self.injury_keywords = {
            'critical': ['out', 'dnp', 'ir', 'ruled out', 'season-ending', 'surgery'],
            'high': ['questionable', 'doubtful', 'limited', 'injury report'],
            'medium': ['probable', 'day-to-day', 'monitoring'],
            'low': ['full participant', 'cleared', 'practicing']
        }
        
        self.lineup_keywords = [
            'starting', 'starter', 'bench', 'rotation', 'snap count',
            'first team', 'depth chart', 'role change'
        ]
        
        self.performance_keywords = [
            'hot streak', 'slump', 'breakout', 'regression', 'trend',
            'target share', 'usage', 'touches', 'workload'
        ]
        
        logger.info(f"NewsFeedMonitor initialized with {len(self.player_names)} players")
    
    def add_news_item(
        self,
        player_name: str,
        headline: str,
        content: str,
        source: str = 'manual',
        timestamp: Optional[datetime] = None
    ) -> Optional[NewsItem]:
        """
        Add a news item manually.
        
        Args:
            player_name: Player's name
            headline: News headline
            content: Full news content
            source: News source identifier
            timestamp: When news was published (defaults to now)
        
        Returns:
            Created NewsItem or None if player not found
        """
        if player_name not in self.player_names:
            logger.warning(f"Player not found: {player_name}")
            return None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Classify the news
        category = self._classify_category(headline, content)
        severity = self._classify_severity(headline, content, category)
        impact_score = self._calculate_impact_score(category, severity, content)
        tags = self._extract_tags(headline, content)
        
        news_item = NewsItem(
            player_name=player_name,
            headline=headline,
            content=content,
            source=source,
            timestamp=timestamp,
            category=category,
            severity=severity,
            impact_score=impact_score,
            tags=tags
        )
        
        # Store news item
        self.news_items.append(news_item)
        self.player_news_index[player_name].append(news_item)
        
        # Update injury status if applicable
        if category == 'injury':
            self._update_injury_status(player_name, content, severity)
        
        logger.info(f"Added news for {player_name}: {category}/{severity} (impact: {impact_score})")
        
        return news_item
    
    def _classify_category(self, headline: str, content: str) -> str:
        """Classify news into category"""
        text = (headline + ' ' + content).lower()
        
        # Check for injury keywords
        if any(keyword in text for keywords in self.injury_keywords.values() 
               for keyword in keywords):
            return 'injury'
        
        # Check for lineup keywords
        if any(keyword in text for keyword in self.lineup_keywords):
            return 'lineup'
        
        # Check for performance keywords
        if any(keyword in text for keyword in self.performance_keywords):
            return 'performance'
        
        return 'other'
    
    def _classify_severity(self, headline: str, content: str, category: str) -> str:
        """Classify severity of news"""
        text = (headline + ' ' + content).lower()
        
        if category == 'injury':
            # Check severity based on injury keywords
            for severity, keywords in self.injury_keywords.items():
                if any(keyword in text for keyword in keywords):
                    return severity
        
        # Default severity based on category
        if category == 'injury':
            return 'medium'
        elif category == 'lineup':
            return 'high' if 'starting' in text else 'medium'
        elif category == 'performance':
            return 'medium'
        
        return 'low'
    
    def _calculate_impact_score(self, category: str, severity: str, content: str) -> float:
        """
        Calculate impact score (0-100).
        
        Higher scores = more important for DFS decisions
        """
        base_scores = {
            'injury': 70.0,
            'lineup': 60.0,
            'performance': 40.0,
            'other': 20.0
        }
        
        severity_multipliers = {
            'critical': 1.4,
            'high': 1.2,
            'medium': 1.0,
            'low': 0.7
        }
        
        score = base_scores[category] * severity_multipliers[severity]
        
        # Boost for high-impact words
        high_impact_words = ['out', 'ruled out', 'starting', 'benched', 'ir']
        content_lower = content.lower()
        boost = sum(5.0 for word in high_impact_words if word in content_lower)
        
        return min(100.0, score + boost)
    
    def _extract_tags(self, headline: str, content: str) -> List[str]:
        """Extract relevant tags from news"""
        tags = []
        text = (headline + ' ' + content).lower()
        
        # Add category tags
        if 'injury' in text or 'hurt' in text:
            tags.append('injury')
        if 'starting' in text:
            tags.append('starter')
        if 'bench' in text:
            tags.append('bench')
        if 'weather' in text:
            tags.append('weather')
        if 'questionable' in text or 'doubtful' in text:
            tags.append('game_time_decision')
        
        return tags
    
    def _update_injury_status(self, player_name: str, content: str, severity: str):
        """Update player injury status"""
        content_lower = content.lower()
        
        if severity == 'critical' or 'out' in content_lower or 'ruled out' in content_lower:
            self.injury_status[player_name] = 'OUT'
        elif 'doubtful' in content_lower:
            self.injury_status[player_name] = 'DOUBTFUL'
        elif 'questionable' in content_lower:
            self.injury_status[player_name] = 'QUESTIONABLE'
        elif 'probable' in content_lower:
            self.injury_status[player_name] = 'PROBABLE'
        elif severity == 'low' or 'cleared' in content_lower:
            self.injury_status[player_name] = 'ACTIVE'
    
    def get_player_news(
        self,
        player_name: str,
        hours: int = 24,
        min_impact: float = 0.0,
        categories: Optional[List[str]] = None
    ) -> List[NewsItem]:
        """
        Get recent news for a player.
        
        Args:
            player_name: Player's name
            hours: How many hours back to look
            min_impact: Minimum impact score
            categories: Filter by categories (None = all)
        
        Returns:
            List of NewsItems
        """
        if player_name not in self.player_news_index:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        news = self.player_news_index[player_name]
        
        # Filter by time
        news = [item for item in news if item.timestamp >= cutoff]
        
        # Filter by impact
        news = [item for item in news if item.impact_score >= min_impact]
        
        # Filter by category
        if categories:
            news = [item for item in news if item.category in categories]
        
        # Sort by timestamp (most recent first)
        news.sort(key=lambda x: x.timestamp, reverse=True)
        
        return news
    
    def get_injury_report(self) -> pd.DataFrame:
        """
        Get current injury report for all players.
        
        Returns:
            DataFrame with injury statuses
        """
        injury_data = []
        
        for player_name in self.player_names:
            status = self.injury_status.get(player_name, 'ACTIVE')
            
            # Get most recent injury news
            injury_news = self.get_player_news(
                player_name,
                hours=72,
                categories=['injury']
            )
            
            if status != 'ACTIVE' or injury_news:
                injury_data.append({
                    'player': player_name,
                    'status': status,
                    'recent_news_count': len(injury_news),
                    'latest_update': injury_news[0].timestamp if injury_news else None
                })
        
        return pd.DataFrame(injury_data)
    
    def get_critical_alerts(self, hours: int = 2) -> List[NewsItem]:
        """
        Get critical news items from recent hours.
        
        Args:
            hours: How many hours back to check
        
        Returns:
            List of critical NewsItems
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        critical = [
            item for item in self.news_items
            if item.timestamp >= cutoff and item.severity == 'critical'
        ]
        
        critical.sort(key=lambda x: x.timestamp, reverse=True)
        
        return critical
    
    def update_player_projections(
        self,
        players_df: pd.DataFrame,
        adjustment_factor: float = 0.15
    ) -> pd.DataFrame:
        """
        Update player projections based on recent news.
        
        Args:
            players_df: Current player projections
            adjustment_factor: How much to adjust (0.0-1.0)
        
        Returns:
            Updated DataFrame
        """
        updated_df = players_df.copy()
        
        for idx, player in updated_df.iterrows():
            player_name = player['name']
            
            # Get recent high-impact news
            recent_news = self.get_player_news(
                player_name,
                hours=24,
                min_impact=50.0
            )
            
            if not recent_news:
                continue
            
            # Calculate adjustment
            total_impact = sum(item.impact_score for item in recent_news)
            avg_impact = total_impact / len(recent_news)
            
            # Determine adjustment direction
            if any(item.severity == 'critical' for item in recent_news):
                # Critical news = reduce projection significantly
                adjustment = -adjustment_factor * (avg_impact / 100.0)
            elif any(tag in ['starter', 'increased_role'] for item in recent_news for tag in item.tags):
                # Positive news = increase projection
                adjustment = adjustment_factor * (avg_impact / 100.0) * 0.5
            else:
                # Neutral/negative = slight reduction
                adjustment = -adjustment_factor * (avg_impact / 100.0) * 0.3
            
            # Apply adjustment
            old_projection = player['projection']
            new_projection = old_projection * (1 + adjustment)
            
            # Don't let projection drop below floor
            floor = player.get('floor', old_projection * 0.3)
            new_projection = max(new_projection, floor)
            
            updated_df.at[idx, 'projection'] = new_projection
            
            logger.info(f"Updated {player_name} projection: {old_projection:.1f} -> {new_projection:.1f}")
        
        return updated_df
    
    def get_news_summary(self, hours: int = 24) -> str:
        """
        Get text summary of recent news.
        
        Args:
            hours: How many hours back
        
        Returns:
            Formatted summary string
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [item for item in self.news_items if item.timestamp >= cutoff]
        
        if not recent:
            return f"No news in the last {hours} hours."
        
        # Sort by impact
        recent.sort(key=lambda x: x.impact_score, reverse=True)
        
        summary = f"News Summary (Last {hours} hours)\n"
        summary += f"Total items: {len(recent)}\n\n"
        
        # Group by severity
        by_severity = {}
        for item in recent:
            if item.severity not in by_severity:
                by_severity[item.severity] = []
            by_severity[item.severity].append(item)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in by_severity:
                summary += f"{severity.upper()}: {len(by_severity[severity])} items\n"
                for item in by_severity[severity][:5]:  # Top 5
                    summary += f"  - {item.player_name}: {item.headline}\n"
                summary += "\n"
        
        return summary
    
    def clear_old_news(self, hours: int = 168):
        """
        Clear news older than specified hours (default 1 week).
        
        Args:
            hours: Age threshold in hours
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter news items
        self.news_items = [item for item in self.news_items if item.timestamp >= cutoff]
        
        # Rebuild player index
        self.player_news_index = {name: [] for name in self.player_names}
        for item in self.news_items:
            self.player_news_index[item.player_name].append(item)
        
        logger.info(f"Cleared news older than {hours} hours")
