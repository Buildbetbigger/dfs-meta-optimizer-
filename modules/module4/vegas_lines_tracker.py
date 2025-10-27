"""
Module 4: Vegas Lines Tracker
Tracks betting lines, spreads, totals, and implied team totals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class GameLine:
    """Represents betting line for a game"""
    game_id: str
    home_team: str
    away_team: str
    spread: float  # negative = home favored
    total: float
    home_ml: int  # money line
    away_ml: int
    timestamp: datetime


@dataclass
class LineMovement:
    """Represents a line movement"""
    game_id: str
    metric: str  # 'spread', 'total', 'home_ml', 'away_ml'
    old_value: float
    new_value: float
    change: float
    timestamp: datetime


class VegasLinesTracker:
    """
    Tracks Vegas betting lines and movements.
    
    Features:
    - Line movement tracking (spreads, totals, money lines)
    - Implied team totals calculation
    - Line movement alerts
    - Historical line data
    - Sharp money detection
    """
    
    def __init__(self):
        """Initialize Vegas lines tracker"""
        # Current lines
        self.current_lines: Dict[str, GameLine] = {}
        
        # Line history
        self.line_history: Dict[str, List[GameLine]] = {}
        
        # Movement tracking
        self.movements: List[LineMovement] = []
        
        logger.info("VegasLinesTracker initialized")
    
    def update_line(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        spread: float,
        total: float,
        home_ml: Optional[int] = None,
        away_ml: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Update betting line for a game.
        
        Args:
            game_id: Unique game identifier (e.g., 'KC@BUF')
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            spread: Point spread (negative = home favored)
            total: Over/under total points
            home_ml: Home money line (optional)
            away_ml: Away money line (optional)
            timestamp: When line was posted (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        new_line = GameLine(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            spread=spread,
            total=total,
            home_ml=home_ml or 0,
            away_ml=away_ml or 0,
            timestamp=timestamp
        )
        
        # Check for movements if line exists
        if game_id in self.current_lines:
            old_line = self.current_lines[game_id]
            self._detect_movements(old_line, new_line)
        
        # Update current line
        self.current_lines[game_id] = new_line
        
        # Add to history
        if game_id not in self.line_history:
            self.line_history[game_id] = []
        self.line_history[game_id].append(new_line)
        
        logger.info(f"Updated line for {game_id}: {spread}/{total}")
    
    def _detect_movements(self, old_line: GameLine, new_line: GameLine):
        """Detect significant line movements"""
        threshold = {
            'spread': 0.5,  # 0.5 point movement
            'total': 1.0,   # 1 point movement
            'ml': 10        # 10 cent movement
        }
        
        # Check spread movement
        if abs(new_line.spread - old_line.spread) >= threshold['spread']:
            movement = LineMovement(
                game_id=new_line.game_id,
                metric='spread',
                old_value=old_line.spread,
                new_value=new_line.spread,
                change=new_line.spread - old_line.spread,
                timestamp=new_line.timestamp
            )
            self.movements.append(movement)
            logger.info(f"Spread movement in {new_line.game_id}: {old_line.spread} -> {new_line.spread}")
        
        # Check total movement
        if abs(new_line.total - old_line.total) >= threshold['total']:
            movement = LineMovement(
                game_id=new_line.game_id,
                metric='total',
                old_value=old_line.total,
                new_value=new_line.total,
                change=new_line.total - old_line.total,
                timestamp=new_line.timestamp
            )
            self.movements.append(movement)
            logger.info(f"Total movement in {new_line.game_id}: {old_line.total} -> {new_line.total}")
    
    def get_implied_total(self, game_id: str, team: str) -> Optional[float]:
        """
        Calculate implied team total.
        
        Formula: 
        - If home: (total - spread) / 2
        - If away: (total + spread) / 2
        
        Args:
            game_id: Game identifier
            team: Team abbreviation
        
        Returns:
            Implied total or None if game not found
        """
        if game_id not in self.current_lines:
            return None
        
        line = self.current_lines[game_id]
        
        if team == line.home_team:
            # Home team: (total - spread) / 2
            return (line.total - line.spread) / 2
        elif team == line.away_team:
            # Away team: (total + spread) / 2
            return (line.total + line.spread) / 2
        else:
            logger.warning(f"Team {team} not found in game {game_id}")
            return None
    
    def get_all_implied_totals(self) -> Dict[str, float]:
        """
        Get implied totals for all teams.
        
        Returns:
            Dictionary of {team: implied_total}
        """
        implied_totals = {}
        
        for game_id, line in self.current_lines.items():
            home_total = self.get_implied_total(game_id, line.home_team)
            away_total = self.get_implied_total(game_id, line.away_team)
            
            if home_total:
                implied_totals[line.home_team] = home_total
            if away_total:
                implied_totals[line.away_team] = away_total
        
        return implied_totals
    
    def get_line_movements(
        self,
        game_id: Optional[str] = None,
        hours: int = 24,
        min_movement: float = 0.0
    ) -> List[LineMovement]:
        """
        Get recent line movements.
        
        Args:
            game_id: Filter by game (None = all games)
            hours: How many hours back to look
            min_movement: Minimum movement size
        
        Returns:
            List of LineMovements
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        movements = self.movements
        
        # Filter by time
        movements = [m for m in movements if m.timestamp >= cutoff]
        
        # Filter by game
        if game_id:
            movements = [m for m in movements if m.game_id == game_id]
        
        # Filter by movement size
        movements = [m for m in movements if abs(m.change) >= min_movement]
        
        # Sort by time (most recent first)
        movements.sort(key=lambda x: x.timestamp, reverse=True)
        
        return movements
    
    def get_sharp_money_indicators(self) -> List[Dict]:
        """
        Identify potential sharp money movements.
        
        Sharp money indicators:
        - Large spread movement (1+ point)
        - Line moving against public betting %
        - Reverse line movement
        
        Returns:
            List of games with sharp money indicators
        """
        sharp_indicators = []
        
        for game_id in self.current_lines:
            if game_id not in self.line_history or len(self.line_history[game_id]) < 2:
                continue
            
            history = self.line_history[game_id]
            opening_line = history[0]
            current_line = self.current_lines[game_id]
            
            spread_movement = current_line.spread - opening_line.spread
            total_movement = current_line.total - opening_line.total
            
            # Large spread movement
            if abs(spread_movement) >= 1.0:
                sharp_indicators.append({
                    'game_id': game_id,
                    'indicator': 'large_spread_movement',
                    'movement': spread_movement,
                    'opening_line': opening_line.spread,
                    'current_line': current_line.spread
                })
            
            # Large total movement
            if abs(total_movement) >= 2.0:
                sharp_indicators.append({
                    'game_id': game_id,
                    'indicator': 'large_total_movement',
                    'movement': total_movement,
                    'opening_line': opening_line.total,
                    'current_line': current_line.total
                })
        
        return sharp_indicators
    
    def get_vegas_summary(self) -> str:
        """
        Get text summary of current Vegas lines.
        
        Returns:
            Formatted summary string
        """
        if not self.current_lines:
            return "No Vegas lines available."
        
        summary = "Vegas Lines Summary\n"
        summary += f"Total games: {len(self.current_lines)}\n\n"
        
        # Sort by implied total
        implied_totals = self.get_all_implied_totals()
        
        for game_id, line in sorted(self.current_lines.items()):
            summary += f"{game_id}:\n"
            summary += f"  Spread: {line.spread:+.1f} (favors {line.home_team if line.spread < 0 else line.away_team})\n"
            summary += f"  Total: {line.total:.1f}\n"
            
            home_total = implied_totals.get(line.home_team, 0)
            away_total = implied_totals.get(line.away_team, 0)
            
            summary += f"  Implied: {line.home_team} {home_total:.1f}, {line.away_team} {away_total:.1f}\n"
            
            # Recent movements
            movements = self.get_line_movements(game_id=game_id, hours=24)
            if movements:
                summary += f"  Recent movements: {len(movements)}\n"
            
            summary += "\n"
        
        # Add sharp money indicators
        sharp = self.get_sharp_money_indicators()
        if sharp:
            summary += "Sharp Money Indicators:\n"
            for indicator in sharp:
                summary += f"  {indicator['game_id']}: {indicator['indicator']} ({indicator['movement']:+.1f})\n"
        
        return summary
    
    def update_player_game_environment(
        self,
        players_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Update player data with game environment from Vegas lines.
        
        Adds columns:
        - team_implied_total
        - game_total
        - is_favorite
        
        Args:
            players_df: Current player data with 'team' column
        
        Returns:
            Updated DataFrame
        """
        updated_df = players_df.copy()
        
        # Get implied totals
        implied_totals = self.get_all_implied_totals()
        
        # Initialize new columns
        updated_df['team_implied_total'] = 0.0
        updated_df['game_total'] = 0.0
        updated_df['is_favorite'] = False
        
        for idx, player in updated_df.iterrows():
            team = player.get('team', '')
            
            if not team:
                continue
            
            # Set implied total
            if team in implied_totals:
                updated_df.at[idx, 'team_implied_total'] = implied_totals[team]
            
            # Find game and set game total
            for game_id, line in self.current_lines.items():
                if team in [line.home_team, line.away_team]:
                    updated_df.at[idx, 'game_total'] = line.total
                    
                    # Determine if favorite
                    if team == line.home_team:
                        updated_df.at[idx, 'is_favorite'] = line.spread < 0
                    else:
                        updated_df.at[idx, 'is_favorite'] = line.spread > 0
                    
                    break
        
        logger.info("Updated player game environment from Vegas lines")
        
        return updated_df
    
    def get_game_script_prediction(self, game_id: str) -> Dict:
        """
        Predict game script based on Vegas lines.
        
        Args:
            game_id: Game identifier
        
        Returns:
            Dictionary with game script predictions
        """
        if game_id not in self.current_lines:
            return {}
        
        line = self.current_lines[game_id]
        
        home_implied = self.get_implied_total(game_id, line.home_team)
        away_implied = self.get_implied_total(game_id, line.away_team)
        
        # Determine game script
        spread_abs = abs(line.spread)
        
        if spread_abs >= 7.0:
            script = 'blowout_expected'
        elif spread_abs >= 3.0:
            script = 'moderate_favorite'
        else:
            script = 'close_game'
        
        # Determine pace
        if line.total >= 50.0:
            pace = 'high_scoring'
        elif line.total >= 45.0:
            pace = 'moderate_scoring'
        else:
            pace = 'low_scoring'
        
        return {
            'game_id': game_id,
            'spread': line.spread,
            'total': line.total,
            'home_implied': home_implied,
            'away_implied': away_implied,
            'script': script,
            'pace': pace,
            'favorite': line.home_team if line.spread < 0 else line.away_team,
            'underdog': line.away_team if line.spread < 0 else line.home_team
        }
    
    def clear_old_data(self, hours: int = 168):
        """
        Clear line data older than specified hours (default 1 week).
        
        Args:
            hours: Age threshold in hours
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Clear old movements
        self.movements = [m for m in self.movements if m.timestamp >= cutoff]
        
        # Clear old line history
        for game_id in list(self.line_history.keys()):
            self.line_history[game_id] = [
                line for line in self.line_history[game_id]
                if line.timestamp >= cutoff
            ]
            
            # Remove game if no history left
            if not self.line_history[game_id]:
                del self.line_history[game_id]
                if game_id in self.current_lines:
                    del self.current_lines[game_id]
        
        logger.info(f"Cleared Vegas data older than {hours} hours")
