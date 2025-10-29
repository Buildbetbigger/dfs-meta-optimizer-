"""
Module 4: Injury Tracker
Tracks player injury status and impact on DFS decisions
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class InjuryTracker:
    """
    Tracks NFL player injury status.
    
    Injury statuses (in order of severity):
    - OUT: Will not play (0% chance)
    - DOUBTFUL: Very unlikely to play (~25% chance)
    - QUESTIONABLE: Uncertain (50-75% chance)
    - PROBABLE: Likely to play (removed in 2016, but sometimes used)
    - HEALTHY: No injury designation
    
    Impact on DFS:
    - OUT players: Remove completely
    - DOUBTFUL: Avoid (too risky)
    - QUESTIONABLE: Reduce projection, monitor news
    - HEALTHY: Full projection
    """
    
    INJURY_STATUS_PRIORITY = {
        'OUT': 0,
        'DOUBTFUL': 1,
        'QUESTIONABLE': 2,
        'PROBABLE': 3,
        'HEALTHY': 4
    }
    
    INJURY_IMPACT = {
        'OUT': 0.0,          # 0% of projection
        'DOUBTFUL': 0.3,     # 30% of projection
        'QUESTIONABLE': 0.75,  # 75% of projection
        'PROBABLE': 0.95,    # 95% of projection
        'HEALTHY': 1.0       # 100% of projection
    }
    
    def __init__(self):
        """Initialize injury tracker"""
        self.injury_cache = {}
        logger.info("InjuryTracker initialized")
    
    def scrape_injury_report(self, source: str = 'fantasypros') -> pd.DataFrame:
        """
        Scrape injury report from source.
        
        Args:
            source: 'fantasypros', 'espn', or 'nfl'
        
        Returns:
            DataFrame with columns: player_name, team, position, status, injury
        """
        if source == 'fantasypros':
            return self._scrape_fantasypros()
        elif source == 'espn':
            return self._scrape_espn()
        elif source == 'nfl':
            return self._scrape_nfl()
        else:
            logger.error(f"Unknown source: {source}")
            return pd.DataFrame()
    
    def _scrape_fantasypros(self) -> pd.DataFrame:
        """
        Scrape FantasyPros injury report.
        
        URL: https://www.fantasypros.com/nfl/injury-report.php
        """
        try:
            url = "https://www.fantasypros.com/nfl/injury-report.php"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find injury table
            table = soup.find('table', {'id': 'injury-table'})
            
            if not table:
                logger.warning("Could not find injury table on FantasyPros")
                return self._get_sample_injuries()
            
            # Parse table rows
            injuries = []
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    injuries.append({
                        'player_name': cols[0].text.strip(),
                        'position': cols[1].text.strip(),
                        'team': cols[2].text.strip(),
                        'injury': cols[3].text.strip(),
                        'status': cols[4].text.strip().upper()
                    })
            
            df = pd.DataFrame(injuries)
            logger.info(f"Scraped {len(df)} injuries from FantasyPros")
            
            return df
            
        except Exception as e:
            logger.error(f"Error scraping FantasyPros: {e}")
            return self._get_sample_injuries()
    
    def _scrape_espn(self) -> pd.DataFrame:
        """
        Scrape ESPN injury report.
        
        URL: https://www.espn.com/nfl/injuries
        """
        try:
            url = "https://www.espn.com/nfl/injuries"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            injuries = []
            
            # ESPN structure: teams with injury lists
            team_sections = soup.find_all('div', class_='ResponsiveTable')
            
            for section in team_sections:
                # Get team name
                team_header = section.find_previous('div', class_='Table__Title')
                if team_header:
                    team = team_header.text.strip()
                else:
                    continue
                
                # Get injuries
                rows = section.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        injuries.append({
                            'player_name': cols[0].text.strip(),
                            'position': cols[1].text.strip(),
                            'injury': cols[2].text.strip(),
                            'status': cols[3].text.strip().upper(),
                            'team': self._normalize_team_name(team)
                        })
            
            df = pd.DataFrame(injuries)
            logger.info(f"Scraped {len(df)} injuries from ESPN")
            
            return df
            
        except Exception as e:
            logger.error(f"Error scraping ESPN: {e}")
            return self._get_sample_injuries()
    
    def _scrape_nfl(self) -> pd.DataFrame:
        """
        Scrape NFL.com injury report.
        
        URL: https://www.nfl.com/injuries/
        """
        try:
            url = "https://www.nfl.com/injuries/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # NFL.com might use dynamic loading - this is a simplified version
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This would need adjustment based on actual NFL.com structure
            logger.warning("NFL.com scraping not fully implemented - using fallback")
            return self._get_sample_injuries()
            
        except Exception as e:
            logger.error(f"Error scraping NFL.com: {e}")
            return self._get_sample_injuries()
    
    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name to abbreviation"""
        team_map = {
            'Arizona Cardinals': 'ARI',
            'Atlanta Falcons': 'ATL',
            'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF',
            'Carolina Panthers': 'CAR',
            'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE',
            'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN',
            'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC',
            'Las Vegas Raiders': 'LV',
            'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR',
            'Miami Dolphins': 'MIA',
            'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE',
            'New Orleans Saints': 'NO',
            'New York Giants': 'NYG',
            'New York Jets': 'NYJ',
            'Philadelphia Eagles': 'PHI',
            'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF',
            'Seattle Seahawks': 'SEA',
            'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN',
            'Washington Commanders': 'WAS'
        }
        
        return team_map.get(team, team)
    
    def _get_sample_injuries(self) -> pd.DataFrame:
        """
        Return sample injury data when scraping fails.
        
        This is a fallback - in production, you'd want to handle this better.
        """
        logger.warning("Using sample injury data - scraping failed")
        
        return pd.DataFrame({
            'player_name': [],
            'position': [],
            'team': [],
            'injury': [],
            'status': []
        })
    
    def add_injury_status_to_players(
        self,
        players_df: pd.DataFrame,
        injury_df: Optional[pd.DataFrame] = None,
        source: str = 'fantasypros'
    ) -> pd.DataFrame:
        """
        Add injury status to players DataFrame.
        
        Args:
            players_df: DataFrame with player data
            injury_df: Pre-fetched injury data (optional)
            source: Source to scrape if injury_df not provided
        
        Returns:
            DataFrame with injury columns added
        """
        df = players_df.copy()
        
        # Get injury data if not provided
        if injury_df is None:
            injury_df = self.scrape_injury_report(source)
        
        # Initialize injury columns
        df['injury_status'] = 'HEALTHY'
        df['injury_type'] = ''
        df['injury_impact'] = 1.0
        
        if injury_df.empty:
            logger.warning("No injury data available - all players marked HEALTHY")
            return df
        
        # Normalize names for matching
        df['name_normalized'] = df['name'].str.lower().str.strip()
        injury_df['player_normalized'] = injury_df['player_name'].str.lower().str.strip()
        
        # Match injuries to players
        matched = 0
        for idx, injury_row in injury_df.iterrows():
            player_name = injury_row['player_normalized']
            status = injury_row['status']
            injury_type = injury_row['injury']
            
            # Find matching player
            mask = df['name_normalized'] == player_name
            
            if mask.any():
                df.loc[mask, 'injury_status'] = status
                df.loc[mask, 'injury_type'] = injury_type
                df.loc[mask, 'injury_impact'] = self.INJURY_IMPACT.get(status, 1.0)
                matched += 1
        
        # Clean up
        df = df.drop('name_normalized', axis=1)
        
        logger.info(f"Matched {matched} injured players")
        
        # Log critical injuries
        out_players = df[df['injury_status'] == 'OUT']
        if not out_players.empty:
            logger.warning(f"⚠️ {len(out_players)} players OUT:")
            for _, player in out_players.iterrows():
                logger.warning(f"  {player['name']} ({player['position']}/{player['team']}) - {player['injury_type']}")
        
        questionable = df[df['injury_status'] == 'QUESTIONABLE']
        if not questionable.empty:
            logger.info(f"ℹ️ {len(questionable)} players QUESTIONABLE")
        
        return df
    
    def adjust_projections_for_injury(
        self,
        players_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adjust player projections based on injury status.
        
        Args:
            players_df: DataFrame with injury_status and injury_impact columns
        
        Returns:
            DataFrame with adjusted projections
        """
        df = players_df.copy()
        
        # Store original projections
        if 'projection_original' not in df.columns:
            df['projection_original'] = df['projection']
        
        # Adjust projections based on injury impact
        df['projection'] = df['projection_original'] * df['injury_impact']
        
        # Adjust ceiling/floor if present
        if 'ceiling' in df.columns:
            if 'ceiling_original' not in df.columns:
                df['ceiling_original'] = df['ceiling']
            df['ceiling'] = df['ceiling_original'] * df['injury_impact']
        
        if 'floor' in df.columns:
            if 'floor_original' not in df.columns:
                df['floor_original'] = df['floor']
            df['floor'] = df['floor_original'] * df['injury_impact']
        
        # Log adjustments
        adjusted = df[df['injury_impact'] < 1.0]
        if not adjusted.empty:
            logger.info(f"Adjusted projections for {len(adjusted)} injured players")
            
            for _, player in adjusted.head(10).iterrows():  # Show top 10
                original = player['projection_original']
                new = player['projection']
                logger.info(f"  {player['name']}: {original:.1f} → {new:.1f} pts "
                          f"({player['injury_status']})")
        
        return df
    
    def get_injury_report(self, players_df: pd.DataFrame) -> str:
        """
        Generate text injury report.
        
        Args:
            players_df: DataFrame with injury data
        
        Returns:
            Formatted injury report string
        """
        report = "=" * 60 + "\n"
        report += "INJURY REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Group by status
        for status in ['OUT', 'DOUBTFUL', 'QUESTIONABLE']:
            injured = players_df[players_df['injury_status'] == status]
            
            if not injured.empty:
                if status == 'OUT':
                    emoji = "🔴"
                elif status == 'DOUBTFUL':
                    emoji = "🟠"
                else:
                    emoji = "🟡"
                
                report += f"{emoji} {status} ({len(injured)} players):\n"
                
                for _, player in injured.iterrows():
                    report += f"  {player['name']} ({player['position']}/{player['team']}) - {player['injury_type']}\n"
                
                report += "\n"
        
        # Healthy count
        healthy = players_df[players_df['injury_status'] == 'HEALTHY']
        report += f"✅ HEALTHY: {len(healthy)} players\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report
    
    def filter_healthy_players(
        self,
        players_df: pd.DataFrame,
        exclude_statuses: List[str] = ['OUT', 'DOUBTFUL']
    ) -> pd.DataFrame:
        """
        Filter out injured players.
        
        Args:
            players_df: DataFrame with injury_status column
            exclude_statuses: List of statuses to exclude
        
        Returns:
            Filtered DataFrame
        """
        original_count = len(players_df)
        
        df = players_df[~players_df['injury_status'].isin(exclude_statuses)].copy()
        
        removed = original_count - len(df)
        if removed > 0:
            logger.info(f"Filtered out {removed} injured players")
        
        return df


# Helper function for quick integration
def add_injury_data(
    players_df: pd.DataFrame,
    source: str = 'fantasypros',
    adjust_projections: bool = True,
    filter_out: bool = True
) -> pd.DataFrame:
    """
    Quick helper to add injury data to players.
    
    Args:
        players_df: Players DataFrame
        source: Injury data source
        adjust_projections: Whether to adjust projections for injury
        filter_out: Whether to remove OUT/DOUBTFUL players
    
    Returns:
        DataFrame with injury data and adjustments
    """
    tracker = InjuryTracker()
    
    # Add injury status
    df = tracker.add_injury_status_to_players(players_df, source=source)
    
    # Adjust projections if requested
    if adjust_projections:
        df = tracker.adjust_projections_for_injury(df)
    
    # Filter out injured if requested
    if filter_out:
        df = tracker.filter_healthy_players(df)
    
    return df
