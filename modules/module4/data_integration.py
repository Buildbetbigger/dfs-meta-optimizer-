"""
DFS Meta-Optimizer v7.0.1 - Data Enrichment Module
Integrates weather and injury data with existing Vegas lines system

NEW IN v7.0.1:
- Weather impact analysis (wind, rain, temperature)
- Injury status tracking (OUT/DOUBTFUL/QUESTIONABLE)
- Projection adjustments based on conditions
- Dome game detection
- Position-specific weather impact

Integrates with v6.3.0 VegasLinesTracker (already in opponent_modeling.py)
"""

import pandas as pd
from typing import Optional, Dict
import logging

# v7.0.0 import style (no modules/ prefix)
from weather_data import WeatherDataProvider
from injury_tracker import InjuryTracker
from opponent_modeling import VegasLinesTracker

logger = logging.getLogger(__name__)


class DataEnrichment:
    """
    Master enrichment system combining all external data.
    
    Integrates:
    1. Vegas lines (via existing VegasLinesTracker)
    2. Weather data (NEW - wind, rain, temperature)
    3. Injury reports (NEW - OUT/DOUBTFUL/QUESTIONABLE)
    
    Compatible with DFS Meta-Optimizer v7.0.0
    """
    
    def __init__(
        self,
        weather_api_key: Optional[str] = None,
        injury_source: str = 'fantasypros'
    ):
        """
        Initialize data enrichment system.
        
        Args:
            weather_api_key: OpenWeatherMap API key (optional, free tier)
            injury_source: 'fantasypros', 'espn', or 'nfl'
        """
        # Use existing v6.3.0 vegas system
        self.vegas_tracker = VegasLinesTracker()
        
        # Add new v7.0.1 systems
        self.weather_provider = WeatherDataProvider(weather_api_key)
        self.injury_tracker = InjuryTracker()
        self.injury_source = injury_source
        
        logger.info("DataEnrichment v7.0.1 initialized")
    
    def enrich_players(
        self,
        players_df: pd.DataFrame,
        add_vegas: bool = True,
        add_weather: bool = True,
        add_injuries: bool = True,
        adjust_projections: bool = True,
        filter_injured: bool = True
    ) -> Dict:
        """
        Enrich players with all external data.
        
        Args:
            players_df: Base player data
            add_vegas: Add Vegas lines (spreads, totals)
            add_weather: Add weather data
            add_injuries: Add injury status
            adjust_projections: Auto-adjust projections
            filter_injured: Remove OUT/DOUBTFUL players
        
        Returns:
            Dict with:
            - enriched_df: Enhanced player DataFrame
            - reports: Text reports for each data source
            - summary: Overall summary statistics
        """
        df = players_df.copy()
        
        reports = {
            'vegas': '',
            'weather': '',
            'injury': ''
        }
        
        logger.info(f"Enriching {len(df)} players with external data")
        
        # 1. Vegas Lines (existing v6.3.0 system)
        if add_vegas:
            logger.info("Adding Vegas lines...")
            vegas_lines = self.vegas_tracker.get_current_lines()
            
            if not vegas_lines.empty:
                df = self._add_vegas_data(df, vegas_lines)
                reports['vegas'] = self._generate_vegas_report(df)
                logger.info("✅ Vegas lines added")
            else:
                logger.warning("⚠️ No Vegas lines available")
        
        # 2. Weather Data (NEW v7.0.1)
        if add_weather:
            logger.info("Adding weather data...")
            df = self.weather_provider.add_weather_to_players(df)
            reports['weather'] = self.weather_provider.get_weather_report(df)
            logger.info("✅ Weather data added")
        
        # 3. Injury Status (NEW v7.0.1)
        if add_injuries:
            logger.info("Fetching injury reports...")
            injury_report = self.injury_tracker.scrape_injury_report(self.injury_source)
            
            if not injury_report.empty:
                df = self.injury_tracker.add_injury_status_to_players(df, injury_report)
                
                if adjust_projections:
                    df = self.injury_tracker.adjust_projections_for_injury(df)
                
                if filter_injured:
                    original_count = len(df)
                    df = self.injury_tracker.filter_healthy_players(df)
                    logger.info(f"Filtered {original_count - len(df)} injured players")
                
                reports['injury'] = self.injury_tracker.get_injury_report(
                    df if not filter_injured else players_df
                )
                logger.info("✅ Injury data added")
            else:
                logger.warning("⚠️ No injury data available")
        
        # 4. Calculate composite adjustments
        if adjust_projections:
            df = self._calculate_composite_adjustments(df)
        
        logger.info(f"Data enrichment complete: {len(df)} players")
        
        return {
            'enriched_df': df,
            'reports': reports,
            'summary': self._generate_summary(df, reports)
        }
    
    def _add_vegas_data(
        self,
        df: pd.DataFrame,
        vegas_lines: pd.DataFrame
    ) -> pd.DataFrame:
        """Add Vegas lines to players DataFrame"""
        
        # Create team to Vegas mapping
        vegas_map = {}
        for _, row in vegas_lines.iterrows():
            vegas_map[row['home_team']] = {
                'spread': row['home_spread'],
                'total': row['total'],
                'implied_total': row['home_implied']
            }
            vegas_map[row['away_team']] = {
                'spread': row['away_spread'],
                'total': row['total'],
                'implied_total': row['away_implied']
            }
        
        # Add to players
        df['vegas_spread'] = df['team'].map(
            lambda t: vegas_map.get(t, {}).get('spread', 0)
        )
        df['vegas_total'] = df['team'].map(
            lambda t: vegas_map.get(t, {}).get('total', 46)
        )
        df['vegas_implied_total'] = df['team'].map(
            lambda t: vegas_map.get(t, {}).get('implied_total', 23)
        )
        
        return df
    
    def _calculate_composite_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final projection adjustments.
        
        Combines:
        - Vegas implied total (game environment)
        - Weather impact (conditions)
        - Injury status (availability)
        """
        df = df.copy()
        
        # Store original projections
        if 'projection_base' not in df.columns:
            df['projection_base'] = df.get('projection_original', df['projection'])
        
        # Initialize adjustment factor
        df['total_adjustment'] = 1.0
        
        # Vegas adjustment (game environment)
        if 'vegas_implied_total' in df.columns:
            # Normalize to league average (23 points)
            # 28 pts = 1.2x, 23 pts = 1.0x, 18 pts = 0.8x
            df['vegas_adjustment'] = (df['vegas_implied_total'] / 23.0).clip(0.7, 1.3)
            df['total_adjustment'] *= df['vegas_adjustment']
        
        # Weather adjustment
        if 'weather_impact' in df.columns:
            # weather_impact is 0-100, convert to multiplier
            df['weather_adjustment'] = df['weather_impact'] / 100.0
            df['total_adjustment'] *= df['weather_adjustment']
        
        # Injury adjustment (already in injury_impact column)
        if 'injury_impact' in df.columns:
            df['total_adjustment'] *= df['injury_impact']
        
        # Apply adjustments
        df['projection_adjusted'] = df['projection_base'] * df['total_adjustment']
        df['projection'] = df['projection_adjusted']
        
        # Adjust ceiling/floor
        if 'ceiling' in df.columns:
            if 'ceiling_base' not in df.columns:
                df['ceiling_base'] = df.get('ceiling_original', df['ceiling'])
            df['ceiling'] = df['ceiling_base'] * df['total_adjustment']
        
        if 'floor' in df.columns:
            if 'floor_base' not in df.columns:
                df['floor_base'] = df.get('floor_original', df['floor'])
            df['floor'] = df['floor_base'] * df['total_adjustment']
        
        # Log significant adjustments
        big_adj = df[
            (df['total_adjustment'] > 1.15) | 
            (df['total_adjustment'] < 0.85)
        ]
        
        if not big_adj.empty:
            logger.info(f"Found {len(big_adj)} players with significant adjustments:")
            for _, player in big_adj.head(10).iterrows():
                base = player.get('projection_base', player['projection'])
                adjusted = player['projection_adjusted']
                change = (adjusted / base - 1) * 100 if base > 0 else 0
                
                logger.info(
                    f"  {player['name']}: {base:.1f} → {adjusted:.1f} "
                    f"({change:+.0f}%)"
                )
        
        return df
    
    def _generate_vegas_report(self, df: pd.DataFrame) -> str:
        """Generate Vegas lines report"""
        report = "=" * 60 + "\n"
        report += "VEGAS LINES REPORT\n"
        report += "=" * 60 + "\n\n"
        
        if 'vegas_implied_total' not in df.columns:
            return report + "No Vegas data available\n"
        
        # High-scoring games (team totals > 26)
        high_scoring = df[df['vegas_implied_total'] > 26].groupby('team').first()
        if not high_scoring.empty:
            report += "🔥 HIGH-SCORING ENVIRONMENTS (Team Total > 26):\n"
            for team, row in high_scoring.iterrows():
                report += f"  {team}: {row['vegas_implied_total']:.1f} points "
                report += f"(O/U {row['vegas_total']:.1f})\n"
            report += "\n"
        
        # Heavy favorites (spread < -7)
        favorites = df[df['vegas_spread'] < -7].groupby('team').first()
        if not favorites.empty:
            report += "💪 HEAVY FAVORITES (Spread < -7):\n"
            for team, row in favorites.iterrows():
                report += f"  {team}: {row['vegas_spread']:.1f}\n"
            report += "\n"
        
        report += "=" * 60 + "\n"
        return report
    
    def _generate_summary(self, df: pd.DataFrame, reports: Dict) -> str:
        """Generate comprehensive summary"""
        summary = "=" * 70 + "\n"
        summary += "DATA ENRICHMENT SUMMARY\n"
        summary += "=" * 70 + "\n\n"
        
        summary += f"Total Players: {len(df)}\n\n"
        
        # Vegas summary
        if 'vegas_implied_total' in df.columns:
            avg_total = df['vegas_implied_total'].mean()
            summary += f"Average Vegas Total: {avg_total:.1f} points\n"
            
            favorites = df[df['vegas_spread'] < -3.0]
            summary += f"Heavy Favorites (<-3): {len(favorites)} teams\n\n"
        
        # Weather summary
        if 'weather_wind' in df.columns:
            bad_weather = df[
                (df['weather_wind'] > 15) | 
                (df['weather_temp'] < 32)
            ]
            summary += f"Players Affected by Bad Weather: {len(bad_weather)}\n"
            
            dome_players = df[df.get('is_dome', False)]
            summary += f"Players in Dome Games: {len(dome_players)}\n\n"
        
        # Injury summary
        if 'injury_status' in df.columns:
            out = (df['injury_status'] == 'OUT').sum()
            doubtful = (df['injury_status'] == 'DOUBTFUL').sum()
            questionable = (df['injury_status'] == 'QUESTIONABLE').sum()
            healthy = (df['injury_status'] == 'HEALTHY').sum()
            
            summary += "Injury Status:\n"
            summary += f"  OUT: {out}\n"
            summary += f"  DOUBTFUL: {doubtful}\n"
            summary += f"  QUESTIONABLE: {questionable}\n"
            summary += f"  HEALTHY: {healthy}\n\n"
        
        # Adjustment summary
        if 'total_adjustment' in df.columns:
            avg_adj = df['total_adjustment'].mean()
            summary += f"Average Total Adjustment: {avg_adj:.3f}x\n"
            
            boosted = df[df['total_adjustment'] > 1.15]
            summary += f"Significantly Boosted (>15%): {len(boosted)} players\n"
            
            reduced = df[df['total_adjustment'] < 0.85]
            summary += f"Significantly Reduced (<15%): {len(reduced)} players\n\n"
        
        summary += "=" * 70 + "\n"
        
        return summary
    
    def get_full_report(self, enrichment_result: Dict) -> str:
        """
        Get combined report with all data sources.
        
        Args:
            enrichment_result: Result from enrich_players()
        
        Returns:
            Full text report
        """
        full_report = "\n\n"
        
        # Vegas report
        if enrichment_result['reports']['vegas']:
            full_report += enrichment_result['reports']['vegas'] + "\n\n"
        
        # Weather report
        if enrichment_result['reports']['weather']:
            full_report += enrichment_result['reports']['weather'] + "\n\n"
        
        # Injury report
        if enrichment_result['reports']['injury']:
            full_report += enrichment_result['reports']['injury'] + "\n\n"
        
        # Summary
        full_report += enrichment_result['summary']
        
        return full_report


# Quick helper function
def enrich_players_quick(
    players_df: pd.DataFrame,
    weather_api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick helper to enrich players with all data.
    
    Args:
        players_df: Base player data
        weather_api_key: OpenWeatherMap API key (optional)
    
    Returns:
        Enriched DataFrame
    """
    enrichment = DataEnrichment(weather_api_key=weather_api_key)
    result = enrichment.enrich_players(players_df)
    
    # Print summary
    print(result['summary'])
    
    return result['enriched_df']


# Example usage
if __name__ == '__main__':
    """Example of data enrichment system"""
    
    # Sample player data
    sample_players = pd.DataFrame({
        'name': ['Patrick Mahomes', 'Josh Allen', 'Christian McCaffrey', 'Tyreek Hill'],
        'position': ['QB', 'QB', 'RB', 'WR'],
        'team': ['KC', 'BUF', 'SF', 'MIA'],
        'salary': [9500, 9200, 9300, 8900],
        'projection': [24.5, 23.8, 22.1, 18.9],
        'ceiling': [35.0, 34.0, 32.0, 28.0],
        'floor': [18.0, 17.5, 16.0, 12.0]
    })
    
    print("=" * 70)
    print("DATA ENRICHMENT EXAMPLE - v7.0.1")
    print("=" * 70)
    print("\nOriginal Players:")
    print(sample_players[['name', 'position', 'team', 'projection']])
    
    # Enrich with all data
    enrichment = DataEnrichment()
    result = enrichment.enrich_players(sample_players)
    
    # Show enriched data
    print("\n" + "=" * 70)
    print("ENRICHED PLAYERS")
    print("=" * 70)
    
    enriched = result['enriched_df']
    display_cols = ['name', 'position', 'projection_base', 'projection_adjusted']
    
    if 'vegas_implied_total' in enriched.columns:
        display_cols.append('vegas_implied_total')
    if 'weather_impact' in enriched.columns:
        display_cols.append('weather_impact')
    if 'injury_status' in enriched.columns:
        display_cols.append('injury_status')
    
    available = [c for c in display_cols if c in enriched.columns]
    print(enriched[available])
    
    # Print full report
    print("\n")
    print(enrichment.get_full_report(result))
