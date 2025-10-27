"""
Data Integration Script
Combines Vegas lines, weather data, and injury status into player projections
"""

import pandas as pd
from typing import Optional, Dict
import logging

# Import all data sources
from modules.vegas_lines import VegasLinesProvider
from modules.weather_data import WeatherDataProvider
from modules.injury_tracker import InjuryTracker

logger = logging.getLogger(__name__)


class DataIntegrator:
    """
    Master class to integrate all external data sources.
    
    Combines:
    1. Vegas lines (spreads, totals, implied team totals)
    2. Weather data (temperature, wind, precipitation)
    3. Injury reports (OUT, DOUBTFUL, QUESTIONABLE)
    
    Usage:
        integrator = DataIntegrator(weather_api_key='your_key')
        enhanced_df = integrator.enhance_players(players_df)
    """
    
    def __init__(
        self,
        weather_api_key: Optional[str] = None,
        injury_source: str = 'fantasypros'
    ):
        """
        Initialize data integrator.
        
        Args:
            weather_api_key: OpenWeatherMap API key (optional)
            injury_source: Injury data source ('fantasypros', 'espn', 'nfl')
        """
        self.vegas_provider = VegasLinesProvider()
        self.weather_provider = WeatherDataProvider(weather_api_key)
        self.injury_tracker = InjuryTracker()
        self.injury_source = injury_source
        
        logger.info("DataIntegrator initialized")
    
    def enhance_players(
        self,
        players_df: pd.DataFrame,
        add_vegas: bool = True,
        add_weather: bool = True,
        add_injuries: bool = True,
        adjust_for_injuries: bool = True,
        filter_injured: bool = True
    ) -> Dict:
        """
        Enhance players DataFrame with all external data.
        
        Args:
            players_df: Base player data with projections
            add_vegas: Add Vegas lines
            add_weather: Add weather data
            add_injuries: Add injury status
            adjust_for_injuries: Adjust projections for injured players
            filter_injured: Remove OUT/DOUBTFUL players
        
        Returns:
            Dictionary with:
            - enhanced_df: Enhanced player DataFrame
            - vegas_lines: Vegas lines DataFrame
            - injury_report: Injury report DataFrame
            - reports: Text reports (vegas, weather, injury)
        """
        df = players_df.copy()
        
        reports = {
            'vegas': '',
            'weather': '',
            'injury': ''
        }
        
        vegas_lines_df = None
        injury_report_df = None
        
        logger.info(f"Starting data integration for {len(df)} players")
        
        # 1. Add Vegas lines
        if add_vegas:
            logger.info("Adding Vegas lines...")
            vegas_lines_df = self.vegas_provider.get_current_lines()
            
            if not vegas_lines_df.empty:
                df = self.vegas_provider.add_vegas_to_players(df, vegas_lines_df)
                reports['vegas'] = self.vegas_provider.get_vegas_report(vegas_lines_df)
                logger.info(f"✅ Added Vegas lines to {len(df)} players")
            else:
                logger.warning("⚠️ No Vegas lines available")
        
        # 2. Add weather data
        if add_weather:
            logger.info("Adding weather data...")
            df = self.weather_provider.add_weather_to_players(df)
            reports['weather'] = self.weather_provider.get_weather_report(df)
            logger.info(f"✅ Added weather data to {len(df)} players")
        
        # 3. Add injury status
        if add_injuries:
            logger.info("Fetching injury reports...")
            injury_report_df = self.injury_tracker.scrape_injury_report(self.injury_source)
            
            if not injury_report_df.empty:
                df = self.injury_tracker.add_injury_status_to_players(df, injury_report_df)
                
                # Adjust projections for injuries
                if adjust_for_injuries:
                    df = self.injury_tracker.adjust_projections_for_injury(df)
                
                # Filter out injured players
                if filter_injured:
                    original_count = len(df)
                    df = self.injury_tracker.filter_healthy_players(df)
                    logger.info(f"Filtered {original_count - len(df)} injured players")
                
                reports['injury'] = self.injury_tracker.get_injury_report(
                    df if not filter_injured else players_df  # Show all for report
                )
                logger.info(f"✅ Added injury data")
            else:
                logger.warning("⚠️ No injury data available")
        
        # 4. Calculate composite adjustments
        df = self._calculate_final_adjustments(df)
        
        logger.info(f"Data integration complete: {len(df)} players enhanced")
        
        return {
            'enhanced_df': df,
            'vegas_lines': vegas_lines_df,
            'injury_report': injury_report_df,
            'reports': reports,
            'summary': self._generate_summary(df, reports)
        }
    
    def _calculate_final_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final projection adjustments based on all data.
        
        Factors:
        - Vegas implied total (game environment)
        - Weather impact (passing/kicking conditions)
        - Injury status (availability/effectiveness)
        """
        df = df.copy()
        
        # Store original if not already stored
        if 'projection_base' not in df.columns:
            df['projection_base'] = df.get('projection_original', df['projection'])
        
        # Initialize adjustment factor
        df['total_adjustment'] = 1.0
        
        # Vegas adjustment (if available)
        if 'vegas_implied_total' in df.columns:
            # Normalize implied total to adjustment factor
            # League average ~23 points, so:
            # 28 pts = 1.2x multiplier
            # 23 pts = 1.0x multiplier
            # 18 pts = 0.8x multiplier
            df['vegas_adjustment'] = (df['vegas_implied_total'] / 23.0).clip(0.7, 1.3)
            df['total_adjustment'] *= df['vegas_adjustment']
        
        # Weather adjustment (if available)
        if 'weather_impact' in df.columns:
            # weather_impact is 0-100 scale, convert to multiplier
            df['weather_adjustment'] = df['weather_impact'] / 100.0
            df['total_adjustment'] *= df['weather_adjustment']
        
        # Injury adjustment already applied to projection
        # Just include it in total_adjustment for tracking
        if 'injury_impact' in df.columns:
            df['total_adjustment'] *= df['injury_impact']
        
        # Apply total adjustment to projections
        df['projection_adjusted'] = df['projection_base'] * df['total_adjustment']
        
        # For backward compatibility, update main projection
        df['projection'] = df['projection_adjusted']
        
        # Adjust ceiling/floor too
        if 'ceiling' in df.columns:
            if 'ceiling_base' not in df.columns:
                df['ceiling_base'] = df.get('ceiling_original', df['ceiling'])
            df['ceiling'] = df['ceiling_base'] * df['total_adjustment']
        
        if 'floor' in df.columns:
            if 'floor_base' not in df.columns:
                df['floor_base'] = df.get('floor_original', df['floor'])
            df['floor'] = df['floor_base'] * df['total_adjustment']
        
        # Log significant adjustments
        big_adjustments = df[
            (df['total_adjustment'] > 1.15) | 
            (df['total_adjustment'] < 0.85)
        ].copy()
        
        if not big_adjustments.empty:
            logger.info(f"Found {len(big_adjustments)} players with significant adjustments:")
            for _, player in big_adjustments.head(10).iterrows():
                base = player.get('projection_base', player['projection'])
                adjusted = player['projection_adjusted']
                change_pct = (adjusted / base - 1) * 100 if base > 0 else 0
                
                logger.info(f"  {player['name']}: {base:.1f} → {adjusted:.1f} "
                          f"({change_pct:+.0f}%) - adj={player['total_adjustment']:.2f}")
        
        return df
    
    def _generate_summary(self, df: pd.DataFrame, reports: Dict) -> str:
        """Generate comprehensive summary report"""
        summary = "=" * 70 + "\n"
        summary += "DATA INTEGRATION SUMMARY\n"
        summary += "=" * 70 + "\n\n"
        
        summary += f"Total Players: {len(df)}\n\n"
        
        # Vegas summary
        if 'vegas_spread' in df.columns:
            avg_total = df['vegas_implied_total'].mean()
            summary += f"Average Vegas Total: {avg_total:.1f} points\n"
            
            favorites = df[df['vegas_spread'] < -3.0]
            summary += f"Heavy Favorites (<-3): {len(favorites)} teams\n"
            
            underdogs = df[df['vegas_spread'] > 3.0]
            summary += f"Heavy Underdogs (>+3): {len(underdogs)} teams\n\n"
        
        # Weather summary
        if 'weather_wind' in df.columns:
            bad_weather = df[
                (df['weather_wind'] > 15) | 
                (df['weather_temp'] < 32) |
                (df['weather_conditions'].isin(['Rain', 'Snow', 'Thunderstorm']))
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
            
            summary += f"Injury Status:\n"
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
            summary += f"Significantly Reduced (<-15%): {len(reduced)} players\n\n"
        
        summary += "=" * 70 + "\n"
        
        return summary
    
    def get_all_reports(self, integration_result: Dict) -> str:
        """
        Get combined report with all data.
        
        Args:
            integration_result: Result from enhance_players()
        
        Returns:
            Combined text report
        """
        full_report = "\n\n"
        
        # Vegas report
        if integration_result['reports']['vegas']:
            full_report += integration_result['reports']['vegas'] + "\n\n"
        
        # Weather report
        if integration_result['reports']['weather']:
            full_report += integration_result['reports']['weather'] + "\n\n"
        
        # Injury report
        if integration_result['reports']['injury']:
            full_report += integration_result['reports']['injury'] + "\n\n"
        
        # Summary
        full_report += integration_result['summary']
        
        return full_report


# Quick helper functions

def enhance_players_quick(
    players_df: pd.DataFrame,
    weather_api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick helper to enhance players with all data.
    
    Args:
        players_df: Base player data
        weather_api_key: OpenWeatherMap API key (optional)
    
    Returns:
        Enhanced DataFrame
    """
    integrator = DataIntegrator(weather_api_key=weather_api_key)
    result = integrator.enhance_players(players_df)
    
    # Print reports
    print(result['summary'])
    
    return result['enhanced_df']


def enhance_with_reports(
    players_df: pd.DataFrame,
    weather_api_key: Optional[str] = None
) -> Dict:
    """
    Enhance players and get full reports.
    
    Args:
        players_df: Base player data
        weather_api_key: OpenWeatherMap API key
    
    Returns:
        Full integration result with reports
    """
    integrator = DataIntegrator(weather_api_key=weather_api_key)
    result = integrator.enhance_players(players_df)
    
    # Print all reports
    print(integrator.get_all_reports(result))
    
    return result


# Example usage
if __name__ == '__main__':
    """
    Example usage of data integration
    """
    
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
    print("DATA INTEGRATION EXAMPLE")
    print("=" * 70)
    print("\nOriginal Player Data:")
    print(sample_players[['name', 'position', 'team', 'projection']])
    
    # Enhance with all data (without weather API key for demo)
    integrator = DataIntegrator()
    result = integrator.enhance_players(sample_players)
    
    # Show enhanced data
    print("\n" + "=" * 70)
    print("ENHANCED PLAYER DATA")
    print("=" * 70)
    
    enhanced = result['enhanced_df']
    
    display_cols = ['name', 'position', 'projection_base', 'projection_adjusted', 'total_adjustment']
    if 'vegas_implied_total' in enhanced.columns:
        display_cols.insert(3, 'vegas_implied_total')
    if 'weather_impact' in enhanced.columns:
        display_cols.insert(4, 'weather_impact')
    if 'injury_status' in enhanced.columns:
        display_cols.append('injury_status')
    
    available_cols = [col for col in display_cols if col in enhanced.columns]
    print(enhanced[available_cols])
    
    # Print all reports
    print("\n")
    print(integrator.get_all_reports(result))
