"""
Module 4: Weather Data Integration
Fetches weather data for NFL games to inform DFS decisions
"""

import pandas as pd
import requests
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WeatherDataProvider:
    """
    Fetches weather data for NFL games.
    
    Weather impacts:
    - Wind: Affects passing game, kickers
    - Rain/Snow: Reduces passing, increases running
    - Temperature: Affects ball handling, player performance
    - Dome games: No weather impact
    """
    
    # NFL stadium locations
    STADIUM_CITIES = {
        'ARI': 'Phoenix',  # State Farm Stadium (retractable roof)
        'ATL': 'Atlanta',  # Mercedes-Benz Stadium (retractable roof)
        'BAL': 'Baltimore',
        'BUF': 'Buffalo',
        'CAR': 'Charlotte',
        'CHI': 'Chicago',
        'CIN': 'Cincinnati',
        'CLE': 'Cleveland',
        'DAL': 'Arlington',  # AT&T Stadium (retractable roof)
        'DEN': 'Denver',
        'DET': 'Detroit',  # Ford Field (dome)
        'GB': 'Green Bay',
        'HOU': 'Houston',  # NRG Stadium (retractable roof)
        'IND': 'Indianapolis',  # Lucas Oil Stadium (retractable roof)
        'JAX': 'Jacksonville',
        'KC': 'Kansas City',
        'LAC': 'Los Angeles',  # SoFi Stadium (indoor)
        'LAR': 'Los Angeles',  # SoFi Stadium (indoor)
        'LV': 'Las Vegas',  # Allegiant Stadium (dome)
        'MIA': 'Miami',
        'MIN': 'Minneapolis',  # U.S. Bank Stadium (dome)
        'NE': 'Foxborough',
        'NO': 'New Orleans',  # Superdome (dome)
        'NYG': 'East Rutherford',
        'NYJ': 'East Rutherford',
        'PHI': 'Philadelphia',
        'PIT': 'Pittsburgh',
        'SEA': 'Seattle',
        'SF': 'San Francisco',
        'TB': 'Tampa',
        'TEN': 'Nashville',
        'WAS': 'Landover'
    }
    
    # Dome/retractable roof stadiums (weather doesn't matter)
    DOME_TEAMS = {
        'ATL', 'DET', 'IND', 'LAC', 'LAR', 'LV', 'MIN', 'NO'
    }
    
    # Retractable roof teams (usually closed in bad weather)
    RETRACTABLE_TEAMS = {
        'ARI', 'DAL', 'HOU'
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather data provider.
        
        Args:
            api_key: OpenWeatherMap API key (free tier: 1000 calls/day)
                    Get free key at: https://openweathermap.org/api
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/forecast"
        
        if not api_key:
            logger.warning("No API key provided. Weather data will be estimated.")
        
        logger.info("WeatherDataProvider initialized")
    
    def _is_dome_game(self, team: str) -> bool:
        """Check if team plays in dome/indoor stadium"""
        return team in self.DOME_TEAMS
    
    def _is_retractable(self, team: str) -> bool:
        """Check if team has retractable roof"""
        return team in self.RETRACTABLE_TEAMS
    
    def get_weather_for_city(
        self,
        city: str,
        game_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get weather forecast for city.
        
        Args:
            city: City name (e.g., 'Green Bay', 'Buffalo')
            game_date: Date of game (uses closest forecast time)
        
        Returns:
            Dictionary with weather data
        """
        if not self.api_key:
            logger.warning(f"No API key - returning default weather for {city}")
            return self._get_default_weather()
        
        try:
            params = {
                'q': f"{city},US",
                'appid': self.api_key,
                'units': 'imperial'  # Fahrenheit
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Get first forecast (or closest to game_date)
            forecast = data['list'][0]
            
            weather_data = {
                'temperature': forecast['main']['temp'],
                'feels_like': forecast['main']['feels_like'],
                'wind_speed': forecast['wind']['speed'],
                'wind_gust': forecast['wind'].get('gust', forecast['wind']['speed']),
                'conditions': forecast['weather'][0]['main'],
                'description': forecast['weather'][0]['description'],
                'precipitation_prob': forecast.get('pop', 0) * 100,  # Probability of precipitation
                'humidity': forecast['main']['humidity'],
                'city': city
            }
            
            logger.info(f"Weather for {city}: {weather_data['temperature']}°F, "
                       f"{weather_data['wind_speed']} mph wind, {weather_data['conditions']}")
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather for {city}: {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self) -> Dict:
        """Return default/neutral weather when API unavailable"""
        return {
            'temperature': 65,
            'feels_like': 65,
            'wind_speed': 5,
            'wind_gust': 8,
            'conditions': 'Clear',
            'description': 'clear sky',
            'precipitation_prob': 0,
            'humidity': 50,
            'city': 'Unknown'
        }
    
    def get_weather_impact_score(self, weather: Dict) -> Dict:
        """
        Calculate weather impact on different positions.
        
        Returns impact scores (0-100, where 100 = perfect conditions)
        
        Args:
            weather: Weather data dictionary
        
        Returns:
            Impact scores for QB, WR, RB, K, DST
        """
        temp = weather['temperature']
        wind = weather['wind_speed']
        conditions = weather['conditions'].lower()
        precip = weather['precipitation_prob']
        
        # Base scores (perfect conditions)
        scores = {
            'QB': 100,
            'WR': 100,
            'TE': 100,
            'RB': 100,
            'K': 100,
            'DST': 100
        }
        
        # Wind impact (primarily affects passing and kicking)
        if wind > 20:
            scores['QB'] -= 30
            scores['WR'] -= 30
            scores['TE'] -= 25
            scores['K'] -= 40
        elif wind > 15:
            scores['QB'] -= 20
            scores['WR'] -= 20
            scores['TE'] -= 15
            scores['K'] -= 30
        elif wind > 10:
            scores['QB'] -= 10
            scores['WR'] -= 10
            scores['TE'] -= 8
            scores['K'] -= 15
        
        # Precipitation impact
        if 'rain' in conditions or 'snow' in conditions or precip > 70:
            scores['QB'] -= 15
            scores['WR'] -= 15
            scores['TE'] -= 10
            scores['RB'] += 5  # RBs benefit from rain
            scores['K'] -= 20
            
            if 'snow' in conditions:
                scores['QB'] -= 10  # Extra penalty for snow
                scores['WR'] -= 10
                scores['K'] -= 15
        
        # Extreme temperature impact
        if temp < 20:  # Very cold
            scores['QB'] -= 10
            scores['WR'] -= 10
            scores['K'] -= 15
        elif temp > 95:  # Very hot
            scores['RB'] -= 5
            scores['DST'] -= 5
        
        # Cap scores at 0-100
        for key in scores:
            scores[key] = max(0, min(100, scores[key]))
        
        return scores
    
    def add_weather_to_players(
        self,
        players_df: pd.DataFrame,
        game_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Add weather data to players DataFrame.
        
        Args:
            players_df: DataFrame with player data (must have 'team' column)
            game_date: Game date for forecast
        
        Returns:
            DataFrame with weather columns added
        """
        df = players_df.copy()
        
        # Initialize weather columns
        df['weather_temp'] = 65.0
        df['weather_wind'] = 5.0
        df['weather_conditions'] = 'Clear'
        df['weather_impact'] = 100  # 0-100 score
        df['is_dome'] = False
        
        # Get unique teams
        teams = df['team'].unique()
        
        # Fetch weather for each team's city
        weather_cache = {}
        
        for team in teams:
            # Skip dome teams
            if self._is_dome_game(team):
                df.loc[df['team'] == team, 'is_dome'] = True
                df.loc[df['team'] == team, 'weather_impact'] = 100
                continue
            
            city = self.STADIUM_CITIES.get(team)
            if not city:
                logger.warning(f"No city found for team {team}")
                continue
            
            # Get weather (use cache to avoid duplicate API calls)
            if city not in weather_cache:
                weather_cache[city] = self.get_weather_for_city(city, game_date)
            
            weather = weather_cache[city]
            impact_scores = self.get_weather_impact_score(weather)
            
            # Add weather data to players from this team
            team_mask = df['team'] == team
            df.loc[team_mask, 'weather_temp'] = weather['temperature']
            df.loc[team_mask, 'weather_wind'] = weather['wind_speed']
            df.loc[team_mask, 'weather_conditions'] = weather['conditions']
            
            # Add position-specific impact
            for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
                pos_mask = team_mask & (df['position'] == position)
                if pos_mask.any():
                    df.loc[pos_mask, 'weather_impact'] = impact_scores.get(position, 100)
        
        logger.info(f"Added weather data to {len(df)} players")
        
        # Log bad weather games
        bad_weather = df[
            (df['weather_wind'] > 15) | 
            (df['weather_conditions'].isin(['Rain', 'Snow', 'Thunderstorm']))
        ]
        if not bad_weather.empty:
            logger.warning(f"⚠️ {len(bad_weather)} players affected by bad weather")
            for team in bad_weather['team'].unique():
                team_weather = df[df['team'] == team].iloc[0]
                logger.warning(f"  {team}: {team_weather['weather_temp']:.0f}°F, "
                             f"{team_weather['weather_wind']:.0f} mph wind, "
                             f"{team_weather['weather_conditions']}")
        
        return df
    
    def get_weather_report(self, players_df: pd.DataFrame) -> str:
        """
        Generate text weather report for all games.
        
        Args:
            players_df: DataFrame with weather data added
        
        Returns:
            Formatted weather report string
        """
        report = "=" * 60 + "\n"
        report += "WEATHER REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Get unique teams with weather data
        teams_weather = players_df.groupby('team').first()[
            ['weather_temp', 'weather_wind', 'weather_conditions', 'is_dome']
        ].reset_index()
        
        # Dome games
        dome_teams = teams_weather[teams_weather['is_dome']]
        if not dome_teams.empty:
            report += "🏟️  DOME GAMES (No weather impact):\n"
            for _, row in dome_teams.iterrows():
                report += f"  {row['team']}\n"
            report += "\n"
        
        # Outdoor games
        outdoor_teams = teams_weather[~teams_weather['is_dome']].sort_values(
            'weather_wind', ascending=False
        )
        
        if not outdoor_teams.empty:
            report += "🌤️  OUTDOOR GAMES:\n"
            for _, row in outdoor_teams.iterrows():
                temp = row['weather_temp']
                wind = row['weather_wind']
                conditions = row['weather_conditions']
                
                # Weather emoji
                if wind > 15:
                    emoji = "💨"
                elif conditions in ['Rain', 'Thunderstorm']:
                    emoji = "🌧️"
                elif conditions == 'Snow':
                    emoji = "❄️"
                elif temp > 85:
                    emoji = "🔥"
                elif temp < 32:
                    emoji = "🥶"
                else:
                    emoji = "✅"
                
                report += f"  {emoji} {row['team']}: {temp:.0f}°F, {wind:.0f} mph wind, {conditions}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report


# Helper function for quick integration
def add_weather_data(
    players_df: pd.DataFrame,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick helper to add weather data to players.
    
    Args:
        players_df: Players DataFrame
        api_key: OpenWeatherMap API key (optional)
    
    Returns:
        DataFrame with weather columns
    """
    provider = WeatherDataProvider(api_key)
    return provider.add_weather_to_players(players_df)
