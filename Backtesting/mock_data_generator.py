"""
Mock Historical Data Generator - Phase 2
Generates realistic historical player performance data for covariance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

class MockDataGenerator:
    """Generate realistic mock DFS historical data"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        np.random.seed(seed)
        self.positions = ['QB', 'RB', 'WR', 'TE', 'DST']
        
    def generate_player_pool(self, num_players: int = 50) -> List[Dict]:
        """Generate pool of players with characteristics"""
        players = []
        
        # Position distribution
        position_counts = {'QB': 10, 'RB': 15, 'WR': 15, 'TE': 8, 'DST': 2}
        
        player_id = 1
        for pos, count in position_counts.items():
            for i in range(count):
                # Player archetypes
                if np.random.random() < 0.3:
                    archetype = 'elite'  # High floor, high ceiling
                    base_projection = 18 if pos != 'DST' else 10
                    volatility = 0.25
                    ownership_base = 25
                elif np.random.random() < 0.5:
                    archetype = 'solid'  # Medium everything
                    base_projection = 12 if pos != 'DST' else 7
                    volatility = 0.35
                    ownership_base = 12
                else:
                    archetype = 'boom_bust'  # Low floor, high ceiling
                    base_projection = 10 if pos != 'DST' else 6
                    volatility = 0.55
                    ownership_base = 8
                
                players.append({
                    'player_id': f"{pos}{player_id}",
                    'name': f"{pos} Player {i+1}",
                    'position': pos,
                    'archetype': archetype,
                    'base_projection': base_projection,
                    'volatility': volatility,
                    'ownership_base': ownership_base,
                    'salary': int(3000 + (base_projection * 300) + np.random.randint(-500, 500))
                })
                player_id += 1
        
        return players
    
    def generate_historical_weeks(
        self, 
        players: List[Dict], 
        num_weeks: int = 8
    ) -> pd.DataFrame:
        """
        Generate historical performance data with realistic correlations
        """
        records = []
        
        # Week-level variance (game environment)
        for week in range(1, num_weeks + 1):
            # Random game environment for this week
            week_variance = np.random.normal(1.0, 0.15)
            
            for player in players:
                # Base factors
                base_proj = player['base_projection']
                volatility = player['volatility']
                
                # Actual score with realistic variance
                actual_score = base_proj * week_variance * np.random.normal(1.0, volatility)
                actual_score = max(0, actual_score)  # No negative scores
                
                # Ownership with some week-to-week variance
                ownership = player['ownership_base'] * np.random.normal(1.0, 0.2)
                ownership = np.clip(ownership, 1, 100)
                
                records.append({
                    'week': week,
                    'player_id': player['player_id'],
                    'name': player['name'],
                    'position': player['position'],
                    'archetype': player['archetype'],
                    'salary': player['salary'],
                    'projected_score': base_proj,
                    'actual_score': round(actual_score, 2),
                    'ownership': round(ownership, 1),
                    'value': round(actual_score / (player['salary'] / 1000), 2)
                })
        
        return pd.DataFrame(records)
    
    def add_game_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add game-level correlation effects"""
        # In real data: QB-WR same team correlated, QB-opposing DST anti-correlated
        # For mock: simulate with random game IDs
        
        df['game_id'] = np.random.randint(1, 9, size=len(df))
        
        # Add correlation tag for stacking
        df['stackable'] = (df['position'].isin(['QB', 'WR'])) & (np.random.random(len(df)) < 0.3)
        
        return df
    
    def generate_contest_results(
        self, 
        num_lineups: int = 20,
        num_weeks: int = 5
    ) -> pd.DataFrame:
        """Generate mock contest results for validation"""
        results = []
        
        for week in range(1, num_weeks + 1):
            for lineup_id in range(1, num_lineups + 1):
                # Simulate lineup performance
                total_score = np.random.normal(140, 25)
                placement = np.random.randint(1, 1001)
                entry_fee = 10
                
                # Prize calculation (simplified)
                if placement == 1:
                    winnings = 1000
                elif placement <= 10:
                    winnings = 50
                elif placement <= 100:
                    winnings = 15
                else:
                    winnings = 0
                
                roi = ((winnings - entry_fee) / entry_fee) * 100
                
                results.append({
                    'week': week,
                    'lineup_id': lineup_id,
                    'total_score': round(total_score, 2),
                    'placement': placement,
                    'entry_fee': entry_fee,
                    'winnings': winnings,
                    'profit': winnings - entry_fee,
                    'roi': round(roi, 1)
                })
        
        return pd.DataFrame(results)
    
    def generate_full_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete mock dataset"""
        print("Generating mock historical data...")
        
        # 1. Player pool
        players = self.generate_player_pool(50)
        
        # 2. Historical performances
        historical_df = self.generate_historical_weeks(players, num_weeks=8)
        historical_df = self.add_game_correlations(historical_df)
        
        # 3. Contest results
        results_df = self.generate_contest_results(num_lineups=20, num_weeks=5)
        
        print(f"Generated {len(historical_df)} player-week records")
        print(f"Generated {len(results_df)} contest results")
        
        return {
            'players': pd.DataFrame(players),
            'historical': historical_df,
            'contest_results': results_df
        }

if __name__ == "__main__":
    # Test generation
    generator = MockDataGenerator()
    data = generator.generate_full_dataset()
    
    print("\n=== Sample Historical Data ===")
    print(data['historical'].head(10))
    
    print("\n=== Sample Contest Results ===")
    print(data['contest_results'].head())
    
    # Save to CSV for inspection
    data['historical'].to_csv('/tmp/mock_historical.csv', index=False)
    data['contest_results'].to_csv('/tmp/mock_results.csv', index=False)
    print("\nData saved to /tmp/")
