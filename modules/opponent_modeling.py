"""
DFS Meta-Optimizer - Opponent Modeling v6.1.0

NEW IN v6.1.0:
- Bring-Back Recommendations (opposing team leverage)
- Game Stack Detection (multi-team correlation analysis)
- Enhanced Leverage Calculations
- Matchup-based Correlation Adjustments

Core Features:
- Contest-size aware leverage scoring
- Field ownership estimation
- Bayesian ownership updates
- Win probability calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# GAME STACK DETECTION - NEW IN v6.1.0
# ============================================================================

@dataclass
class GameInfo:
    """Information about an NFL game."""
    home_team: str
    away_team: str
    game_total: float
    home_implied: float
    away_implied: float
    
@dataclass
class GameStackOpportunity:
    """Identifies high-correlation game stack opportunities."""
    game_id: str
    teams: Tuple[str, str]
    primary_team: str
    primary_qb: Optional[str]
    primary_receivers: List[str]
    bring_back_team: str
    bring_back_candidates: List[Dict]
    correlation_score: float
    game_total: float
    leverage_score: float
    ownership_discount: float
    
class GameStackDetector:
    """
    Detect and score game stacking opportunities.
    
    Game stacks involve players from both teams in high-scoring games,
    capitalizing on game script correlation.
    """
    
    def __init__(self, player_pool: pd.DataFrame, games: List[GameInfo]):
        """
        Initialize game stack detector.
        
        Args:
            player_pool: DataFrame with player data
            games: List of GameInfo objects with matchup data
        """
        self.player_pool = player_pool
        self.games = games
        self.game_map = self._build_game_map()
        
    def _build_game_map(self) -> Dict[str, GameInfo]:
        """Build mapping of teams to games."""
        game_map = {}
        for game in self.games:
            game_map[game.home_team] = game
            game_map[game.away_team] = game
        return game_map
    
    def find_opportunities(self, min_game_total: float = 47.0) -> List[GameStackOpportunity]:
        """
        Find high-quality game stacking opportunities.
        
        Args:
            min_game_total: Minimum over/under to consider
            
        Returns:
            List of GameStackOpportunity objects
        """
        opportunities = []
        
        # Find high-scoring games
        high_scoring_games = [
            g for g in self.games 
            if g.game_total >= min_game_total
        ]
        
        for game in high_scoring_games:
            # Analyze both directions (home primary, away bring-back and vice versa)
            opp1 = self._analyze_game_stack(
                game, 
                primary_team=game.home_team,
                bring_back_team=game.away_team
            )
            if opp1:
                opportunities.append(opp1)
            
            opp2 = self._analyze_game_stack(
                game,
                primary_team=game.away_team,
                bring_back_team=game.home_team
            )
            if opp2:
                opportunities.append(opp2)
        
        # Sort by leverage score
        opportunities.sort(key=lambda x: x.leverage_score, reverse=True)
        
        return opportunities
    
    def _analyze_game_stack(
        self, 
        game: GameInfo,
        primary_team: str,
        bring_back_team: str
    ) -> Optional[GameStackOpportunity]:
        """Analyze a potential game stack."""
        
        # Find QB from primary team
        primary_qbs = self.player_pool[
            (self.player_pool['team'] == primary_team) &
            (self.player_pool['position'] == 'QB')
        ]
        
        if primary_qbs.empty:
            return None
        
        primary_qb = primary_qbs.iloc[0]
        
        # Find receivers from primary team
        primary_receivers = self.player_pool[
            (self.player_pool['team'] == primary_team) &
            (self.player_pool['position'].isin(['WR', 'TE']))
        ].nlargest(3, 'projection')
        
        # Find bring-back candidates
        bring_back = self.player_pool[
            (self.player_pool['team'] == bring_back_team) &
            (self.player_pool['position'].isin(['WR', 'TE', 'RB']))
        ].nlargest(5, 'projection')
        
        if primary_receivers.empty or bring_back.empty:
            return None
        
        # Calculate correlation score
        correlation_score = self._calculate_game_correlation(
            game, primary_team, bring_back_team
        )
        
        # Calculate leverage score
        avg_ownership = np.mean([
            primary_qb.get('ownership', 15),
            primary_receivers['ownership'].mean(),
            bring_back['ownership'].mean()
        ])
        
        leverage_score = correlation_score * (1.0 - avg_ownership / 100)
        
        return GameStackOpportunity(
            game_id=f"{game.home_team}@{game.away_team}",
            teams=(primary_team, bring_back_team),
            primary_team=primary_team,
            primary_qb=primary_qb['name'],
            primary_receivers=primary_receivers['name'].tolist(),
            bring_back_team=bring_back_team,
            bring_back_candidates=[
                {
                    'name': row['name'],
                    'position': row['position'],
                    'projection': row.get('projection', 0),
                    'ownership': row.get('ownership', 10),
                    'salary': row.get('salary', 0)
                }
                for _, row in bring_back.iterrows()
            ],
            correlation_score=correlation_score,
            game_total=game.game_total,
            leverage_score=leverage_score,
            ownership_discount=(100 - avg_ownership) / 100
        )
    
    def _calculate_game_correlation(
        self,
        game: GameInfo,
        primary_team: str,
        bring_back_team: str
    ) -> float:
        """
        Calculate expected correlation for game stack.
        
        Higher totals = more correlated scoring
        """
        base_correlation = 0.35  # Base game script correlation
        
        # Adjust for game total
        if game.game_total >= 52:
            total_bonus = 0.15
        elif game.game_total >= 49:
            total_bonus = 0.10
        else:
            total_bonus = 0.05
        
        # Adjust for implied totals (closer = more correlated)
        total_diff = abs(game.home_implied - game.away_implied)
        if total_diff < 3:
            balance_bonus = 0.10
        elif total_diff < 6:
            balance_bonus = 0.05
        else:
            balance_bonus = 0.0
        
        return min(0.65, base_correlation + total_bonus + balance_bonus)


# ============================================================================
# BRING-BACK LOGIC - NEW IN v6.1.0
# ============================================================================

class BringBackAnalyzer:
    """
    Analyze and recommend bring-back plays.
    
    Bring-backs are opposing players that correlate with your primary stack
    due to game script (shootouts, back-and-forth scoring).
    """
    
    def __init__(self, player_pool: pd.DataFrame):
        self.player_pool = player_pool
        
    def find_bring_backs(
        self,
        primary_stack_players: List[str],
        min_correlation: float = 0.25
    ) -> List[Dict]:
        """
        Find optimal bring-back plays for a primary stack.
        
        Args:
            primary_stack_players: Names of players in primary stack
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of bring-back candidates with scores
        """
        # Get primary stack info
        primary_players = self.player_pool[
            self.player_pool['name'].isin(primary_stack_players)
        ]
        
        if primary_players.empty:
            return []
        
        primary_team = primary_players.iloc[0]['team']
        opponent = primary_players.iloc[0].get('opponent', '')
        
        if not opponent:
            return []
        
        # Find QB in primary stack
        has_qb = any(
            p['position'] == 'QB' 
            for _, p in primary_players.iterrows()
        )
        
        # Find bring-back candidates from opposing team
        candidates = self.player_pool[
            (self.player_pool['team'] == opponent) &
            (self.player_pool['position'].isin(['WR', 'TE', 'RB']))
        ].copy()
        
        if candidates.empty:
            return []
        
        # Score each candidate
        bring_backs = []
        for _, player in candidates.iterrows():
            score = self._score_bring_back(
                player,
                primary_team,
                has_qb,
                len(primary_stack_players)
            )
            
            if score['correlation'] >= min_correlation:
                bring_backs.append({
                    'name': player['name'],
                    'position': player['position'],
                    'team': player['team'],
                    'salary': player.get('salary', 0),
                    'projection': player.get('projection', 0),
                    'ownership': player.get('ownership', 10),
                    'correlation': score['correlation'],
                    'leverage_score': score['leverage'],
                    'bring_back_score': score['total']
                })
        
        # Sort by bring-back score
        bring_backs.sort(key=lambda x: x['bring_back_score'], reverse=True)
        
        return bring_backs[:5]  # Top 5 candidates
    
    def _score_bring_back(
        self,
        player: pd.Series,
        primary_team: str,
        has_qb: bool,
        stack_size: int
    ) -> Dict[str, float]:
        """Score a bring-back candidate."""
        
        # Base correlation by position
        position = player['position']
        if position == 'WR':
            base_corr = 0.41 if has_qb else 0.33
        elif position == 'TE':
            base_corr = 0.28 if has_qb else 0.22
        elif position == 'RB':
            base_corr = 0.18
        else:
            base_corr = 0.15
        
        # Adjust for stack size (larger stacks = more correlated)
        stack_bonus = (stack_size - 2) * 0.05
        correlation = min(0.65, base_corr + stack_bonus)
        
        # Calculate leverage (low ownership with high projection)
        ownership = player.get('ownership', 15)
        projection = player.get('projection', 0)
        
        if projection > 0:
            leverage = projection * (1.0 - ownership / 100)
        else:
            leverage = 0.0
        
        # Total bring-back score
        total = (correlation * 0.5) + (leverage * 0.5)
        
        return {
            'correlation': correlation,
            'leverage': leverage,
            'total': total
        }
    
    def recommend_for_lineup(
        self,
        lineup_players: List[Dict],
        available_slots: int = 1
    ) -> List[Dict]:
        """
        Recommend bring-back plays for a specific lineup.
        
        Args:
            lineup_players: Current lineup players
            available_slots: Number of slots available for bring-backs
            
        Returns:
            Recommended bring-back players
        """
        # Find primary stack
        teams = defaultdict(list)
        for player in lineup_players:
            teams[player['team']].append(player)
        
        # Identify team with QB (primary stack)
        primary_stack = []
        for team, players in teams.items():
            if any(p['position'] == 'QB' for p in players):
                primary_stack = [p['name'] for p in players]
                break
        
        if not primary_stack:
            return []
        
        # Find bring-backs
        bring_backs = self.find_bring_backs(primary_stack)
        
        # Filter out players already in lineup
        current_names = {p['name'] for p in lineup_players}
        bring_backs = [
            bb for bb in bring_backs
            if bb['name'] not in current_names
        ]
        
        return bring_backs[:available_slots]


# ============================================================================
# OPPONENT MODEL - Enhanced v6.1.0
# ============================================================================

class OpponentModel:
    """
    Advanced opponent modeling for DFS optimization.
    
    Focuses on beating the field through leverage rather than
    pure point maximization.
    """
    
    def __init__(
        self,
        player_pool: pd.DataFrame,
        contest_size: int = 10000,
        games: Optional[List[GameInfo]] = None
    ):
        """
        Initialize opponent model.
        
        Args:
            player_pool: DataFrame with player data
            contest_size: Number of entries in contest
            games: Optional list of GameInfo objects
        """
        self.player_pool = player_pool.copy()
        self.contest_size = contest_size
        self.games = games or []
        
        # Initialize v6.1.0 features
        if self.games:
            self.game_stack_detector = GameStackDetector(
                self.player_pool, self.games
            )
        else:
            self.game_stack_detector = None
            
        self.bring_back_analyzer = BringBackAnalyzer(self.player_pool)
        
        # Calculate base leverage
        self._calculate_leverage_scores()
        
    def _calculate_leverage_scores(self):
        """Calculate leverage scores for all players."""
        if 'projection' not in self.player_pool.columns:
            self.player_pool['leverage_score'] = 0.0
            return
        
        if 'ownership' not in self.player_pool.columns:
            self.player_pool['ownership'] = 10.0
        
        # Contest size adjustment
        if self.contest_size >= 100000:
            ownership_power = 1.8
        elif self.contest_size >= 10000:
            ownership_power = 1.5
        elif self.contest_size >= 1000:
            ownership_power = 1.2
        else:
            ownership_power = 1.0
        
        # Calculate leverage: Projection * (1 - Ownership)^power
        self.player_pool['leverage_score'] = (
            self.player_pool['projection'] *
            (1 - self.player_pool['ownership'] / 100) ** ownership_power
        )
        
        logger.info(f"Calculated leverage scores for {len(self.player_pool)} players")
        
    def get_field_ownership_estimate(self) -> Dict[str, float]:
        """
        Estimate field ownership percentages.
        
        Returns:
            Dictionary mapping player names to estimated ownership %
        """
        ownership = {}
        for _, player in self.player_pool.iterrows():
            ownership[player['name']] = player.get('ownership', 10.0)
        
        return ownership
    
    def calculate_win_probability(
        self,
        lineup: Dict,
        field_lineups: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate probability of lineup winning contest.
        
        Args:
            lineup: Lineup dictionary
            field_lineups: Optional sample of field lineups
            
        Returns:
            Win probability (0.0 to 1.0)
        """
        if not field_lineups:
            # Use ownership-based estimate
            lineup_own = lineup.get('ownership', 50)
            
            # Lower ownership = higher differentiation = higher win probability
            base_prob = 1.0 / self.contest_size
            own_multiplier = max(1.0, (100 - lineup_own) / 50)
            
            return min(0.10, base_prob * own_multiplier * 100)
        
        # Monte Carlo simulation
        wins = 0
        simulations = min(10000, len(field_lineups) * 100)
        
        for _ in range(simulations):
            # Simulate lineup scores
            lineup_score = np.random.normal(
                lineup['projection'],
                lineup['projection'] * 0.20
            )
            
            # Sample random opponent
            opp = random.choice(field_lineups)
            opp_score = np.random.normal(
                opp['projection'],
                opp['projection'] * 0.20
            )
            
            if lineup_score > opp_score:
                wins += 1
        
        return wins / simulations
    
    def analyze_leverage_opportunities(self) -> pd.DataFrame:
        """
        Identify players with highest leverage.
        
        Returns:
            DataFrame sorted by leverage score
        """
        leverage_players = self.player_pool.nlargest(20, 'leverage_score')
        
        return leverage_players[[
            'name', 'position', 'team', 'salary',
            'projection', 'ownership', 'leverage_score'
        ]]
    
    def get_game_stack_opportunities(
        self,
        min_game_total: float = 47.0
    ) -> List[GameStackOpportunity]:
        """
        Get game stacking opportunities (v6.1.0).
        
        Args:
            min_game_total: Minimum over/under
            
        Returns:
            List of GameStackOpportunity objects
        """
        if not self.game_stack_detector:
            logger.warning("No game data provided - cannot detect game stacks")
            return []
        
        return self.game_stack_detector.find_opportunities(min_game_total)
    
    def get_bring_back_recommendations(
        self,
        primary_stack_players: List[str]
    ) -> List[Dict]:
        """
        Get bring-back recommendations (v6.1.0).
        
        Args:
            primary_stack_players: Names of players in primary stack
            
        Returns:
            List of bring-back candidates
        """
        return self.bring_back_analyzer.find_bring_backs(primary_stack_players)
    
    def update_ownership_bayesian(
        self,
        player_name: str,
        observed_ownership: float,
        confidence: float = 0.5
    ):
        """
        Update player ownership using Bayesian approach.
        
        Args:
            player_name: Player name
            observed_ownership: Newly observed ownership %
            confidence: Confidence in new observation (0-1)
        """
        idx = self.player_pool[self.player_pool['name'] == player_name].index
        
        if len(idx) == 0:
            return
        
        current = self.player_pool.loc[idx[0], 'ownership']
        
        # Bayesian update: weight between prior and observation
        updated = (
            current * (1 - confidence) +
            observed_ownership * confidence
        )
        
        self.player_pool.loc[idx[0], 'ownership'] = updated
        
        # Recalculate leverage
        self._calculate_leverage_scores()
        
        logger.info(
            f"Updated {player_name} ownership: {current:.1f}% -> {updated:.1f}%"
        )
    
    def get_contrarian_plays(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify most contrarian high-value plays.
        
        Args:
            top_n: Number of plays to return
            
        Returns:
            DataFrame with contrarian plays
        """
        # Contrarian = high projection, low ownership
        self.player_pool['contrarian_score'] = (
            self.player_pool['projection'] /
            (self.player_pool['ownership'] + 1)  # Avoid division by zero
        )
        
        contrarian = self.player_pool.nlargest(top_n, 'contrarian_score')
        
        return contrarian[[
            'name', 'position', 'team', 'salary',
            'projection', 'ownership', 'contrarian_score'
        ]]
    
    def simulate_contest_outcomes(
        self,
        lineups: List[Dict],
        num_simulations: int = 1000
    ) -> Dict:
        """
        Simulate contest outcomes for a set of lineups.
        
        Args:
            lineups: List of lineup dictionaries
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with simulation results
        """
        results = {
            'lineups': lineups,
            'simulations': num_simulations,
            'win_probabilities': [],
            'top_10_probabilities': [],
            'roi_estimates': []
        }
        
        for lineup in lineups:
            wins = 0
            top_10_finishes = 0
            
            for _ in range(num_simulations):
                # Simulate lineup score
                score = np.random.normal(
                    lineup['projection'],
                    lineup['projection'] * 0.20
                )
                
                # Estimate percentile based on ownership differential
                own_diff = 50 - lineup['ownership']  # Average is 50%
                percentile = 0.50 + (own_diff / 100) * 0.3
                percentile = max(0.1, min(0.9, percentile))
                
                # Check if wins
                if np.random.random() < percentile:
                    rank = int(self.contest_size * (1 - percentile))
                    if rank == 1:
                        wins += 1
                    if rank <= max(10, int(self.contest_size * 0.001)):
                        top_10_finishes += 1
            
            results['win_probabilities'].append(wins / num_simulations)
            results['top_10_probabilities'].append(top_10_finishes / num_simulations)
            
            # Rough ROI estimate (needs payout structure)
            roi = (wins / num_simulations) * self.contest_size * 0.8 - 1.0
            results['roi_estimates'].append(roi)
        
        return results
    
    def generate_ownership_report(self) -> Dict:
        """
        Generate comprehensive ownership analysis report.
        
        Returns:
            Dictionary with ownership statistics
        """
        report = {
            'total_players': len(self.player_pool),
            'avg_ownership': self.player_pool['ownership'].mean(),
            'ownership_std': self.player_pool['ownership'].std(),
            'chalk_plays': len(
                self.player_pool[self.player_pool['ownership'] > 20]
            ),
            'contrarian_plays': len(
                self.player_pool[self.player_pool['ownership'] < 5]
            ),
            'high_leverage_count': len(
                self.player_pool[self.player_pool['leverage_score'] > 
                self.player_pool['leverage_score'].quantile(0.8)]
            )
        }
        
        # Top chalk by position
        report['chalk_by_position'] = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            pos_players = self.player_pool[self.player_pool['position'] == pos]
            if not pos_players.empty:
                chalk = pos_players.nlargest(3, 'ownership')
                report['chalk_by_position'][pos] = [
                    {
                        'name': row['name'],
                        'ownership': row['ownership'],
                        'leverage': row.get('leverage_score', 0)
                    }
                    for _, row in chalk.iterrows()
                ]
        
        return report


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def create_opponent_model(
    player_pool: pd.DataFrame,
    contest_size: int = 10000,
    games: Optional[List[Dict]] = None
) -> OpponentModel:
    """
    Create an opponent model instance.
    
    Args:
        player_pool: DataFrame with player data
        contest_size: Contest size
        games: Optional list of game dictionaries with:
               - home_team, away_team, game_total, home_implied, away_implied
               
    Returns:
        OpponentModel instance
    """
    # Convert game dicts to GameInfo objects
    game_objects = []
    if games:
        for g in games:
            game_objects.append(GameInfo(
                home_team=g.get('home_team', ''),
                away_team=g.get('away_team', ''),
                game_total=g.get('game_total', 45.0),
                home_implied=g.get('home_implied', 22.5),
                away_implied=g.get('away_implied', 22.5)
            ))
    
    return OpponentModel(player_pool, contest_size, game_objects)


if __name__ == '__main__':
    # Example usage
    sample_data = {
        'name': ['Mahomes', 'Hill', 'Kelce', 'Allen', 'Diggs'],
        'position': ['QB', 'WR', 'TE', 'QB', 'WR'],
        'team': ['KC', 'KC', 'KC', 'BUF', 'BUF'],
        'opponent': ['BUF', 'BUF', 'BUF', 'KC', 'KC'],
        'salary': [8500, 8000, 7500, 8200, 7800],
        'projection': [26.0, 18.5, 16.0, 25.0, 17.5],
        'ownership': [18.0, 14.0, 12.0, 16.0, 13.0]
    }
    
    df = pd.DataFrame(sample_data)
    
    games = [{
        'home_team': 'KC',
        'away_team': 'BUF',
        'game_total': 54.5,
        'home_implied': 28.0,
        'away_implied': 26.5
    }]
    
    model = create_opponent_model(df, contest_size=100000, games=games)
    
    # Test game stacks
    opportunities = model.get_game_stack_opportunities(min_game_total=50.0)
    print(f"\nFound {len(opportunities)} game stack opportunities")
    
    # Test bring-backs
    bring_backs = model.get_bring_back_recommendations(['Mahomes', 'Hill', 'Kelce'])
    print(f"\nTop bring-back recommendations: {len(bring_backs)}")
    for bb in bring_backs[:3]:
        print(f"  {bb['name']} ({bb['position']}): {bb['bring_back_score']:.2f}")
