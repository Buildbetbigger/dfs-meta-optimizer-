"""
Optimization Engine

Generates optimal lineups using opponent modeling insights.
Instead of just maximizing points, optimizes for competitive advantage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from config.settings import (
    SALARY_CAP,
    ROSTER_SIZE,
    CAPTAIN_MULTIPLIER,
    OPTIMIZATION_MODES,
    MAX_PLAYER_EXPOSURE,
    MIN_PLAYER_DIFFERENCE
)


class LineupOptimizer:
    """
    Generates lineups optimized for competitive advantage
    """
    
    def __init__(self, players_df: pd.DataFrame, opponent_model):
        """
        Initialize the optimizer
        
        Args:
            players_df: DataFrame with player data and metrics
            opponent_model: OpponentModel instance
        """
        # CRITICAL FIX: Use opponent_model's DataFrame which has leverage_score calculated
        self.players_df = opponent_model.players_df.copy()
        
        # CRITICAL FIX: Clean all player names on initialization
        self.players_df['name'] = (
            self.players_df['name']
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        
        self.opponent_model = opponent_model
        self.generated_lineups = []
        self.player_usage = {name: 0 for name in self.players_df['name']}
        self.lineup_signatures = set()  # Track generated lineups
    
    def _get_player_data_safe(self, player_name: str) -> Optional[pd.Series]:
        """
        Safely retrieve player data with robust name matching
        
        Args:
            player_name: Player name to look up
            
        Returns:
            Player data as Series, or None if not found
        """
        # Clean the search name
        clean_name = str(player_name).strip()
        
        # Try exact match first
        matches = self.players_df[self.players_df['name'] == clean_name]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Try with additional cleaning
        matches = self.players_df[
            self.players_df['name'].str.strip() == clean_name
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Try case-insensitive match
        matches = self.players_df[
            self.players_df['name'].str.lower() == clean_name.lower()
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Player not found
        print(f"WARNING: Player '{player_name}' not found in DataFrame!")
        return None
    
    def generate_lineups(self,
                        num_lineups: int = 20,
                        mode: str = 'balanced',
                        max_exposure: float = 0.4) -> List[Dict]:
        """
        Generate multiple optimized lineups with diversity
        
        Args:
            num_lineups: Number of lineups to generate
            mode: Optimization mode from OPTIMIZATION_MODES
            max_exposure: Maximum exposure per player (0-1)
            
        Returns:
            List of lineup dictionaries
        """
        print(f"\n🎯 Generating {num_lineups} lineups in {mode} mode...")
        print(f"   Salary cap: ${SALARY_CAP:,}")
        print(f"   Required usage: 96%+ (${int(SALARY_CAP * 0.96):,}+)")
        
        if mode not in OPTIMIZATION_MODES:
            print(f"⚠️ Warning: Mode '{mode}' not found, using 'balanced'")
            mode = 'balanced'
        
        mode_config = OPTIMIZATION_MODES[mode]
        self.generated_lineups = []
        self.lineup_signatures = set()
        max_attempts = num_lineups * 10  # Prevent infinite loops
        attempts = 0
        
        # Reset player usage tracking
        self.player_usage = {name: 0 for name in self.players_df['name']}
        
        while len(self.generated_lineups) < num_lineups and attempts < max_attempts:
            attempts += 1
            
            lineup = self._generate_single_lineup(
                mode_config=mode_config,
                max_exposure=max_exposure,
                iteration=len(self.generated_lineups),
                total_lineups=num_lineups
            )
            
            if lineup:
                # Create signature for deduplication
                signature = self._get_lineup_signature(lineup)
                
                if signature not in self.lineup_signatures:
                    self.lineup_signatures.add(signature)
                    self.generated_lineups.append(lineup)
                    
                    # Update player usage
                    all_players = [lineup['captain']] + lineup['flex']
                    for player in all_players:
                        self.player_usage[player] = self.player_usage.get(player, 0) + 1
                    
                    if len(self.generated_lineups) % 5 == 0:
                        print(f"   Generated {len(self.generated_lineups)}/{num_lineups}...")
        
        print(f"✅ Generated {len(self.generated_lineups)} unique lineups in {attempts} attempts\n")
        
        return self.generated_lineups
    
    def _get_lineup_signature(self, lineup: Dict) -> str:
        """Create unique signature for lineup deduplication"""
        all_players = sorted([lineup['captain']] + lineup['flex'])
        return '|'.join(all_players)
    
    def _calculate_player_scores(self,
                                 mode_config: Dict,
                                 iteration: int = 0,
                                 total_lineups: int = 20) -> pd.Series:
        """
        Calculate player scores with progressive randomization for diversity
        
        Args:
            mode_config: Mode configuration with weights
            iteration: Current lineup number (for progressive randomization)
            total_lineups: Total lineups to generate
            
        Returns:
            Series of player scores
        """
        df = self.players_df.copy()
        
        # CRITICAL FIX: Get weights with safe defaults
        projection_weight = mode_config.get('projection_weight', 1.0)
        leverage_weight = mode_config.get('leverage_weight', 0.5)
        ceiling_weight = mode_config.get('ceiling_weight', 0.3)
        ownership_weight = mode_config.get('ownership_weight', 0.01)
        
        # Base score from weighted combination
        score = (
            df['projection'] * projection_weight +
            df['leverage_score'] * leverage_weight +
            df['ceiling'] * ceiling_weight -
            df['ownership'] * ownership_weight
        )
        
        # CRITICAL FIX: Progressive randomization (15-50% variance)
        # More lineups generated = stronger randomization needed
        progress = iteration / max(total_lineups, 1)
        min_variance = 0.85  # 15% variance
        max_variance_range = 0.50  # Up to 50% variance
        variance_low = min_variance - (progress * max_variance_range / 2)
        variance_high = 1 / variance_low
        
        random_factor = np.random.uniform(variance_low, variance_high, len(df))
        
        # Usage-based penalty (promote less-used players)
        usage_counts = pd.Series({name: self.player_usage.get(name, 0) for name in df['name']})
        usage_penalty = 1 - (usage_counts / max(usage_counts.max(), 1)) * 0.3  # Up to 30% penalty
        
        # Anti-chalk bonus in leverage modes
        if leverage_weight > 0:
            chalk_bonus = (100 - df['ownership']) / 100 * 0.2  # Up to 20% bonus for low-owned
            score = score * (1 + chalk_bonus)
        
        # Apply randomization and penalties
        score = score * random_factor * usage_penalty
        
        return score
    
    def _select_captain(self,
                       available_players: pd.DataFrame,
                       mode_config: Dict,
                       iteration: int = 0) -> Optional[str]:
        """
        Select captain using probabilistic selection for diversity
        
        Args:
            available_players: DataFrame of available players with scores
            mode_config: Mode configuration
            iteration: Current iteration for randomization
            
        Returns:
            Captain player name or None
        """
        # Get more candidates for better diversity
        num_candidates = min(15, len(available_players))
        captain_candidates = available_players.nlargest(num_candidates, 'optimizer_score')
        
        # CRITICAL FIX: Probabilistic selection (not deterministic)
        # Calculate selection probabilities based on scores
        scores = captain_candidates['optimizer_score'].values
        min_score = scores.min()
        normalized_scores = scores - min_score + 1  # Ensure all positive
        
        # Temperature increases with iteration (more random over time)
        temperature = 1.0 + (iteration / 20)  # Gradually increases
        probabilities = np.exp(normalized_scores / temperature)
        probabilities = probabilities / probabilities.sum()
        
        # Select captain probabilistically
        captain_idx = np.random.choice(len(captain_candidates), p=probabilities)
        captain_name = captain_candidates.iloc[captain_idx]['name']
        
        return captain_name
    
    def _select_flex_players(self,
                           available_players: pd.DataFrame,
                           remaining_salary: int,
                           needed: int,
                           iteration: int = 0) -> List[str]:
        """
        Select FLEX players using stochastic selection strategies
        
        Args:
            available_players: DataFrame of available players
            remaining_salary: Remaining salary cap
            needed: Number of players needed
            iteration: Current iteration
            
        Returns:
            List of selected player names
        """
        selected = []
        remaining = remaining_salary
        
        # CRITICAL FIX: Three rotating strategies for diversity
        strategy = iteration % 3
        
        for i in range(needed):
            # Filter affordable players
            affordable = available_players[
                (available_players['salary'] <= remaining) &
                (~available_players['name'].isin(selected))
            ].copy()
            
            if affordable.empty:
                return []  # Can't fill lineup
            
            # Apply strategy
            if strategy == 0:  # Greedy with noise
                # Pick from top candidates with probability
                num_candidates = min(8, len(affordable))
                candidates = affordable.nlargest(num_candidates, 'optimizer_score')
                
                scores = candidates['optimizer_score'].values
                probabilities = scores / scores.sum()
                
                idx = np.random.choice(len(candidates), p=probabilities)
                player_name = candidates.iloc[idx]['name']
                
            elif strategy == 1:  # Balanced selection
                # Mix of top and random
                if i < needed // 2:
                    # First half: top picks with noise
                    candidates = affordable.nlargest(5, 'optimizer_score')
                    player_name = candidates.sample(n=1).iloc[0]['name']
                else:
                    # Second half: weighted random
                    scores = affordable['optimizer_score'].values
                    min_score = scores.min()
                    normalized = scores - min_score + 1
                    probabilities = normalized / normalized.sum()
                    
                    idx = np.random.choice(len(affordable), p=probabilities)
                    player_name = affordable.iloc[idx]['name']
            
            else:  # strategy == 2: Exploratory
                # More random, weighted by score
                scores = affordable['optimizer_score'].values
                min_score = scores.min()
                normalized = (scores - min_score + 1) ** 0.7  # Less aggressive weighting
                probabilities = normalized / normalized.sum()
                
                idx = np.random.choice(len(affordable), p=probabilities)
                player_name = affordable.iloc[idx]['name']
            
            # Add player and update remaining
            selected.append(player_name)
            player_salary = affordable[affordable['name'] == player_name].iloc[0]['salary']
            remaining -= player_salary
            
        return selected
    
    def _optimize_salary_usage(self,
                               captain: str,
                               flex_players: List[str]) -> Tuple[str, List[str]]:
        """
        Optimize lineup to use 96%+ of salary cap
        
        Args:
            captain: Captain player name
            flex_players: List of FLEX player names
            
        Returns:
            Tuple of (optimized_captain, optimized_flex_players)
        """
        all_players = [captain] + flex_players
        
        # Calculate current salary
        total_salary = 0
        captain_data = self._get_player_data_safe(captain)
        if captain_data is not None:
            total_salary += captain_data['salary'] * CAPTAIN_MULTIPLIER
        
        for player in flex_players:
            player_data = self._get_player_data_safe(player)
            if player_data is not None:
                total_salary += player_data['salary']
        
        min_required = int(SALARY_CAP * 0.96)
        
        if total_salary >= min_required:
            return captain, flex_players  # Already good
        
        # Try to upgrade players to use more salary
        remaining = SALARY_CAP - total_salary
        
        # Try upgrading FLEX players
        for i in range(len(flex_players)):
            current_player = flex_players[i]
            current_data = self._get_player_data_safe(current_player)
            if current_data is None:
                continue
            
            current_salary = current_data['salary']
            max_upgrade_salary = current_salary + remaining
            
            # Find better player to upgrade to
            upgrade_candidates = self.players_df[
                (self.players_df['salary'] > current_salary) &
                (self.players_df['salary'] <= max_upgrade_salary) &
                (~self.players_df['name'].isin(all_players))
            ].copy()
            
            if not upgrade_candidates.empty:
                # Pick best upgrade by projection
                best_upgrade = upgrade_candidates.nlargest(1, 'projection').iloc[0]
                salary_increase = best_upgrade['salary'] - current_salary
                
                if salary_increase <= remaining:
                    # Make the upgrade
                    flex_players[i] = best_upgrade['name']
                    total_salary += salary_increase
                    remaining -= salary_increase
                    all_players = [captain] + flex_players
                    
                    if total_salary >= min_required:
                        return captain, flex_players
        
        return captain, flex_players
    
    def _generate_single_lineup(self,
                               mode_config: Dict,
                               max_exposure: float = 0.4,
                               iteration: int = 0,
                               total_lineups: int = 20) -> Optional[Dict]:
        """
        Generate a single optimized lineup
        
        Args:
            mode_config: Mode configuration
            max_exposure: Maximum player exposure
            iteration: Current iteration number
            total_lineups: Total lineups to generate
            
        Returns:
            Lineup dictionary or None if generation failed
        """
        # Calculate player scores with progressive randomization
        player_scores = self._calculate_player_scores(mode_config, iteration, total_lineups)
        
        df = self.players_df.copy()
        df['optimizer_score'] = player_scores
        
        # Filter players at max exposure
        available = df[
            df['name'].map(lambda x: self.player_usage.get(x, 0)) < (max_exposure * total_lineups)
        ].copy()
        
        if len(available) < ROSTER_SIZE:
            return None  # Not enough players
        
        # Select captain with probabilistic selection
        captain = self._select_captain(available, mode_config, iteration)
        if not captain:
            return None
        
        captain_data = self._get_player_data_safe(captain)
        if captain_data is None:
            return None
        
        captain_salary = captain_data['salary'] * CAPTAIN_MULTIPLIER
        remaining_salary = SALARY_CAP - captain_salary
        
        # Select FLEX players with stochastic strategies
        available_flex = available[available['name'] != captain].copy()
        flex_players = self._select_flex_players(
            available_flex,
            remaining_salary,
            ROSTER_SIZE - 1,
            iteration
        )
        
        if len(flex_players) != ROSTER_SIZE - 1:
            return None  # Could not fill lineup
        
        # CRITICAL FIX: Optimize salary usage (aim for 96%+)
        captain, flex_players = self._optimize_salary_usage(captain, flex_players)
        
        # Validate lineup
        validation = self._validate_lineup(captain, flex_players)
        
        if not validation['valid']:
            return None
        
        # Calculate metrics
        all_players = [captain] + flex_players
        
        # Clean player names before passing to opponent model
        clean_captain = captain.strip()
        clean_flex = [p.strip() for p in flex_players]
        
        metrics = self.opponent_model.calculate_lineup_metrics(
            clean_flex,
            clean_captain
        )
        
        return {
            'captain': captain,
            'flex': flex_players,
            'metrics': metrics,
            'validation': validation
        }
    
    def _validate_lineup(self, captain: str, flex_players: List[str]) -> Dict:
        """
        Validate lineup meets all constraints
        
        Args:
            captain: Captain player name
            flex_players: List of FLEX player names
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'salary_used': 0,
            'salary_pct': 0
        }
        
        # Check captain
        captain_data = self._get_player_data_safe(captain)
        if captain_data is None:
            result['valid'] = False
            result['errors'].append(f"Captain '{captain}' not found")
            return result
        
        # Calculate total salary
        total_salary = captain_data['salary'] * CAPTAIN_MULTIPLIER
        
        for player in flex_players:
            player_data = self._get_player_data_safe(player)
            if player_data is None:
                result['valid'] = False
                result['errors'].append(f"Player '{player}' not found")
                return result
            total_salary += player_data['salary']
        
        result['salary_used'] = total_salary
        result['salary_pct'] = (total_salary / SALARY_CAP) * 100
        
        # CRITICAL: Validate salary usage (must be 96%+)
        if total_salary > SALARY_CAP:
            result['valid'] = False
            result['errors'].append(f"Over salary cap: ${total_salary:,} > ${SALARY_CAP:,}")
        elif total_salary < int(SALARY_CAP * 0.96):
            result['valid'] = False
            result['errors'].append(
                f"Under 96% cap usage: ${total_salary:,} < ${int(SALARY_CAP * 0.96):,}"
            )
        
        # Check roster size
        if len(flex_players) != ROSTER_SIZE - 1:
            result['valid'] = False
            result['errors'].append(f"Invalid roster size: {len(flex_players) + 1} (need {ROSTER_SIZE})")
        
        # Check for duplicates
        all_players = [captain] + flex_players
        if len(all_players) != len(set(all_players)):
            result['valid'] = False
            result['errors'].append("Duplicate players in lineup")
        
        return result
    
    def get_portfolio_analysis(self) -> Dict:
        """
        Analyze the generated lineup portfolio with safe lookups
        
        Returns:
            Dictionary with portfolio statistics
        """
        if not self.generated_lineups:
            return {
                'total_lineups': 0,
                'unique_lineups': 0,
                'avg_projection': 0,
                'avg_ownership': 0,
                'avg_leverage': 0,
                'player_exposure': {}
            }
        
        # Count unique lineups
        unique_sigs = len(self.lineup_signatures)
        
        # Calculate averages
        total_proj = 0
        total_own = 0
        total_lev = 0
        player_counts = {}
        
        for lineup in self.generated_lineups:
            metrics = lineup['metrics']
            total_proj += metrics.get('total_projection', 0)
            total_own += metrics.get('avg_ownership', 0)
            total_lev += metrics.get('lineup_leverage', 0)
            
            # Count player usage
            all_players = [lineup['captain']] + lineup['flex']
            for player in all_players:
                player_counts[player] = player_counts.get(player, 0) + 1
        
        num_lineups = len(self.generated_lineups)
        
        # Calculate exposure percentages with safe lookups
        player_exposure = {}
        for player, count in player_counts.items():
            player_data = self._get_player_data_safe(player)
            
            if player_data is not None:
                player_exposure[player] = {
                    'count': count,
                    'exposure': (count / num_lineups) * 100,
                    'ownership': player_data.get('ownership', 15.0),
                    'projection': player_data.get('projection', 0)
                }
            else:
                # Use placeholder data if player not found
                player_exposure[player] = {
                    'count': count,
                    'exposure': (count / num_lineups) * 100,
                    'ownership': 15.0,
                    'projection': 0
                }
        
        return {
            'total_lineups': num_lineups,
            'unique_lineups': unique_sigs,
            'avg_projection': total_proj / num_lineups if num_lineups > 0 else 0,
            'avg_ownership': total_own / num_lineups if num_lineups > 0 else 0,
            'avg_leverage': total_lev / num_lineups if num_lineups > 0 else 0,
            'player_exposure': player_exposure
        }
    
    def export_to_csv(self, filename: str = 'lineups.csv'):
        """
        Export generated lineups to CSV format
        
        Args:
            filename: Output filename
        """
        if not self.generated_lineups:
            print("No lineups to export")
            return
        
        rows = []
        for i, lineup in enumerate(self.generated_lineups, 1):
            # DraftKings format
            row = {
                'CPT': lineup['captain'],
                'FLEX1': lineup['flex'][0] if len(lineup['flex']) > 0 else '',
                'FLEX2': lineup['flex'][1] if len(lineup['flex']) > 1 else '',
                'FLEX3': lineup['flex'][2] if len(lineup['flex']) > 2 else '',
                'FLEX4': lineup['flex'][3] if len(lineup['flex']) > 3 else '',
                'FLEX5': lineup['flex'][4] if len(lineup['flex']) > 4 else '',
                'Projection': f"{lineup['metrics']['total_projection']:.1f}",
                'Salary': lineup['metrics']['total_salary'],
                'Ownership': f"{lineup['metrics']['avg_ownership']:.1f}",
                'Leverage': f"{lineup['metrics']['lineup_leverage']:.2f}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(rows)} lineups to {filename}")
