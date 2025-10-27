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
        self.players_df = players_df.copy()
        
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
        print(f"Available names: {self.players_df['name'].tolist()[:10]}...")
        return None
    
    def generate_lineups(self,
                        num_lineups: int,
                        mode: str = 'balanced',
                        max_exposure: float = MAX_PLAYER_EXPOSURE,
                        enforce_diversity: bool = True) -> List[Dict]:
        """
        Generate multiple optimized lineups
        
        Args:
            num_lineups: Number of lineups to generate
            mode: Optimization mode (anti_chalk, leverage, balanced, safe)
            max_exposure: Maximum % of lineups a player can appear in
            enforce_diversity: Whether to enforce minimum player differences
        
        Returns:
            List of lineup dictionaries
        """
        self.generated_lineups = []
        self.player_usage = {name: 0 for name in self.players_df['name']}
        
        # CRITICAL FIX: Track generated lineups to prevent duplicates
        seen_lineups = set()
        
        mode_config = OPTIMIZATION_MODES.get(mode, OPTIMIZATION_MODES['balanced'])
        
        attempts = 0
        max_attempts = num_lineups * 15  # Allow more retries for salary validation
        
        rejected_salary_count = 0
        rejected_duplicate_count = 0
        
        print(f"\n🎯 Generating {num_lineups} lineups in {mode} mode...")
        print(f"   Salary cap: ${SALARY_CAP:,}")
        print(f"   Required usage: 96%+ (${SALARY_CAP * 0.96:,.0f}+)")
        
        for i in range(num_lineups):
            lineup_found = False
            
            while attempts < max_attempts and not lineup_found:
                attempts += 1
                
                lineup = self._generate_single_lineup(
                    mode_config=mode_config,
                    max_exposure=max_exposure,
                    iteration=i,
                    total_lineups=num_lineups
                )
                
                if not lineup:
                    rejected_salary_count += 1
                    continue
                
                # CRITICAL FIX: Check for duplicates
                lineup_signature = self._get_lineup_signature(lineup)
                
                if lineup_signature in seen_lineups:
                    # Duplicate detected - try again
                    rejected_duplicate_count += 1
                    continue
                
                # CRITICAL FIX: Clean player names in lineup before storing
                lineup['captain'] = str(lineup['captain']).strip()
                lineup['flex'] = [str(p).strip() for p in lineup['flex']]
                lineup['players'] = [str(p).strip() for p in lineup['players']]
                
                # New unique lineup found!
                seen_lineups.add(lineup_signature)
                self.generated_lineups.append(lineup)
                
                # Update usage tracking
                for player in lineup['players']:
                    clean_player = str(player).strip()
                    if clean_player in self.player_usage:
                        self.player_usage[clean_player] += 1
                
                lineup_found = True
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"   ✓ Generated {i + 1}/{num_lineups} lineups (attempts: {attempts})")
        
        print(f"\n✅ Generation complete!")
        print(f"   Total attempts: {attempts}")
        print(f"   Rejected (low salary): {rejected_salary_count}")
        print(f"   Rejected (duplicates): {rejected_duplicate_count}")
        print(f"   Success rate: {(num_lineups/attempts)*100:.1f}%")
        
        return self.generated_lineups
    
    def _get_lineup_signature(self, lineup: Dict) -> str:
        """
        Create a unique signature for a lineup to detect duplicates
        
        Args:
            lineup: Lineup dictionary
            
        Returns:
            String signature of the lineup
        """
        # Sort all players (captain + flex) to create consistent signature
        all_players = sorted([lineup['captain']] + lineup['flex'])
        return '|'.join(all_players)
    
    def _generate_single_lineup(self,
                               mode_config: Dict,
                               max_exposure: float,
                               iteration: int,
                               total_lineups: int) -> Optional[Dict]:
        """
        Generate a single optimized lineup
        
        Args:
            mode_config: Configuration for optimization mode
            max_exposure: Maximum exposure per player
            iteration: Current iteration number
            total_lineups: Total number of lineups being generated
        
        Returns:
            Dictionary with lineup data or None if generation failed
        """
        # Calculate player scores based on mode (with progressive randomization)
        player_scores = self._calculate_player_scores(mode_config, iteration, total_lineups)
        
        # Apply exposure limits
        max_usage = int(total_lineups * max_exposure)
        available_players = self.players_df[
            self.players_df['name'].apply(
                lambda x: self.player_usage.get(str(x).strip(), 0) < max_usage
            )
        ].copy()
        
        if len(available_players) < ROSTER_SIZE:
            return None
        
        # Merge scores with available players
        available_players = available_players.merge(
            player_scores,
            on='name',
            how='left'
        )
        
        # Select captain using strategic scoring (probabilistic)
        captain = self._select_captain(available_players, mode_config, iteration)
        if captain is None:
            return None
        
        # Select flex players (stochastic selection)
        flex_players = self._select_flex_players(
            available_players,
            captain,
            mode_config,
            iteration
        )
        
        if len(flex_players) < ROSTER_SIZE - 1:
            return None
        
        # Build complete lineup
        lineup_players = [captain] + flex_players
        
        # Calculate lineup metrics
        metrics = self.opponent_model.calculate_lineup_metrics(
            lineup_players,
            captain
        )
        
        # CRITICAL FIX: Validate salary usage (should use 96%+ of cap)
        salary_used = metrics['total_salary']
        salary_used_pct = (salary_used / SALARY_CAP) * 100
        
        # DEBUG: Print salary info for troubleshooting
        if salary_used_pct < 96.0:
            if iteration < 3:  # Only print first 3 rejections to avoid spam
                print(f"\n   ⚠️  Lineup {iteration + 1} REJECTED:")
                print(f"       Captain: {captain}")
                print(f"       Salary: ${salary_used:,} ({salary_used_pct:.1f}%)")
                print(f"       Need: ${SALARY_CAP * 0.96:,.0f}+ (96%+)")
            return None
        
        return {
            'lineup_id': iteration + 1,
            'captain': captain,
            'flex': flex_players,
            'players': lineup_players,
            'metrics': metrics,
            'mode': mode_config['description']
        }
    
    def _calculate_player_scores(self, mode_config: Dict, iteration: int = 0, total_lineups: int = 1) -> pd.DataFrame:
        """
        Calculate player scores based on optimization mode
        
        Args:
            mode_config: Mode configuration dictionary
            iteration: Current iteration number (for progressive randomization)
            total_lineups: Total lineups being generated
        
        Returns:
            DataFrame with player names and scores
        """
        leverage_weight = mode_config['leverage_weight']
        projection_weight = mode_config['projection_weight']
        
        # Normalize metrics to 0-1 scale
        leverage_norm = (
            (self.players_df['leverage'] - self.players_df['leverage'].min()) /
            (self.players_df['leverage'].max() - self.players_df['leverage'].min())
        )
        
        projection_norm = (
            (self.players_df['projection'] - self.players_df['projection'].min()) /
            (self.players_df['projection'].max() - self.players_df['projection'].min())
        )
        
        # Calculate weighted score
        base_score = (leverage_norm * leverage_weight) + \
                    (projection_norm * projection_weight)
        
        # Apply chalk penalty if in contrarian mode
        if mode_config.get('prioritize_contrarian', False):
            chalk_penalty = self.players_df['ownership'] / 100
            base_score = base_score * (1 - chalk_penalty * 0.5)  # Increased from 0.3 to 0.5
        
        # CRITICAL FIX: Much stronger randomization for lineup diversity
        # Progressive randomization: more lineups = more variance needed
        progress_factor = min(iteration / max(total_lineups * 0.5, 1), 1.0)
        
        # Start with 15-25% variance, increase to 30-50% variance as we generate more
        min_factor = 0.85 - (progress_factor * 0.15)  # 0.85 -> 0.70
        max_factor = 1.15 + (progress_factor * 0.35)  # 1.15 -> 1.50
        
        random_factor = np.random.uniform(min_factor, max_factor, len(self.players_df))
        final_score = base_score * random_factor
        
        # Add usage-based penalty to promote diversity
        for idx, player_name in enumerate(self.players_df['name']):
            usage_count = self.player_usage.get(player_name, 0)
            if usage_count > 0 and total_lineups > 1:
                usage_penalty = 1.0 - (usage_count / total_lineups) * 0.3
                final_score.iloc[idx] *= usage_penalty
        
        return pd.DataFrame({
            'name': self.players_df['name'],
            'optimizer_score': final_score
        })
    
    def _select_captain(self,
                       available_players: pd.DataFrame,
                       mode_config: Dict,
                       iteration: int = 0) -> Optional[str]:
        """
        Select the optimal captain using probabilistic selection
        
        Args:
            available_players: DataFrame of available players with scores
            mode_config: Mode configuration
            iteration: Current iteration for randomization
        
        Returns:
            Captain player name or None
        """
        # CRITICAL FIX: Probabilistic selection instead of deterministic
        # Get more candidates for better diversity
        num_candidates = min(15, len(available_players))
        captain_candidates = available_players.nlargest(num_candidates, 'optimizer_score')
        
        # Calculate selection probabilities based on scores (not uniform!)
        scores = captain_candidates['optimizer_score'].values
        # Softmax-like probabilities with temperature
        temperature = 1.5 + (iteration * 0.05)  # Increase randomness over iterations
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / exp_scores.sum()
        
        # Try candidates in random order weighted by probability
        candidate_indices = np.random.choice(
            len(captain_candidates),
            size=min(num_candidates, 10),
            replace=False,
            p=probabilities
        )
        
        for idx in candidate_indices:
            player = captain_candidates.iloc[idx]
            captain_salary = player['salary'] * CAPTAIN_MULTIPLIER
            remaining = SALARY_CAP - captain_salary
            
            # Check if we can build a valid lineup with this captain
            min_flex_salary = available_players[
                available_players['name'] != player['name']
            ].nsmallest(5, 'salary')['salary'].sum()
            
            if min_flex_salary <= remaining:
                return player['name']
        
        return None
    
    def _select_flex_players(self,
                            available_players: pd.DataFrame,
                            captain: str,
                            mode_config: Dict,
                            iteration: int = 0) -> List[str]:
        """
        Select optimal flex players using stochastic selection
        
        Args:
            available_players: DataFrame of available players
            captain: Selected captain name
            mode_config: Mode configuration
            iteration: Current iteration for randomization
        
        Returns:
            List of flex player names
        """
        captain_data = available_players[
            available_players['name'] == captain
        ].iloc[0]
        
        captain_salary = captain_data['salary'] * CAPTAIN_MULTIPLIER
        remaining_salary = SALARY_CAP - captain_salary
        
        # Get eligible flex players
        flex_candidates = available_players[
            available_players['name'] != captain
        ].copy()
        
        # Apply team correlation bonus
        flex_candidates['adjusted_score'] = flex_candidates['optimizer_score']
        same_team_mask = flex_candidates['team'] == captain_data['team']
        flex_candidates.loc[same_team_mask, 'adjusted_score'] *= 1.15
        
        # CRITICAL FIX: Stochastic selection instead of pure greedy
        # Use multiple selection strategies based on iteration
        
        if iteration % 3 == 0:
            # Strategy 1: Probabilistic weighted selection
            selected_flex = self._stochastic_flex_selection(
                flex_candidates, remaining_salary, temperature=1.2
            )
        elif iteration % 3 == 1:
            # Strategy 2: High variance selection
            selected_flex = self._stochastic_flex_selection(
                flex_candidates, remaining_salary, temperature=2.0
            )
        else:
            # Strategy 3: Mixed strategy (some top, some random)
            selected_flex = self._mixed_flex_selection(
                flex_candidates, remaining_salary
            )
        
        return selected_flex
    
    def _optimize_lineup_salary(self,
                                selected_flex: List[str],
                                flex_candidates: pd.DataFrame,
                                remaining_salary: int,
                                current_salary: int) -> List[str]:
        """
        Optimize lineup to use more salary cap
        
        Tries to upgrade lower-salary players to higher-salary players
        to maximize cap usage while maintaining lineup quality.
        
        Args:
            selected_flex: Currently selected flex players
            flex_candidates: All available flex candidates
            remaining_salary: Total remaining salary after captain
            current_salary: Current salary used by flex players
        
        Returns:
            Optimized list of flex player names
        """
        if len(selected_flex) < ROSTER_SIZE - 1:
            # Lineup incomplete - don't optimize yet
            return selected_flex
        
        salary_left = remaining_salary - current_salary
        min_acceptable_remaining = 1000  # Should use 98% of cap
        
        # If we're already using enough salary, return
        if salary_left < min_acceptable_remaining:
            return selected_flex
        
        # Get current players' data
        selected_names = set(selected_flex)
        selected_players = flex_candidates[
            flex_candidates['name'].isin(selected_names)
        ]
        
        if len(selected_players) == 0:
            # Couldn't find players - return original
            return selected_flex
        
        # Try to upgrade players (replace cheaper with more expensive)
        max_attempts = 10
        for attempt in range(max_attempts):
            if salary_left < min_acceptable_remaining:
                break
            
            # Find cheapest player in lineup
            cheapest = selected_players.nsmallest(1, 'salary').iloc[0]
            cheapest_salary = cheapest['salary']
            cheapest_name = cheapest['name']
            
            # Find upgrade candidates (better players we can afford)
            upgrade_budget = cheapest_salary + salary_left
            
            upgrade_candidates = flex_candidates[
                (~flex_candidates['name'].isin(selected_names)) &
                (flex_candidates['salary'] > cheapest_salary) &
                (flex_candidates['salary'] <= upgrade_budget)
            ]
            
            if len(upgrade_candidates) == 0:
                break
            
            # Pick best upgrade by adjusted score
            upgrade = upgrade_candidates.nlargest(1, 'adjusted_score').iloc[0]
            
            # Make the swap
            selected_flex.remove(cheapest_name)
            selected_flex.append(upgrade['name'])
            
            # Update tracking
            salary_diff = upgrade['salary'] - cheapest['salary']
            current_salary += salary_diff
            salary_left -= salary_diff
            
            selected_names.remove(cheapest_name)
            selected_names.add(upgrade['name'])
            
            # Update selected_players DataFrame
            selected_players = flex_candidates[
                flex_candidates['name'].isin(selected_names)
            ]
        
        return selected_flex
    
    def _stochastic_flex_selection(self,
                                   flex_candidates: pd.DataFrame,
                                   remaining_salary: int,
                                   temperature: float = 1.5) -> List[str]:
        """
        Probabilistic flex player selection
        
        Args:
            flex_candidates: Available flex players
            remaining_salary: Remaining salary cap
            temperature: Controls randomness (higher = more random)
        
        Returns:
            List of selected flex player names
        """
        selected_flex = []
        current_salary = 0
        available = flex_candidates.copy()
        
        for _ in range(ROSTER_SIZE - 1):
            if len(available) == 0:
                break
            
            # Filter by salary constraint
            affordable = available[
                available['salary'] <= (remaining_salary - current_salary)
            ]
            
            if len(affordable) == 0:
                break
            
            # Calculate selection probabilities
            scores = affordable['adjusted_score'].values
            exp_scores = np.exp(scores / temperature)
            probabilities = exp_scores / exp_scores.sum()
            
            # Select one player probabilistically
            selected_idx = np.random.choice(len(affordable), p=probabilities)
            selected_player = affordable.iloc[selected_idx]
            
            selected_flex.append(selected_player['name'])
            current_salary += selected_player['salary']
            
            # Remove selected player from pool
            available = available[available['name'] != selected_player['name']]
        
        # CRITICAL FIX: Optimize salary usage
        if len(selected_flex) == ROSTER_SIZE - 1:
            selected_flex = self._optimize_lineup_salary(
                selected_flex,
                flex_candidates,
                remaining_salary,
                current_salary
            )
        
        return selected_flex
    
    def _mixed_flex_selection(self,
                             flex_candidates: pd.DataFrame,
                             remaining_salary: int) -> List[str]:
        """
        Mixed strategy: take some top players, some random
        
        Args:
            flex_candidates: Available flex players
            remaining_salary: Remaining salary cap
        
        Returns:
            List of selected flex player names
        """
        selected_flex = []
        current_salary = 0
        
        # Sort by score
        sorted_candidates = flex_candidates.sort_values(
            'adjusted_score',
            ascending=False
        )
        
        # Take 2-3 top players
        num_top = np.random.randint(2, 4)
        
        for _, player in sorted_candidates.iterrows():
            if len(selected_flex) >= num_top:
                break
            
            if current_salary + player['salary'] <= remaining_salary:
                selected_flex.append(player['name'])
                current_salary += player['salary']
        
        # Fill rest with weighted random selection
        available = flex_candidates[
            ~flex_candidates['name'].isin(selected_flex)
        ]
        
        while len(selected_flex) < ROSTER_SIZE - 1 and len(available) > 0:
            affordable = available[
                available['salary'] <= (remaining_salary - current_salary)
            ]
            
            if len(affordable) == 0:
                break
            
            # Weighted random selection
            scores = affordable['adjusted_score'].values
            probabilities = scores / scores.sum()
            
            selected_idx = np.random.choice(len(affordable), p=probabilities)
            selected_player = affordable.iloc[selected_idx]
            
            selected_flex.append(selected_player['name'])
            current_salary += selected_player['salary']
            
            available = available[available['name'] != selected_player['name']]
        
        # CRITICAL FIX: Optimize salary usage
        if len(selected_flex) == ROSTER_SIZE - 1:
            selected_flex = self._optimize_lineup_salary(
                selected_flex,
                flex_candidates,
                remaining_salary,
                current_salary
            )
        
        return selected_flex
    
    def get_portfolio_analysis(self) -> Dict:
        """
        Analyze the generated lineup portfolio
        
        Returns:
            Dictionary with portfolio statistics
        """
        if not self.generated_lineups:
            return {}
        
        analysis = {
            'total_lineups': len(self.generated_lineups),
            'player_exposure': {},
            'salary_stats': {},
            'projection_stats': {},
            'ownership_stats': {}
        }
        
        # Calculate player exposure
        for player_name, usage in self.player_usage.items():
            if usage > 0:
                exposure_pct = (usage / len(self.generated_lineups)) * 100
                
                # CRITICAL FIX: Safe player lookup
                player_data = self._get_player_data_safe(player_name)
                
                if player_data is not None:
                    analysis['player_exposure'][player_name] = {
                        'count': usage,
                        'exposure': exposure_pct,
                        'salary': player_data['salary'],
                        'projection': player_data['projection']
                    }
        
        # Salary statistics
        salaries = [lu['metrics']['total_salary'] for lu in self.generated_lineups]
        analysis['salary_stats'] = {
            'min': min(salaries),
            'max': max(salaries),
            'avg': sum(salaries) / len(salaries),
            'avg_remaining': SALARY_CAP - (sum(salaries) / len(salaries))
        }
        
        # Projection statistics
        projections = [lu['metrics']['total_projection'] for lu in self.generated_lineups]
        analysis['projection_stats'] = {
            'min': min(projections),
            'max': max(projections),
            'avg': sum(projections) / len(projections)
        }
        
        # Ownership statistics
        ownerships = [lu['metrics']['avg_ownership'] for lu in self.generated_lineups]
        analysis['ownership_stats'] = {
            'min': min(ownerships),
            'max': max(ownerships),
            'avg': sum(ownerships) / len(ownerships)
        }
        
        return analysis
    
    def export_to_csv(self, filename: str = 'lineups.csv'):
        """
        Export lineups to CSV format for DFS sites
        
        Args:
            filename: Output filename
        """
        if not self.generated_lineups:
            print("No lineups to export!")
            return
        
        export_data = []
        
        for lineup in self.generated_lineups:
            row = {
                'CPT': lineup['captain'],
                'FLEX1': lineup['flex'][0] if len(lineup['flex']) > 0 else '',
                'FLEX2': lineup['flex'][1] if len(lineup['flex']) > 1 else '',
                'FLEX3': lineup['flex'][2] if len(lineup['flex']) > 2 else '',
                'FLEX4': lineup['flex'][3] if len(lineup['flex']) > 3 else '',
                'FLEX5': lineup['flex'][4] if len(lineup['flex']) > 4 else '',
                'Salary': lineup['metrics']['total_salary'],
                'Projection': lineup['metrics']['total_projection'],
                'Ownership': lineup['metrics']['avg_ownership']
            }
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(self.generated_lineups)} lineups to {filename}")
    
    def compare_to_traditional(self) -> Dict:
        """
        Compare this optimizer's approach to traditional projection-only optimization
        
        Returns:
            Dictionary with comparison metrics
        """
        # Generate a "traditional" lineup (pure projection optimization)
        traditional_config = {
            'leverage_weight': 0.0,
            'projection_weight': 1.0,
            'prioritize_contrarian': False
        }
        
        traditional_scores = self._calculate_player_scores(traditional_config, 0, 1)
        
        # Calculate average leverage and ownership for our lineups vs traditional
        if not self.generated_lineups:
            return {}
        
        our_avg_ownership = sum(
            lu['metrics']['avg_ownership'] for lu in self.generated_lineups
        ) / len(self.generated_lineups)
        
        our_avg_leverage = sum(
            lu['metrics']['lineup_leverage'] for lu in self.generated_lineups
        ) / len(self.generated_lineups)
        
        return {
            'our_approach': {
                'avg_ownership': our_avg_ownership,
                'avg_leverage': our_avg_leverage,
                'strategy': 'Opponent modeling + leverage optimization'
            },
            'traditional_approach': {
                'strategy': 'Pure projection maximization',
                'typical_ownership': 'Higher (follows chalk)',
                'typical_leverage': 'Lower (ignores ownership)'
            }
        }
