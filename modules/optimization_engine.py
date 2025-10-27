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
        max_attempts = num_lineups * 10  # Allow retries for duplicate prevention
        
        for i in range(num_lineups):
            while attempts < max_attempts:
                attempts += 1
                
                lineup = self._generate_single_lineup(
                    mode_config=mode_config,
                    max_exposure=max_exposure,
                    iteration=i,
                    total_lineups=num_lineups
                )
                
                if not lineup:
                    continue
                
                # CRITICAL FIX: Check for duplicates
                lineup_signature = self._get_lineup_signature(lineup)
                
                if lineup_signature in seen_lineups:
                    # Duplicate detected - try again
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
                
                break  # Success - move to next lineup
        
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
            total_lineups: Total number of lineups to generate
        
        Returns:
            Lineup dictionary or None if generation failed
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
        
        return selected_flex
    
    def get_portfolio_analysis(self) -> Dict:
        """
        Analyze the complete portfolio of generated lineups
        
        Returns:
            Dictionary with portfolio statistics
        """
        if not self.generated_lineups:
            return {}
        
        # Calculate aggregate metrics
        all_projections = [lu['metrics']['total_projection'] 
                          for lu in self.generated_lineups]
        all_ceilings = [lu['metrics']['total_ceiling'] 
                       for lu in self.generated_lineups]
        all_ownership = [lu['metrics']['avg_ownership'] 
                        for lu in self.generated_lineups]
        all_uniqueness = [lu['metrics']['uniqueness'] 
                         for lu in self.generated_lineups]
        
        # Player exposure analysis
        total_lineups = len(self.generated_lineups)
        exposure_data = []
        
        for player_name, usage_count in self.player_usage.items():
            if usage_count > 0:
                exposure_pct = (usage_count / total_lineups) * 100
                
                # CRITICAL FIX: Use safe player lookup
                player_data = self._get_player_data_safe(player_name)
                
                if player_data is not None:
                    exposure_data.append({
                        'name': player_name,
                        'usage_count': usage_count,
                        'exposure_pct': exposure_pct,
                        'ownership': player_data['ownership'],
                        'leverage': player_data['leverage']
                    })
                else:
                    # Player not found - use placeholder data
                    print(f"WARNING: Skipping exposure data for '{player_name}' - not found in DataFrame")
                    exposure_data.append({
                        'name': player_name,
                        'usage_count': usage_count,
                        'exposure_pct': exposure_pct,
                        'ownership': 15.0,  # Default
                        'leverage': 100.0   # Default
                    })
        
        # Handle empty exposure_data
        if not exposure_data:
            return {
                'total_lineups': total_lineups,
                'avg_projection': np.mean(all_projections) if all_projections else 0,
                'avg_ceiling': np.mean(all_ceilings) if all_ceilings else 0,
                'avg_ownership': np.mean(all_ownership) if all_ownership else 0,
                'avg_uniqueness': np.mean(all_uniqueness) if all_uniqueness else 0,
                'projection_range': (min(all_projections), max(all_projections)) if all_projections else (0, 0),
                'ceiling_range': (min(all_ceilings), max(all_ceilings)) if all_ceilings else (0, 0),
                'ownership_range': (min(all_ownership), max(all_ownership)) if all_ownership else (0, 0),
                'player_exposure': pd.DataFrame(),
                'unique_players_used': 0,
                'most_exposed': [],
                'least_exposed': []
            }
        
        exposure_df = pd.DataFrame(exposure_data).sort_values(
            'exposure_pct',
            ascending=False
        )
        
        return {
            'total_lineups': total_lineups,
            'avg_projection': np.mean(all_projections),
            'avg_ceiling': np.mean(all_ceilings),
            'avg_ownership': np.mean(all_ownership),
            'avg_uniqueness': np.mean(all_uniqueness),
            'projection_range': (min(all_projections), max(all_projections)),
            'ceiling_range': (min(all_ceilings), max(all_ceilings)),
            'ownership_range': (min(all_ownership), max(all_ownership)),
            'player_exposure': exposure_df,
            'unique_players_used': len(exposure_df),
            'most_exposed': exposure_df.head(5).to_dict('records') if len(exposure_df) >= 5 else exposure_df.to_dict('records'),
            'least_exposed': exposure_df.tail(5).to_dict('records') if len(exposure_df) >= 5 else exposure_df.to_dict('records')
        }
    
    def compare_to_traditional(self) -> Dict:
        """
        Compare opponent-modeled lineups to traditional optimization
        
        Returns:
            Comparison statistics
        """
        if not self.generated_lineups:
            return {}
        
        # Generate a "traditional" lineup (pure projection optimization)
        traditional_config = {
            'leverage_weight': 0.0,
            'projection_weight': 1.0,
            'prioritize_contrarian': False
        }
        
        traditional_scores = self._calculate_player_scores(traditional_config, 0, 1)
        traditional_players = self.players_df.merge(
            traditional_scores,
            on='name'
        ).nlargest(10, 'optimizer_score')
        
        # Compare metrics
        our_avg_ownership = np.mean([
            lu['metrics']['avg_ownership'] 
            for lu in self.generated_lineups
        ])
        
        our_avg_projection = np.mean([
            lu['metrics']['total_projection'] 
            for lu in self.generated_lineups
        ])
        
        traditional_avg_ownership = traditional_players['ownership'].mean()
        
        return {
            'our_avg_ownership': our_avg_ownership,
            'traditional_avg_ownership': traditional_avg_ownership,
            'ownership_difference': our_avg_ownership - traditional_avg_ownership,
            'our_avg_projection': our_avg_projection,
            'differentiation_score': abs(our_avg_ownership - traditional_avg_ownership),
            'uniqueness_advantage': 100 - our_avg_ownership
        }
    
    def export_lineups(self, filename: str = 'lineups.csv'):
        """
        Export lineups to CSV for DraftKings upload
        
        Args:
            filename: Output filename
        """
        if not self.generated_lineups:
            return
        
        export_data = []
        
        for lineup in self.generated_lineups:
            row = {
                'Lineup': lineup['lineup_id'],
                'CPT': lineup['captain'],
            }
            
            # Add flex spots
            for i, flex_player in enumerate(lineup['flex'], 1):
                row[f'FLEX{i}'] = flex_player
            
            # Add metrics
            row['Total_Salary'] = int(lineup['metrics']['total_salary'])
            row['Projected'] = round(lineup['metrics']['total_projection'], 2)
            row['Ceiling'] = round(lineup['metrics']['total_ceiling'], 2)
            row['Ownership'] = round(lineup['metrics']['avg_ownership'], 2)
            row['Uniqueness'] = round(lineup['metrics']['uniqueness'], 2)
            
            export_data.append(row)
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(filename, index=False)
        
        return filename
