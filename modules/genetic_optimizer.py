"""
Module 2: Genetic Optimizer
Enhanced genetic algorithm with multi-objective optimization
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LineupCandidate:
    """Represents a candidate lineup in the genetic algorithm"""
    players: List[str]
    fitness: float
    metrics: Dict[str, float]


class GeneticOptimizer:
    """
    Enhanced genetic algorithm for DFS lineup optimization.
    
    Key Features:
    - Multi-objective fitness function (5 components)
    - Adaptive mutation rates
    - Tournament selection
    - Elite preservation
    - Constraint repair (salary, duplicates)
    """
    
    def __init__(
        self,
        players_df: pd.DataFrame,
        opponent_model,
        stacking_engine,
        salary_cap: int = 50000
    ):
        """
        Initialize genetic optimizer.
        
        Args:
            players_df: DataFrame with player data
            opponent_model: OpponentModel instance
            stacking_engine: StackingEngine instance
            salary_cap: Maximum salary allowed
        """
        self.players_df = players_df.copy()
        self.opponent_model = opponent_model
        self.stacking_engine = stacking_engine
        self.salary_cap = salary_cap
        
        # GA parameters
        self.population_size = 200
        self.generations = 100
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.elite_size = 20
        self.tournament_size = 5
        
        # Fitness weights (will be adjusted by mode)
        self.weights = {
            'projection': 0.30,
            'ceiling': 0.25,
            'leverage': 0.25,
            'correlation': 0.10,
            'ownership': 0.10
        }
        
        # Evolution tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        logger.info("GeneticOptimizer initialized")
    
    def set_mode_weights(self, mode: str):
        """
        Adjust fitness weights based on optimization mode.
        
        Modes:
        - CASH: Focus on projection and floor
        - GPP: Focus on ceiling and leverage
        - CONTRARIAN: Focus on low ownership
        """
        if mode == 'CASH':
            self.weights = {
                'projection': 0.50,
                'ceiling': 0.15,
                'leverage': 0.10,
                'correlation': 0.20,
                'ownership': 0.05
            }
        elif mode == 'GPP':
            self.weights = {
                'projection': 0.25,
                'ceiling': 0.30,
                'leverage': 0.25,
                'correlation': 0.10,
                'ownership': 0.10
            }
        elif mode == 'CONTRARIAN':
            self.weights = {
                'projection': 0.20,
                'ceiling': 0.25,
                'leverage': 0.20,
                'correlation': 0.10,
                'ownership': 0.25
            }
        else:  # DEFAULT/BALANCED
            self.weights = {
                'projection': 0.30,
                'ceiling': 0.25,
                'leverage': 0.25,
                'correlation': 0.10,
                'ownership': 0.10
            }
        
        logger.info(f"Mode set to {mode}, weights updated")
    
    def evolve(
        self,
        target_mode: str = 'GPP',
        enforce_stack: bool = True,
        max_ownership: float = None,
        num_lineups: int = 20
    ) -> List[Dict]:
        """
        Run genetic algorithm evolution.
        
        Args:
            target_mode: Optimization mode (CASH, GPP, CONTRARIAN)
            enforce_stack: Require QB stacks in lineups
            max_ownership: Maximum total ownership threshold
            num_lineups: Number of distinct lineups to generate
        
        Returns:
            List of optimized lineups
        """
        logger.info(f"Starting evolution: {self.generations} generations, population {self.population_size}")
        
        # Set mode-specific weights
        self.set_mode_weights(target_mode)
        
        # Initialize population
        population = self._initialize_population(enforce_stack)
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._calculate_fitness(ind) for ind in population]
            
            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
            
            # Selection
            parents = self._tournament_selection(population, fitness_scores)
            
            # Create next generation
            offspring = []
            
            # Elite preservation
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [population[i] for i in elite_indices]
            offspring.extend(elite)
            
            # Generate rest through crossover and mutation
            while len(offspring) < self.population_size:
                # Select two parents
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation (adaptive rate)
                mutation_rate = self._adaptive_mutation_rate(generation)
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                # Repair constraints
                child = self._repair_lineup(child, enforce_stack)
                
                offspring.append(child)
            
            population = offspring
        
        # Final evaluation
        final_fitness = [self._calculate_fitness(ind) for ind in population]
        
        # Get top diverse lineups
        top_lineups = self._extract_diverse_lineups(
            population,
            final_fitness,
            num_lineups,
            max_ownership
        )
        
        logger.info(f"Evolution complete. Generated {len(top_lineups)} lineups")
        
        return top_lineups
    
    def _initialize_population(self, enforce_stack: bool) -> List[List[str]]:
        """Create initial random population"""
        population = []
        max_attempts = self.population_size * 10
        attempts = 0
        
        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            # Random lineup
            lineup = self._generate_random_lineup()
            
            # Repair to ensure validity
            lineup = self._repair_lineup(lineup, enforce_stack)
            
            if lineup:
                population.append(lineup)
        
        # Fill remaining with duplicates if needed
        while len(population) < self.population_size:
            population.append(random.choice(population).copy())
        
        logger.info(f"Population initialized: {len(population)} individuals")
        
        return population
    
    def _generate_random_lineup(self) -> List[str]:
        """Generate a random valid lineup"""
        # Showdown format: 1 CPT + 5 FLEX
        lineup = []
        remaining_salary = self.salary_cap
        
        # Select captain (1.5x salary multiplier)
        available_captains = self.players_df.copy()
        captain = available_captains.sample(1).iloc[0]
        lineup.append(captain['name'])
        remaining_salary -= int(captain['salary'] * 1.5)
        
        # Select 5 flex players
        for _ in range(5):
            # Available players not in lineup
            available = self.players_df[
                (~self.players_df['name'].isin(lineup)) &
                (self.players_df['salary'] <= remaining_salary)
            ]
            
            if available.empty:
                break
            
            # Random selection
            player = available.sample(1).iloc[0]
            lineup.append(player['name'])
            remaining_salary -= player['salary']
        
        return lineup
    
    def _calculate_fitness(self, lineup: List[str]) -> float:
        """
        Calculate multi-objective fitness score.
        
        Components:
        1. Projection (expected points)
        2. Ceiling (upside potential)
        3. Leverage (ceiling/ownership)
        4. Correlation (stacking bonus)
        5. Ownership (contrarian value)
        """
        lineup_df = self.players_df[self.players_df['name'].isin(lineup)]
        
        if lineup_df.empty or len(lineup) != 6:
            return 0.0
        
        # Component 1: Projection
        captain_name = lineup[0]
        captain_proj = lineup_df[lineup_df['name'] == captain_name]['projection'].values[0] * 1.5
        flex_proj = lineup_df[lineup_df['name'] != captain_name]['projection'].sum()
        total_projection = captain_proj + flex_proj
        projection_score = total_projection / 150.0  # Normalize to ~0-1
        
        # Component 2: Ceiling
        captain_ceil = lineup_df[lineup_df['name'] == captain_name]['ceiling'].values[0] * 1.5
        flex_ceil = lineup_df[lineup_df['name'] != captain_name]['ceiling'].sum()
        total_ceiling = captain_ceil + flex_ceil
        ceiling_score = total_ceiling / 200.0  # Normalize to ~0-1
        
        # Component 3: Leverage
        leverage_scores = []
        for player in lineup:
            player_data = lineup_df[lineup_df['name'] == player].iloc[0]
            ceiling = player_data['ceiling']
            ownership = max(player_data['ownership'], 1.0)  # Avoid division by zero
            leverage = ceiling / ownership
            leverage_scores.append(leverage)
        avg_leverage = np.mean(leverage_scores)
        leverage_score = min(avg_leverage / 3.0, 1.0)  # Normalize to 0-1
        
        # Component 4: Correlation
        correlation = self.stacking_engine.calculate_lineup_correlation(lineup)
        correlation_score = correlation / 100.0  # Already 0-100, normalize to 0-1
        
        # Component 5: Ownership (lower is better for contrarian)
        total_ownership = lineup_df['ownership'].sum()
        ownership_score = 1.0 - (total_ownership / 300.0)  # Inverse, normalize to ~0-1
        
        # Weighted sum
        fitness = (
            self.weights['projection'] * projection_score +
            self.weights['ceiling'] * ceiling_score +
            self.weights['leverage'] * leverage_score +
            self.weights['correlation'] * correlation_score +
            self.weights['ownership'] * ownership_score
        )
        
        # Bonus for QB stacks
        if self.stacking_engine._has_qb_stack(lineup, min_size=2):
            fitness *= 1.05  # 5% bonus
        
        return fitness
    
    def _tournament_selection(
        self,
        population: List[List[str]],
        fitness_scores: List[float]
    ) -> List[List[str]]:
        """Select parents using tournament selection"""
        parents = []
        
        for _ in range(len(population) // 2):
            # Random tournament
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """Two-point crossover"""
        if len(parent1) != len(parent2):
            return parent1.copy()
        
        # Two-point crossover
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1))
        
        # Create child
        child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        
        # Remove duplicates (keep first occurrence)
        seen = set()
        unique_child = []
        for player in child:
            if player not in seen:
                seen.add(player)
                unique_child.append(player)
        
        return unique_child
    
    def _mutate(self, lineup: List[str]) -> List[str]:
        """Mutate lineup by replacing a random player"""
        if len(lineup) == 0:
            return lineup
        
        # Select random position to mutate
        mutate_idx = random.randint(0, len(lineup) - 1)
        
        # Get available replacements
        available = self.players_df[~self.players_df['name'].isin(lineup)]
        
        if available.empty:
            return lineup
        
        # Weighted selection (higher projection = higher chance)
        weights = available['projection'].values
        weights = weights / weights.sum()
        
        # Select replacement
        replacement = np.random.choice(available['name'].values, p=weights)
        
        # Create new lineup
        new_lineup = lineup.copy()
        new_lineup[mutate_idx] = replacement
        
        return new_lineup
    
    def _adaptive_mutation_rate(self, generation: int) -> float:
        """Decrease mutation rate as evolution progresses"""
        progress = generation / self.generations
        return self.mutation_rate * (1.0 - 0.5 * progress)  # Reduce by 50% over time
    
    def _repair_lineup(
        self,
        lineup: List[str],
        enforce_stack: bool
    ) -> Optional[List[str]]:
        """
        Repair lineup to satisfy constraints.
        
        Constraints:
        - Exactly 6 players
        - No duplicates
        - Under salary cap
        - Has QB stack (if enforced)
        """
        if not lineup:
            return None
        
        # Remove duplicates
        lineup = list(dict.fromkeys(lineup))
        
        # Ensure 6 players
        max_attempts = 50
        attempts = 0
        
        while len(lineup) < 6 and attempts < max_attempts:
            attempts += 1
            
            # Get current salary
            lineup_df = self.players_df[self.players_df['name'].isin(lineup)]
            captain_salary = lineup_df.iloc[0]['salary'] * 1.5 if len(lineup) > 0 else 0
            flex_salary = lineup_df.iloc[1:]['salary'].sum() if len(lineup) > 1 else 0
            current_salary = captain_salary + flex_salary
            remaining_salary = self.salary_cap - current_salary
            
            # Add affordable player
            available = self.players_df[
                (~self.players_df['name'].isin(lineup)) &
                (self.players_df['salary'] <= remaining_salary)
            ]
            
            if available.empty:
                break
            
            # Weighted selection
            weights = available['projection'].values
            if weights.sum() > 0:
                weights = weights / weights.sum()
                new_player = np.random.choice(available['name'].values, p=weights)
                lineup.append(new_player)
        
        # Check salary constraint
        if len(lineup) == 6:
            lineup_df = self.players_df[self.players_df['name'].isin(lineup)]
            captain_salary = lineup_df.iloc[0]['salary'] * 1.5
            flex_salary = lineup_df.iloc[1:]['salary'].sum()
            total_salary = captain_salary + flex_salary
            
            if total_salary > self.salary_cap:
                return None
        
        # Check stack constraint
        if enforce_stack and len(lineup) == 6:
            if not self.stacking_engine._has_qb_stack(lineup, min_size=2):
                # Try to add QB stack
                lineup = self._add_qb_stack(lineup)
        
        return lineup if len(lineup) == 6 else None
    
    def _add_qb_stack(self, lineup: List[str]) -> List[str]:
        """Try to modify lineup to include QB stack"""
        lineup_df = self.players_df[self.players_df['name'].isin(lineup)]
        
        # Find QBs in lineup
        qbs = lineup_df[lineup_df['position'] == 'QB']
        
        if qbs.empty:
            # Add a QB
            qb_available = self.players_df[
                (self.players_df['position'] == 'QB') &
                (~self.players_df['name'].isin(lineup))
            ]
            
            if not qb_available.empty:
                # Replace lowest projection player with QB
                lowest_proj_idx = lineup_df['projection'].idxmin()
                lowest_player = lineup_df.loc[lowest_proj_idx, 'name']
                
                best_qb = qb_available.nlargest(1, 'projection').iloc[0]
                lineup = [best_qb['name'] if p == lowest_player else p for p in lineup]
                
                lineup_df = self.players_df[self.players_df['name'].isin(lineup)]
                qbs = lineup_df[lineup_df['position'] == 'QB']
        
        # Add pass-catcher from same team
        if not qbs.empty:
            qb = qbs.iloc[0]
            
            # Find pass-catchers from QB's team
            pass_catchers = self.players_df[
                (self.players_df['team'] == qb['team']) &
                (self.players_df['position'].isin(['WR', 'TE'])) &
                (~self.players_df['name'].isin(lineup))
            ]
            
            if not pass_catchers.empty:
                # Replace lowest projection player with pass-catcher
                non_qb_lineup = lineup_df[lineup_df['position'] != 'QB']
                if not non_qb_lineup.empty:
                    lowest_proj_idx = non_qb_lineup['projection'].idxmin()
                    lowest_player = non_qb_lineup.loc[lowest_proj_idx, 'name']
                    
                    best_catcher = pass_catchers.nlargest(1, 'projection').iloc[0]
                    lineup = [best_catcher['name'] if p == lowest_player else p for p in lineup]
        
        return lineup
    
    def _extract_diverse_lineups(
        self,
        population: List[List[str]],
        fitness_scores: List[float],
        num_lineups: int,
        max_ownership: float
    ) -> List[Dict]:
        """Extract top diverse lineups from population"""
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        diverse_lineups = []
        used_players = set()
        
        for idx in sorted_indices:
            if len(diverse_lineups) >= num_lineups:
                break
            
            lineup = population[idx]
            
            # Check ownership constraint
            if max_ownership:
                lineup_df = self.players_df[self.players_df['name'].isin(lineup)]
                total_ownership = lineup_df['ownership'].sum()
                
                if total_ownership > max_ownership:
                    continue
            
            # Check diversity (at least 3 different players from previous lineups)
            is_diverse = True
            for prev_lineup in diverse_lineups:
                overlap = len(set(lineup) & set(prev_lineup['players']))
                if overlap > 3:  # More than 3 shared players
                    is_diverse = False
                    break
            
            if is_diverse:
                # Calculate metrics
                lineup_df = self.players_df[self.players_df['name'].isin(lineup)]
                
                captain_name = lineup[0]
                captain_data = lineup_df[lineup_df['name'] == captain_name].iloc[0]
                
                captain_proj = captain_data['projection'] * 1.5
                flex_proj = lineup_df[lineup_df['name'] != captain_name]['projection'].sum()
                total_proj = captain_proj + flex_proj
                
                captain_ceil = captain_data['ceiling'] * 1.5
                flex_ceil = lineup_df[lineup_df['name'] != captain_name]['ceiling'].sum()
                total_ceil = captain_ceil + flex_ceil
                
                total_ownership = lineup_df['ownership'].sum()
                
                captain_salary = captain_data['salary'] * 1.5
                flex_salary = lineup_df[lineup_df['name'] != captain_name]['salary'].sum()
                total_salary = captain_salary + flex_salary
                
                correlation = self.stacking_engine.calculate_lineup_correlation(lineup)
                
                diverse_lineups.append({
                    'players': lineup,
                    'fitness': fitness_scores[idx],
                    'metrics': {
                        'total_projection': float(total_proj),
                        'total_ceiling': float(total_ceil),
                        'total_ownership': float(total_ownership),
                        'total_salary': int(total_salary),
                        'correlation': float(correlation)
                    },
                    'captain': captain_name
                })
        
        logger.info(f"Extracted {len(diverse_lineups)} diverse lineups")
        
        return diverse_lineups
    
    def get_evolution_stats(self) -> Dict:
        """Get statistics about the evolution process"""
        if not self.best_fitness_history:
            return {}
        
        return {
            'initial_fitness': self.best_fitness_history[0],
            'final_fitness': self.best_fitness_history[-1],
            'improvement': self.best_fitness_history[-1] - self.best_fitness_history[0],
            'improvement_pct': ((self.best_fitness_history[-1] / self.best_fitness_history[0]) - 1) * 100,
            'avg_final': self.avg_fitness_history[-1],
            'generations': len(self.best_fitness_history)
        }
