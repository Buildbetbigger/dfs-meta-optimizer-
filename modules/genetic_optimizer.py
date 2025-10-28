"""
Module 2: Genetic Optimizer
Enhanced genetic algorithm for DFS lineup optimization

Features:
- Multi-objective fitness function (5 components)
- Tournament selection with elite preservation
- Adaptive mutation rates
- Two-point crossover  
- Automatic constraint repair
- Mode-specific weight optimization
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """
    Enhanced Genetic Algorithm for DFS lineup optimization
    
    Optimizes lineups across multiple objectives:
    1. Projected Points - raw scoring potential
    2. Ceiling - upside/tournament winning potential
    3. Leverage - advantage vs field ownership
    4. Correlation - stacking quality
    5. Ownership - contrarian value
    """
    
    def __init__(self,
                 players_df: pd.DataFrame,
                 opponent_model,
                 stacking_engine,
                 salary_cap: int = 50000,
                 roster_size: int = 9):
        """
        Initialize genetic optimizer
        
        Args:
            players_df: DataFrame with player data
            opponent_model: OpponentModeler instance for leverage scoring
            stacking_engine: StackingEngine instance for correlation
            salary_cap: Maximum salary allowed
            roster_size: Number of players in lineup
        """
        self.players_df = players_df
        self.opponent_model = opponent_model
        self.stacking_engine = stacking_engine
        self.salary_cap = salary_cap
        self.roster_size = roster_size
        
        # GA Parameters
        self.population_size = 200
        self.generations = 100
        self.elite_size = 20  # Top 10% preserved
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.tournament_size = 5
        
        # Fitness component weights (can be adjusted per mode)
        self.weights = {
            'projection': 0.30,    # Base points
            'ceiling': 0.25,       # Upside potential
            'leverage': 0.25,      # Field advantage
            'correlation': 0.10,   # Stacking quality
            'ownership': 0.10      # Contrarian bonus
        }
        
        # Track evolution statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        
        logger.info(f"GeneticOptimizer initialized: pop={self.population_size}, "
                   f"gen={self.generations}, roster={self.roster_size}")
    
    def set_mode_weights(self, mode: str):
        """
        Adjust fitness weights based on contest type
        
        Args:
            mode: One of 'CASH', 'GPP', 'CONTRARIAN', 'BALANCED'
        """
        mode_weights = {
            'CASH': {
                'projection': 0.50,  # Prioritize floor
                'ceiling': 0.15,
                'leverage': 0.10,
                'correlation': 0.20,
                'ownership': 0.05
            },
            'GPP': {
                'projection': 0.25,
                'ceiling': 0.30,     # Prioritize ceiling
                'leverage': 0.25,
                'correlation': 0.10,
                'ownership': 0.10
            },
            'CONTRARIAN': {
                'projection': 0.20,
                'ceiling': 0.25,
                'leverage': 0.20,
                'correlation': 0.10,
                'ownership': 0.25    # Heavy contrarian
            },
            'BALANCED': {
                'projection': 0.30,
                'ceiling': 0.25,
                'leverage': 0.25,
                'correlation': 0.10,
                'ownership': 0.10
            }
        }
        
        if mode.upper() in mode_weights:
            self.weights = mode_weights[mode.upper()]
            logger.info(f"Weights set to {mode.upper()} mode")
        else:
            logger.warning(f"Unknown mode '{mode}', using defaults")
    
    def evolve(self,
               target_mode: str = 'GPP',
               enforce_stacks: bool = True,
               num_lineups: int = 20,
               max_ownership: float = 50.0) -> List[Dict]:
        """
        Run genetic algorithm evolution to generate optimal lineups
        
        Args:
            target_mode: Optimization mode (CASH, GPP, CONTRARIAN, BALANCED)
            enforce_stacks: Require QB stacks in lineups
            num_lineups: Number of unique lineups to return
            max_ownership: Maximum average lineup ownership
            
        Returns:
            List of optimized lineups with metrics
        """
        logger.info(f"Starting evolution: mode={target_mode}, gen={self.generations}, "
                   f"pop={self.population_size}")
        
        # Set mode-specific weights
        self.set_mode_weights(target_mode)
        
        # Initialize population
        population = self._initialize_population(enforce_stacks)
        
        # Evolution loop
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self._calculate_fitness(lineup) for lineup in population]
            
            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            diversity = self._calculate_diversity(population)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Gen {generation}: Best={best_fitness:.2f}, "
                           f"Avg={avg_fitness:.2f}, Diversity={diversity:.2f}")
            
            # Selection - Tournament selection
            parents = self._tournament_selection(population, fitness_scores)
            
            # Create next generation
            offspring = []
            
            # Elitism - preserve best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elites = [population[i] for i in elite_indices]
            offspring.extend(elites)
            
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
                
                # Repair any constraint violations
                child = self._repair_lineup(child, enforce_stacks)
                
                offspring.append(child)
            
            population = offspring
        
        # Final evaluation
        final_fitness = [self._calculate_fitness(lineup) for lineup in population]
        
        # Extract top diverse lineups
        top_lineups = self._extract_diverse_lineups(
            population, 
            final_fitness,
            num_lineups,
            max_ownership
        )
        
        logger.info(f"Evolution complete. Generated {len(top_lineups)} lineups")
        
        return top_lineups
    
    def _initialize_population(self, enforce_stacks: bool) -> List[List[str]]:
        """
        Initialize population with random valid lineups
        
        Args:
            enforce_stacks: Whether to require QB stacks
            
        Returns:
            List of lineup (each lineup is list of player names)
        """
        population = []
        max_attempts = self.population_size * 3
        attempts = 0
        
        while len(population) < self.population_size and attempts < max_attempts:
            lineup = self._generate_random_lineup(enforce_stacks)
            
            if lineup and self._is_valid_lineup(lineup):
                population.append(lineup)
            
            attempts += 1
        
        # Fill remaining with basic lineups if needed
        while len(population) < self.population_size:
            # Simple greedy approach for filler
            lineup = self._generate_greedy_lineup()
            if lineup:
                population.append(lineup)
        
        logger.info(f"Initialized population of {len(population)} lineups")
        return population
    
    def _generate_random_lineup(self, enforce_stacks: bool) -> List[str]:
        """Generate a random valid lineup"""
        lineup = []
        available = self.players_df.copy()
        
        # If enforcing stacks, start with a QB stack
        if enforce_stacks:
            # Pick random QB
            qbs = available[available['position'] == 'QB']
            if len(qbs) > 0:
                qb = qbs.sample(1).iloc[0]
                lineup.append(qb['player_name'])
                available = available[available['player_name'] != qb['player_name']]
                
                # Add pass catcher from same team
                pass_catchers = available[
                    (available['team'] == qb['team']) &
                    (available['position'].isin(['WR', 'TE']))
                ]
                if len(pass_catchers) > 0:
                    receiver = pass_catchers.sample(1).iloc[0]
                    lineup.append(receiver['player_name'])
                    available = available[available['player_name'] != receiver['player_name']]
        
        # Fill rest randomly
        while len(lineup) < self.roster_size and len(available) > 0:
            # Weight selection by projection
            weights = available['projection'] / available['projection'].sum()
            player = available.sample(1, weights=weights).iloc[0]
            lineup.append(player['player_name'])
            available = available[available['player_name'] != player['player_name']]
        
        return lineup if len(lineup) == self.roster_size else None
    
    def _generate_greedy_lineup(self) -> List[str]:
        """Generate lineup using simple greedy value approach"""
        lineup = []
        available = self.players_df.copy()
        
        # Calculate value (points per $1000)
        available['value'] = available['projection'] / (available['salary'] / 1000)
        available = available.sort_values('value', ascending=False)
        
        for _, player in available.iterrows():
            if len(lineup) >= self.roster_size:
                break
            
            test_lineup = lineup + [player['player_name']]
            if self._get_lineup_salary(test_lineup) <= self.salary_cap:
                lineup.append(player['player_name'])
        
        return lineup if len(lineup) == self.roster_size else None
    
    def _calculate_fitness(self, lineup: List[str]) -> float:
        """
        Calculate multi-objective fitness score for a lineup
        
        Components:
        1. Projection - Expected points
        2. Ceiling - Upside potential
        3. Leverage - Field advantage
        4. Correlation - Stacking quality
        5. Ownership - Contrarian value
        
        Args:
            lineup: List of player names
            
        Returns:
            Fitness score (higher is better)
        """
        players = self.players_df[self.players_df['player_name'].isin(lineup)]
        
        if len(players) == 0:
            return 0.0
        
        # Component 1: Projection (normalized to 0-100)
        total_projection = players['projection'].sum()
        projection_score = min(100, (total_projection / 150) * 100)  # ~150 is top lineup
        
        # Component 2: Ceiling (normalized to 0-100)
        total_ceiling = players['ceiling'].sum() if 'ceiling' in players.columns else total_projection * 1.3
        ceiling_score = min(100, (total_ceiling / 200) * 100)  # ~200 is max ceiling
        
        # Component 3: Leverage (use opponent model)
        leverage_scores = []
        for player_name in lineup:
            player_data = players[players['player_name'] == player_name]
            if len(player_data) > 0:
                ownership = player_data['ownership'].iloc[0]
                projection = player_data['projection'].iloc[0]
                
                # Simple leverage: projection / ownership
                if ownership > 0:
                    leverage = (projection / ownership) * 10
                    leverage_scores.append(leverage)
        
        leverage_score = np.mean(leverage_scores) if leverage_scores else 50
        leverage_score = min(100, leverage_score)
        
        # Component 4: Correlation (use stacking engine)
        if self.stacking_engine:
            correlation_score = self.stacking_engine.score_lineup_correlation(lineup)
        else:
            correlation_score = 50.0
        
        # Component 5: Ownership (contrarian bonus)
        avg_ownership = players['ownership'].mean() if 'ownership' in players.columns else 15.0
        # Lower ownership = higher score (inverse relationship)
        ownership_score = 100 - min(100, avg_ownership * 2)  # 0% own = 100, 50% own = 0
        
        # Weighted combination
        fitness = (
            self.weights['projection'] * projection_score +
            self.weights['ceiling'] * ceiling_score +
            self.weights['leverage'] * leverage_score +
            self.weights['correlation'] * correlation_score +
            self.weights['ownership'] * ownership_score
        )
        
        return fitness
    
    def _tournament_selection(self, population: List[List[str]], 
                             fitness_scores: List[float]) -> List[List[str]]:
        """
        Tournament selection for parent selection
        
        Args:
            population: Current population
            fitness_scores: Fitness score for each individual
            
        Returns:
            Selected parents
        """
        parents = []
        
        for _ in range(len(population)):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Winner has highest fitness
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Two-point crossover between two parents
        
        Args:
            parent1: First parent lineup
            parent2: Second parent lineup
            
        Returns:
            Child lineup
        """
        if len(parent1) != len(parent2):
            return parent1.copy()
        
        # Select two random crossover points
        point1 = random.randint(0, len(parent1) - 1)
        point2 = random.randint(point1, len(parent1))
        
        # Create child with segment from parent1
        child = parent1[:point1] + parent1[point1:point2] + parent2[point2:]
        
        # Remove duplicates, keeping first occurrence
        seen = set()
        unique_child = []
        for player in child:
            if player not in seen:
                unique_child.append(player)
                seen.add(player)
        
        return unique_child
    
    def _mutate(self, lineup: List[str]) -> List[str]:
        """
        Mutate lineup by replacing random player
        
        Args:
            lineup: Lineup to mutate
            
        Returns:
            Mutated lineup
        """
        mutated = lineup.copy()
        
        if len(mutated) == 0:
            return mutated
        
        # Select random position to mutate
        mutation_idx = random.randint(0, len(mutated) - 1)
        
        # Find replacement candidates
        current_players = set(mutated)
        available = self.players_df[
            ~self.players_df['player_name'].isin(current_players)
        ]
        
        if len(available) > 0:
            # Weight by projection
            weights = available['projection'] / available['projection'].sum()
            replacement = available.sample(1, weights=weights).iloc[0]
            mutated[mutation_idx] = replacement['player_name']
        
        return mutated
    
    def _adaptive_mutation_rate(self, generation: int) -> float:
        """
        Calculate adaptive mutation rate that decreases over generations
        
        Args:
            generation: Current generation number
            
        Returns:
            Mutation rate for this generation
        """
        # Start high, decrease to 50% of initial rate by end
        progress = generation / self.generations
        adaptive_rate = self.mutation_rate * (1 - 0.5 * progress)
        return adaptive_rate
    
    def _repair_lineup(self, lineup: List[str], enforce_stacks: bool) -> List[str]:
        """
        Repair lineup to satisfy all constraints
        
        Args:
            lineup: Potentially invalid lineup
            enforce_stacks: Whether to enforce stacking
            
        Returns:
            Valid lineup
        """
        # Remove duplicates
        lineup = list(dict.fromkeys(lineup))
        
        # Fill to correct size
        max_attempts = 50
        attempts = 0
        
        while len(lineup) < self.roster_size and attempts < max_attempts:
            available = self.players_df[
                ~self.players_df['player_name'].isin(lineup)
            ]
            
            if len(available) == 0:
                break
            
            # Add player that fits under salary cap
            current_salary = self._get_lineup_salary(lineup)
            remaining = self.salary_cap - current_salary
            
            affordable = available[available['salary'] <= remaining]
            
            if len(affordable) > 0:
                weights = affordable['projection'] / affordable['projection'].sum()
                new_player = affordable.sample(1, weights=weights).iloc[0]
                lineup.append(new_player['player_name'])
            else:
                break
            
            attempts += 1
        
        # Adjust if over salary cap
        attempts = 0
        while self._get_lineup_salary(lineup) > self.salary_cap and attempts < max_attempts:
            # Remove most expensive player and replace with cheaper
            players = self.players_df[self.players_df['player_name'].isin(lineup)]
            
            if len(players) == 0:
                break
            
            most_expensive = players.nlargest(1, 'salary').iloc[0]
            lineup.remove(most_expensive['player_name'])
            
            # Add cheaper replacement
            available = self.players_df[
                (~self.players_df['player_name'].isin(lineup)) &
                (self.players_df['salary'] < most_expensive['salary'])
            ]
            
            if len(available) > 0:
                replacement = available.sample(1).iloc[0]
                lineup.append(replacement['player_name'])
            
            attempts += 1
        
        # Trim to correct size
        lineup = lineup[:self.roster_size]
        
        return lineup
    
    def _is_valid_lineup(self, lineup: List[str]) -> bool:
        """Check if lineup satisfies all constraints"""
        if len(lineup) != self.roster_size:
            return False
        
        # Check no duplicates
        if len(set(lineup)) != len(lineup):
            return False
        
        # Check salary cap
        salary = self._get_lineup_salary(lineup)
        if salary > self.salary_cap:
            return False
        
        return True
    
    def _get_lineup_salary(self, lineup: List[str]) -> int:
        """Calculate total salary for a lineup"""
        players = self.players_df[self.players_df['player_name'].isin(lineup)]
        return players['salary'].sum()
    
    def _calculate_diversity(self, population: List[List[str]]) -> float:
        """
        Calculate population diversity (0-100 scale)
        Higher diversity = more different lineups
        
        Args:
            population: List of lineups
            
        Returns:
            Diversity score
        """
        if len(population) < 2:
            return 0.0
        
        # Count unique players across population
        all_players = set()
        for lineup in population:
            all_players.update(lineup)
        
        # Diversity = unique players / total player slots
        total_slots = len(population) * self.roster_size
        diversity = (len(all_players) / min(total_slots, len(self.players_df))) * 100
        
        return min(100, diversity)
    
    def _extract_diverse_lineups(self,
                                 population: List[List[str]],
                                 fitness_scores: List[float],
                                 num_lineups: int,
                                 max_ownership: float) -> List[Dict]:
        """
        Extract top diverse lineups from final population
        
        Args:
            population: Final population
            fitness_scores: Fitness scores
            num_lineups: Number of lineups to extract
            max_ownership: Maximum average ownership
            
        Returns:
            List of lineup dictionaries with full metrics
        """
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        selected_lineups = []
        selected_players = set()
        
        for idx in sorted_indices:
            if len(selected_lineups) >= num_lineups:
                break
            
            lineup = population[idx]
            
            # Check ownership constraint
            players = self.players_df[self.players_df['player_name'].isin(lineup)]
            avg_own = players['ownership'].mean() if 'ownership' in players.columns else 15.0
            
            if avg_own > max_ownership:
                continue
            
            # Check diversity (don't add if too similar to existing)
            overlap = len(set(lineup) & selected_players)
            if overlap > self.roster_size * 0.6 and len(selected_lineups) > 0:
                continue  # Skip if >60% overlap with already selected
            
            # Add lineup
            lineup_dict = self._create_lineup_dict(lineup, fitness_scores[idx])
            selected_lineups.append(lineup_dict)
            selected_players.update(lineup)
        
        return selected_lineups
    
    def _create_lineup_dict(self, lineup: List[str], fitness: float) -> Dict:
        """Create detailed lineup dictionary"""
        players = self.players_df[self.players_df['player_name'].isin(lineup)]
        
        return {
            'players': players.to_dict('records'),
            'total_salary': players['salary'].sum(),
            'total_projection': players['projection'].sum(),
            'total_ceiling': players['ceiling'].sum() if 'ceiling' in players.columns else players['projection'].sum() * 1.3,
            'avg_ownership': players['ownership'].mean() if 'ownership' in players.columns else 15.0,
            'fitness_score': fitness,
            'correlation_score': self.stacking_engine.score_lineup_correlation(lineup) if self.stacking_engine else 0.0
        }
    
    def get_evolution_stats(self) -> Dict:
        """Get statistics from evolution run"""
        if not self.best_fitness_history:
            return {}
        
        return {
            'initial_fitness': self.best_fitness_history[0],
            'final_fitness': self.best_fitness_history[-1],
            'improvement': self.best_fitness_history[-1] - self.best_fitness_history[0],
            'improvement_pct': ((self.best_fitness_history[-1] - self.best_fitness_history[0]) / 
                               self.best_fitness_history[0] * 100) if self.best_fitness_history[0] > 0 else 0,
            'generations': self.generations,
            'population_size': self.population_size,
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history
        }
