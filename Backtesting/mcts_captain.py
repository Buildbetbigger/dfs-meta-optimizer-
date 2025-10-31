"""
Monte Carlo Tree Search (MCTS) - Phase 2 Bonus
Intelligent captain selection for DFS showdown slates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCTSNode:
    """Node in MCTS tree"""
    state: Dict  # Current lineup state
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    value: float = 0.0
    untried_actions: List[str] = None  # Player IDs not yet tried
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = []
    
    def is_fully_expanded(self) -> bool:
        """Check if all children have been explored"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (lineup complete)"""
        return self.state.get('complete', False)
    
    def best_child(self, c_param: float = 1.41) -> 'MCTSNode':
        """
        Select best child using UCB1 formula
        
        UCB1 = (value / visits) + c * sqrt(ln(parent_visits) / visits)
        
        Args:
            c_param: Exploration parameter (default: sqrt(2))
        
        Returns:
            Best child node
        """
        choices_weights = [
            (child.value / child.visits) + 
            c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class MCTSCaptainSelector:
    """
    Monte Carlo Tree Search for optimal captain selection
    
    In showdown DFS:
    - Captain scores 1.5x points but costs more
    - Must balance captain upside vs. lineup flexibility
    - MCTS explores captain + lineup combinations
    """
    
    def __init__(
        self,
        n_iterations: int = 1000,
        exploration_param: float = 1.41,
        max_depth: int = 6
    ):
        """
        Initialize MCTS
        
        Args:
            n_iterations: Number of MCTS iterations
            exploration_param: UCB1 exploration parameter
            max_depth: Max tree depth (lineup size)
        """
        self.n_iterations = n_iterations
        self.c_param = exploration_param
        self.max_depth = max_depth
        
    def evaluate_lineup(self, lineup: Dict) -> float:
        """
        Evaluate lineup quality
        
        Args:
            lineup: Lineup dict with players and captain
        
        Returns:
            Score (higher = better)
        """
        players = lineup.get('players', [])
        captain_id = lineup.get('captain', None)
        
        if not players or not captain_id:
            return 0.0
        
        # Calculate total projected points
        total_proj = 0.0
        for player in players:
            proj = player.get('projected_points', 0)
            if player['player_id'] == captain_id:
                proj *= 1.5  # Captain multiplier
            total_proj += proj
        
        # Salary efficiency
        total_salary = sum(p.get('salary', 0) for p in players)
        salary_left = 50000 - total_salary
        salary_efficiency = 1.0 - (abs(salary_left) / 50000)
        
        # Ownership leverage (lower = better for GPP)
        avg_ownership = np.mean([p.get('ownership', 50) for p in players])
        leverage_bonus = (100 - avg_ownership) / 100
        
        # Correlation bonus (stacking)
        correlation_bonus = 0.0
        # Check if captain and other players from same team
        captain = next((p for p in players if p['player_id'] == captain_id), None)
        if captain:
            same_team = [
                p for p in players 
                if p.get('team') == captain.get('team') and p['player_id'] != captain_id
            ]
            correlation_bonus = len(same_team) * 0.1
        
        # Composite score
        score = (
            total_proj * 1.0 +
            salary_efficiency * 20 +
            leverage_bonus * 15 +
            correlation_bonus * 10
        )
        
        return score
    
    def get_legal_actions(self, state: Dict, available_players: List[Dict]) -> List[str]:
        """
        Get legal player additions
        
        Args:
            state: Current lineup state
            available_players: Pool of available players
        
        Returns:
            List of player IDs that can be added
        """
        current_players = state.get('players', [])
        current_salary = sum(p.get('salary', 0) for p in current_players)
        
        legal = []
        for player in available_players:
            pid = player['player_id']
            
            # Check if already in lineup
            if any(p['player_id'] == pid for p in current_players):
                continue
            
            # Check salary
            if current_salary + player.get('salary', 0) <= 50000:
                legal.append(pid)
        
        return legal
    
    def selection(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree to find expandable node
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child(self.c_param)
        return node
    
    def expansion(self, node: MCTSNode, available_players: List[Dict]) -> MCTSNode:
        """
        Expansion phase: add new child node
        """
        if node.is_terminal():
            return node
        
        # Pick an untried action
        player_id = node.untried_actions.pop()
        
        # Create new state
        new_players = node.state['players'].copy()
        player = next(p for p in available_players if p['player_id'] == player_id)
        new_players.append(player)
        
        new_state = {
            'players': new_players,
            'captain': node.state.get('captain'),
            'complete': len(new_players) >= self.max_depth
        }
        
        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=node,
            untried_actions=self.get_legal_actions(new_state, available_players)
        )
        
        node.children.append(child)
        return child
    
    def simulation(self, node: MCTSNode, available_players: List[Dict]) -> float:
        """
        Simulation phase: random playout to estimate value
        """
        state = node.state.copy()
        players = state['players'].copy()
        
        # Complete lineup randomly
        while len(players) < self.max_depth:
            legal_actions = self.get_legal_actions(
                {'players': players, 'captain': state.get('captain')},
                available_players
            )
            
            if not legal_actions:
                break
            
            # Random action
            player_id = np.random.choice(legal_actions)
            player = next(p for p in available_players if p['player_id'] == player_id)
            players.append(player)
        
        # Evaluate final lineup
        final_state = {
            'players': players,
            'captain': state.get('captain'),
            'complete': True
        }
        
        return self.evaluate_lineup(final_state)
    
    def backpropagation(self, node: MCTSNode, value: float):
        """
        Backpropagation phase: update node values
        """
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def search(
        self,
        available_players: List[Dict],
        captain_id: str
    ) -> Dict:
        """
        Run MCTS to find optimal lineup with given captain
        
        Args:
            available_players: Pool of players
            captain_id: Selected captain player ID
        
        Returns:
            Best lineup found
        """
        # Initialize root
        captain = next(p for p in available_players if p['player_id'] == captain_id)
        initial_state = {
            'players': [captain],
            'captain': captain_id,
            'complete': False
        }
        
        root = MCTSNode(
            state=initial_state,
            untried_actions=self.get_legal_actions(initial_state, available_players)
        )
        
        # MCTS iterations
        for i in range(self.n_iterations):
            # 1. Selection
            node = self.selection(root)
            
            # 2. Expansion
            if not node.is_terminal() and node.untried_actions:
                node = self.expansion(node, available_players)
            
            # 3. Simulation
            value = self.simulation(node, available_players)
            
            # 4. Backpropagation
            self.backpropagation(node, value)
        
        # Return best child of root
        best_child = root.best_child(c_param=0)  # Exploitation only
        
        return best_child.state
    
    def select_optimal_captain(
        self,
        available_players: List[Dict],
        top_n_candidates: int = 5
    ) -> Tuple[str, Dict]:
        """
        Test multiple captain candidates and select best
        
        Args:
            available_players: Player pool
            top_n_candidates: Number of captain candidates to test
        
        Returns:
            (best_captain_id, best_lineup)
        """
        # Sort players by projected points for captain candidates
        candidates = sorted(
            available_players,
            key=lambda p: p.get('projected_points', 0),
            reverse=True
        )[:top_n_candidates]
        
        best_score = -float('inf')
        best_captain = None
        best_lineup = None
        
        logger.info(f"Testing {len(candidates)} captain candidates with MCTS")
        
        for candidate in candidates:
            captain_id = candidate['player_id']
            
            # Run MCTS for this captain
            lineup = self.search(available_players, captain_id)
            
            # Evaluate
            score = self.evaluate_lineup(lineup)
            
            logger.info(f"Captain {captain_id}: score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_captain = captain_id
                best_lineup = lineup
        
        return best_captain, best_lineup

class ShowdownOptimizer:
    """
    Complete showdown slate optimizer using MCTS
    """
    
    def __init__(self, mcts_iterations: int = 500):
        """
        Initialize optimizer
        
        Args:
            mcts_iterations: MCTS iterations per captain
        """
        self.mcts = MCTSCaptainSelector(n_iterations=mcts_iterations)
        
    def optimize_showdown_lineup(
        self,
        players: pd.DataFrame,
        num_lineups: int = 5
    ) -> List[Dict]:
        """
        Generate optimal showdown lineups
        
        Args:
            players: Player pool DataFrame
            num_lineups: Number of lineups to generate
        
        Returns:
            List of optimal lineups
        """
        available_players = players.to_dict('records')
        
        lineups = []
        
        for i in range(num_lineups):
            logger.info(f"Generating lineup {i+1}/{num_lineups}")
            
            # Select captain and build lineup
            captain_id, lineup = self.mcts.select_optimal_captain(
                available_players,
                top_n_candidates=5
            )
            
            lineups.append(lineup)
        
        return lineups

if __name__ == "__main__":
    # Test MCTS
    print("=== MCTS Captain Selection Test ===")
    
    # Mock player pool
    mock_players = [
        {'player_id': f'P{i}', 'position': 'FLEX', 'salary': 5000 + i*500, 
         'projected_points': 10 + i*2, 'ownership': 15 + i*3, 'team': f'TEAM{i%2}'}
        for i in range(12)
    ]
    
    mcts = MCTSCaptainSelector(n_iterations=100)
    
    # Test captain selection
    best_captain, best_lineup = mcts.select_optimal_captain(mock_players, top_n_candidates=3)
    
    print(f"\nBest Captain: {best_captain}")
    print(f"Lineup Size: {len(best_lineup['players'])}")
    print(f"Total Projected: {sum(p['projected_points'] * (1.5 if p['player_id']==best_captain else 1.0) for p in best_lineup['players']):.1f}")
    print(f"Salary Used: {sum(p['salary'] for p in best_lineup['players'])}")
