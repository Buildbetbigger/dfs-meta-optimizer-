"""
Module 3: Exposure Manager
Manages player exposure caps and tracking across lineup portfolios
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExposureRule:
    """Represents an exposure rule for a player or group"""
    player_name: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    min_exposure: float = 0.0  # Minimum exposure % (0-100)
    max_exposure: float = 100.0  # Maximum exposure % (0-100)
    rule_type: str = 'hard'  # 'hard' or 'soft'
    priority: int = 1  # Higher priority = enforced first


class ExposureManager:
    """
    Manages player exposure across a portfolio of lineups.
    
    Features:
    - Hard caps (must not exceed)
    - Soft caps (prefer not to exceed but can if necessary)
    - Player-specific rules
    - Position-based rules
    - Team-based rules
    - Global exposure limits
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        Initialize exposure manager.
        
        Args:
            players_df: DataFrame with player data
        """
        self.players_df = players_df.copy()
        self.exposure_rules: List[ExposureRule] = []
        self.global_max_exposure = 40.0  # Default: 40% max for any player
        self.global_min_exposure = 0.0
        
        logger.info("ExposureManager initialized")
    
    def add_rule(
        self,
        player_name: Optional[str] = None,
        position: Optional[str] = None,
        team: Optional[str] = None,
        min_exposure: float = 0.0,
        max_exposure: float = 100.0,
        rule_type: str = 'hard',
        priority: int = 1
    ):
        """
        Add an exposure rule.
        
        Args:
            player_name: Specific player name (None for position/team rules)
            position: Position group (None for player/team rules)
            team: Team (None for player/position rules)
            min_exposure: Minimum exposure % (0-100)
            max_exposure: Maximum exposure % (0-100)
            rule_type: 'hard' (must enforce) or 'soft' (prefer but flexible)
            priority: Higher number = higher priority
        """
        rule = ExposureRule(
            player_name=player_name,
            position=position,
            team=team,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            rule_type=rule_type,
            priority=priority
        )
        
        self.exposure_rules.append(rule)
        
        # Sort by priority (highest first)
        self.exposure_rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Added {rule_type} exposure rule: {player_name or position or team}, "
                   f"{min_exposure}-{max_exposure}%, priority {priority}")
    
    def remove_rule(self, index: int):
        """Remove an exposure rule by index"""
        if 0 <= index < len(self.exposure_rules):
            removed = self.exposure_rules.pop(index)
            logger.info(f"Removed rule: {removed}")
        else:
            logger.warning(f"Invalid rule index: {index}")
    
    def clear_rules(self):
        """Clear all exposure rules"""
        self.exposure_rules.clear()
        logger.info("All exposure rules cleared")
    
    def set_global_max_exposure(self, max_pct: float):
        """Set global maximum exposure for all players"""
        self.global_max_exposure = max_pct
        logger.info(f"Global max exposure set to {max_pct}%")
    
    def set_global_min_exposure(self, min_pct: float):
        """Set global minimum exposure for all players"""
        self.global_min_exposure = min_pct
        logger.info(f"Global min exposure set to {min_pct}%")
    
    def calculate_current_exposure(
        self,
        lineups: List[Dict],
        include_captain_weight: bool = True
    ) -> Dict[str, float]:
        """
        Calculate current exposure for each player.
        
        Args:
            lineups: List of lineup dictionaries
            include_captain_weight: Weight captains higher (1.5x)
        
        Returns:
            Dictionary of {player_name: exposure_percentage}
        """
        if not lineups:
            return {}
        
        player_counts = {}
        total_lineups = len(lineups)
        
        for lineup in lineups:
            players = lineup.get('players', [])
            captain = lineup.get('captain')
            
            for player in players:
                if player not in player_counts:
                    player_counts[player] = 0
                
                # Weight captain appearances more if specified
                if include_captain_weight and player == captain:
                    player_counts[player] += 1.5
                else:
                    player_counts[player] += 1
        
        # Calculate percentages
        exposure_pct = {
            player: (count / total_lineups) * 100
            for player, count in player_counts.items()
        }
        
        return exposure_pct
    
    def check_exposure_compliance(
        self,
        lineups: List[Dict],
        strict_mode: bool = False
    ) -> Dict:
        """
        Check if lineups comply with exposure rules.
        
        Args:
            lineups: List of lineup dictionaries
            strict_mode: If True, check soft rules as hard rules
        
        Returns:
            Dictionary with compliance status and violations
        """
        exposure = self.calculate_current_exposure(lineups)
        violations = []
        warnings = []
        
        # Check each rule
        for rule in self.exposure_rules:
            is_hard_rule = rule.rule_type == 'hard' or strict_mode
            
            # Player-specific rule
            if rule.player_name:
                player_exp = exposure.get(rule.player_name, 0.0)
                
                if player_exp < rule.min_exposure or player_exp > rule.max_exposure:
                    violation = {
                        'rule_type': rule.rule_type,
                        'player': rule.player_name,
                        'current_exposure': player_exp,
                        'min_allowed': rule.min_exposure,
                        'max_allowed': rule.max_exposure,
                        'severity': 'violation' if is_hard_rule else 'warning'
                    }
                    
                    if is_hard_rule:
                        violations.append(violation)
                    else:
                        warnings.append(violation)
            
            # Position-based rule
            elif rule.position:
                position_players = self.players_df[
                    self.players_df['position'] == rule.position
                ]['name'].tolist()
                
                for player in position_players:
                    player_exp = exposure.get(player, 0.0)
                    
                    if player_exp < rule.min_exposure or player_exp > rule.max_exposure:
                        violation = {
                            'rule_type': rule.rule_type,
                            'player': player,
                            'position': rule.position,
                            'current_exposure': player_exp,
                            'min_allowed': rule.min_exposure,
                            'max_allowed': rule.max_exposure,
                            'severity': 'violation' if is_hard_rule else 'warning'
                        }
                        
                        if is_hard_rule:
                            violations.append(violation)
                        else:
                            warnings.append(violation)
            
            # Team-based rule
            elif rule.team:
                team_players = self.players_df[
                    self.players_df['team'] == rule.team
                ]['name'].tolist()
                
                for player in team_players:
                    player_exp = exposure.get(player, 0.0)
                    
                    if player_exp < rule.min_exposure or player_exp > rule.max_exposure:
                        violation = {
                            'rule_type': rule.rule_type,
                            'player': player,
                            'team': rule.team,
                            'current_exposure': player_exp,
                            'min_allowed': rule.min_exposure,
                            'max_allowed': rule.max_exposure,
                            'severity': 'violation' if is_hard_rule else 'warning'
                        }
                        
                        if is_hard_rule:
                            violations.append(violation)
                        else:
                            warnings.append(violation)
        
        # Check global limits
        for player, exp in exposure.items():
            if exp > self.global_max_exposure:
                violations.append({
                    'rule_type': 'global',
                    'player': player,
                    'current_exposure': exp,
                    'max_allowed': self.global_max_exposure,
                    'severity': 'violation'
                })
        
        is_compliant = len(violations) == 0
        
        return {
            'compliant': is_compliant,
            'violations': violations,
            'warnings': warnings,
            'total_violations': len(violations),
            'total_warnings': len(warnings)
        }
    
    def enforce_exposure_on_lineup(
        self,
        candidate_lineup: Dict,
        existing_lineups: List[Dict],
        total_target_lineups: int
    ) -> bool:
        """
        Check if adding a candidate lineup would violate hard exposure rules.
        
        Args:
            candidate_lineup: Lineup to potentially add
            existing_lineups: Current lineup portfolio
            total_target_lineups: Total number of lineups planned
        
        Returns:
            True if lineup can be added without violating hard rules
        """
        # Simulate adding the lineup
        test_lineups = existing_lineups + [candidate_lineup]
        
        # Calculate projected exposure
        exposure = self.calculate_current_exposure(test_lineups)
        
        # Project to full portfolio
        current_count = len(test_lineups)
        projection_multiplier = total_target_lineups / current_count
        
        # Check hard rules only
        for rule in self.exposure_rules:
            if rule.rule_type != 'hard':
                continue
            
            # Player-specific rule
            if rule.player_name:
                player_exp = exposure.get(rule.player_name, 0.0) * projection_multiplier
                
                if player_exp > rule.max_exposure:
                    logger.debug(f"Lineup rejected: {rule.player_name} would exceed "
                               f"{rule.max_exposure}% (projected {player_exp:.1f}%)")
                    return False
            
            # Position-based rule
            elif rule.position:
                position_players = self.players_df[
                    self.players_df['position'] == rule.position
                ]['name'].tolist()
                
                for player in position_players:
                    player_exp = exposure.get(player, 0.0) * projection_multiplier
                    
                    if player_exp > rule.max_exposure:
                        return False
            
            # Team-based rule
            elif rule.team:
                team_players = self.players_df[
                    self.players_df['team'] == rule.team
                ]['name'].tolist()
                
                for player in team_players:
                    player_exp = exposure.get(player, 0.0) * projection_multiplier
                    
                    if player_exp > rule.max_exposure:
                        return False
        
        # Check global max
        for player, exp in exposure.items():
            projected_exp = exp * projection_multiplier
            if projected_exp > self.global_max_exposure:
                logger.debug(f"Lineup rejected: {player} would exceed global max "
                           f"{self.global_max_exposure}% (projected {projected_exp:.1f}%)")
                return False
        
        return True
    
    def get_exposure_report(
        self,
        lineups: List[Dict],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Generate detailed exposure report.
        
        Args:
            lineups: List of lineup dictionaries
            top_n: Number of top players to include
        
        Returns:
            DataFrame with exposure details
        """
        exposure = self.calculate_current_exposure(lineups)
        
        report_data = []
        
        for player, exp_pct in sorted(exposure.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            player_data = self.players_df[self.players_df['name'] == player]
            
            if player_data.empty:
                continue
            
            player_info = player_data.iloc[0]
            
            # Find applicable rules
            applicable_rules = []
            for rule in self.exposure_rules:
                if rule.player_name == player:
                    applicable_rules.append(f"Player: {rule.min_exposure}-{rule.max_exposure}%")
                elif rule.position == player_info['position']:
                    applicable_rules.append(f"Pos: {rule.min_exposure}-{rule.max_exposure}%")
                elif rule.team == player_info['team']:
                    applicable_rules.append(f"Team: {rule.min_exposure}-{rule.max_exposure}%")
            
            # Check compliance
            is_compliant = exp_pct <= self.global_max_exposure
            
            for rule in self.exposure_rules:
                if rule.rule_type == 'hard':
                    if rule.player_name == player:
                        is_compliant = is_compliant and (rule.min_exposure <= exp_pct <= rule.max_exposure)
            
            report_data.append({
                'Player': player,
                'Position': player_info['position'],
                'Team': player_info['team'],
                'Exposure %': exp_pct,
                'Count': int((exp_pct / 100) * len(lineups)),
                'Salary': int(player_info['salary']),
                'Projection': float(player_info['projection']),
                'Rules': ', '.join(applicable_rules) if applicable_rules else 'Global only',
                'Compliant': '✓' if is_compliant else '✗'
            })
        
        return pd.DataFrame(report_data)
    
    def get_underexposed_players(
        self,
        lineups: List[Dict],
        threshold: float = 5.0
    ) -> List[str]:
        """
        Get list of players below exposure threshold.
        
        Args:
            lineups: Current lineup portfolio
            threshold: Minimum exposure % to not be considered underexposed
        
        Returns:
            List of underexposed player names
        """
        exposure = self.calculate_current_exposure(lineups)
        
        # All players in pool
        all_players = set(self.players_df['name'].tolist())
        
        underexposed = []
        
        for player in all_players:
            current_exp = exposure.get(player, 0.0)
            
            if current_exp < threshold:
                underexposed.append(player)
        
        return underexposed
    
    def suggest_exposure_adjustments(
        self,
        lineups: List[Dict]
    ) -> List[Dict]:
        """
        Suggest adjustments to improve exposure compliance.
        
        Args:
            lineups: Current lineup portfolio
        
        Returns:
            List of adjustment suggestions
        """
        compliance = self.check_exposure_compliance(lineups)
        suggestions = []
        
        # Violations - must fix
        for violation in compliance['violations']:
            player = violation['player']
            current = violation['current_exposure']
            max_allowed = violation.get('max_allowed', self.global_max_exposure)
            
            if current > max_allowed:
                # Over-exposed
                reduce_by = current - max_allowed
                lineups_to_remove = int(np.ceil((reduce_by / 100) * len(lineups)))
                
                suggestions.append({
                    'type': 'reduce_exposure',
                    'player': player,
                    'current': current,
                    'target': max_allowed,
                    'action': f"Remove {player} from ~{lineups_to_remove} lineups",
                    'priority': 'high'
                })
        
        # Warnings - optional fixes
        for warning in compliance['warnings']:
            player = warning['player']
            current = warning['current_exposure']
            max_allowed = warning.get('max_allowed', self.global_max_exposure)
            
            if current > max_allowed:
                suggestions.append({
                    'type': 'reduce_exposure',
                    'player': player,
                    'current': current,
                    'target': max_allowed,
                    'action': f"Consider reducing {player} exposure",
                    'priority': 'medium'
                })
        
        return suggestions
    
    def balance_exposure(
        self,
        lineups: List[Dict],
        target_variance: float = 15.0
    ) -> List[str]:
        """
        Identify players with unbalanced exposure variance.
        
        Args:
            lineups: Current lineup portfolio
            target_variance: Target standard deviation for exposure
        
        Returns:
            List of players with high exposure variance
        """
        exposure = self.calculate_current_exposure(lineups)
        
        if not exposure:
            return []
        
        # Calculate exposure variance
        exposures = list(exposure.values())
        mean_exp = np.mean(exposures)
        std_exp = np.std(exposures)
        
        # Find outliers
        outliers = []
        
        for player, exp in exposure.items():
            if abs(exp - mean_exp) > target_variance:
                outliers.append(player)
        
        return outliers
