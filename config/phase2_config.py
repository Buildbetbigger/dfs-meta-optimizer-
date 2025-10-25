"""
Module 2: Configuration File
Contains all configuration settings for Phase 2 features
"""

# ============================================================================
# GENETIC ALGORITHM SETTINGS
# ============================================================================

GA_SETTINGS = {
    'default': {
        'population_size': 200,
        'generations': 100,
        'mutation_rate': 0.15,
        'crossover_rate': 0.7,
        'elite_size': 20,
        'tournament_size': 5
    },
    'fast': {
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.20,
        'crossover_rate': 0.7,
        'elite_size': 10,
        'tournament_size': 3
    },
    'thorough': {
        'population_size': 300,
        'generations': 150,
        'mutation_rate': 0.12,
        'crossover_rate': 0.7,
        'elite_size': 30,
        'tournament_size': 7
    }
}


# ============================================================================
# FITNESS WEIGHTS BY MODE
# ============================================================================

FITNESS_WEIGHTS = {
    'CASH': {
        'projection': 0.50,
        'ceiling': 0.15,
        'leverage': 0.10,
        'correlation': 0.20,
        'ownership': 0.05
    },
    'GPP': {
        'projection': 0.25,
        'ceiling': 0.30,
        'leverage': 0.25,
        'correlation': 0.10,
        'ownership': 0.10
    },
    'CONTRARIAN': {
        'projection': 0.20,
        'ceiling': 0.25,
        'leverage': 0.20,
        'correlation': 0.10,
        'ownership': 0.25
    },
    'BALANCED': {
        'projection': 0.30,
        'ceiling': 0.25,
        'leverage': 0.25,
        'correlation': 0.10,
        'ownership': 0.10
    }
}


# ============================================================================
# STACKING SETTINGS
# ============================================================================

STACKING_SETTINGS = {
    'correlation_thresholds': {
        'qb_wr': 0.85,      # QB + WR same team
        'qb_te': 0.85,      # QB + TE same team
        'rb_pass': 0.30,    # RB + pass-catcher same team
        'same_team': 0.20,  # Other same-team combinations
        'game_stack': 0.40  # Opposing teams
    },
    'min_stack_size': 2,
    'max_stack_size': 4,
    'min_correlation_score': 0.5,
    'target_correlation': {
        'cash': 50.0,
        'gpp': 65.0,
        'tournament': 70.0
    }
}


# ============================================================================
# LEVERAGE SETTINGS
# ============================================================================

LEVERAGE_SETTINGS = {
    'min_leverage_threshold': 2.0,
    'high_leverage_threshold': 2.5,
    'elite_leverage_threshold': 3.0,
    'leverage_weight_cash': 0.10,
    'leverage_weight_gpp': 0.25,
    'leverage_weight_contrarian': 0.30
}


# ============================================================================
# CONTEST PRESETS
# ============================================================================

CONTEST_PRESETS = {
    'GPP': {
        'mode': 'GENETIC_GPP',
        'enforce_stacks': True,
        'max_ownership': None,
        'ga_profile': 'default',
        'description': 'Large-field GPP optimization'
    },
    'MILLY_MAKER': {
        'mode': 'LEVERAGE_FIRST',
        'enforce_stacks': True,
        'max_ownership': None,
        'ga_profile': 'thorough',
        'min_leverage': 2.5,
        'description': 'Milly Maker specific optimization'
    },
    'CASH': {
        'mode': 'GENETIC_CASH',
        'enforce_stacks': True,
        'max_ownership': 200.0,
        'ga_profile': 'default',
        'description': 'Cash game optimization'
    },
    'DOUBLE_UP': {
        'mode': 'GENETIC_CASH',
        'enforce_stacks': True,
        'max_ownership': 180.0,
        'ga_profile': 'fast',
        'description': 'Double-up optimization'
    },
    'SINGLE_ENTRY': {
        'mode': 'GENETIC_CONTRARIAN',
        'enforce_stacks': True,
        'max_ownership': 150.0,
        'ga_profile': 'thorough',
        'description': 'Single-entry GPP optimization'
    },
    'SATELLITE': {
        'mode': 'GENETIC_GPP',
        'enforce_stacks': True,
        'max_ownership': 180.0,
        'ga_profile': 'default',
        'description': 'Satellite/qualifier optimization'
    },
    'H2H': {
        'mode': 'GENETIC_CASH',
        'enforce_stacks': True,
        'max_ownership': 220.0,
        'ga_profile': 'fast',
        'description': 'Head-to-head optimization'
    },
    'THREE_MAX': {
        'mode': 'LEVERAGE_FIRST',
        'enforce_stacks': True,
        'max_ownership': None,
        'ga_profile': 'default',
        'min_leverage': 2.2,
        'description': '3-max contest optimization'
    }
}


# ============================================================================
# PORTFOLIO OPTIMIZATION SETTINGS
# ============================================================================

PORTFOLIO_SETTINGS = {
    'max_player_exposure_gpp': 0.40,     # 40% max exposure in GPPs
    'max_player_exposure_cash': 0.70,    # 70% max exposure in cash
    'min_unique_players': 3,              # Min unique players between lineups
    'diversity_threshold': 0.60,          # Target lineup diversity
    'captain_diversity_min': 0.30         # Min captain diversity
}


# ============================================================================
# OPTIMIZATION CONSTRAINTS
# ============================================================================

CONSTRAINTS = {
    'showdown': {
        'lineup_size': 6,
        'captain_multiplier': 1.5,
        'salary_cap': 50000,
        'positions': ['CPT', 'FLEX', 'FLEX', 'FLEX', 'FLEX', 'FLEX']
    },
    'classic': {
        'lineup_size': 9,
        'salary_cap': 50000,
        'positions': {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'FLEX': 1,
            'DST': 1
        }
    }
}


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

PERFORMANCE_SETTINGS = {
    'max_lineup_generation_time': 300,    # 5 minutes max
    'enable_parallel_processing': True,
    'max_workers': 4,
    'cache_correlation_matrix': True,
    'enable_progress_tracking': True
}


# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

VALIDATION_SETTINGS = {
    'min_projection': 0.0,
    'max_projection': 100.0,
    'min_ownership': 0.0,
    'max_ownership': 100.0,
    'min_salary': 3000,
    'max_salary': 12000,
    'require_valid_positions': True,
    'require_valid_teams': True,
    'allow_duplicate_players': False
}


# ============================================================================
# UI DISPLAY SETTINGS
# ============================================================================

UI_SETTINGS = {
    'show_evolution_progress': True,
    'show_stacking_analysis': True,
    'show_portfolio_metrics': True,
    'default_num_lineups': 20,
    'max_lineups_display': 50,
    'precision': {
        'projection': 1,
        'ownership': 1,
        'correlation': 1,
        'leverage': 2
    }
}


# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': False,
    'log_file': 'phase2_optimizer.log'
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_ga_settings(profile: str = 'default') -> dict:
    """Get genetic algorithm settings for a profile"""
    return GA_SETTINGS.get(profile, GA_SETTINGS['default'])


def get_contest_preset(contest_type: str) -> dict:
    """Get preset configuration for a contest type"""
    return CONTEST_PRESETS.get(contest_type.upper(), CONTEST_PRESETS['GPP'])


def get_fitness_weights(mode: str) -> dict:
    """Get fitness weights for an optimization mode"""
    return FITNESS_WEIGHTS.get(mode.upper(), FITNESS_WEIGHTS['BALANCED'])


def get_target_correlation(contest_type: str) -> float:
    """Get target correlation for a contest type"""
    contest_map = {
        'CASH': 'cash',
        'DOUBLE_UP': 'cash',
        'H2H': 'cash',
        'GPP': 'gpp',
        'SINGLE_ENTRY': 'tournament',
        'MILLY_MAKER': 'tournament'
    }
    
    key = contest_map.get(contest_type.upper(), 'gpp')
    return STACKING_SETTINGS['target_correlation'][key]


def validate_lineup_config(config: dict) -> bool:
    """Validate lineup configuration"""
    required_keys = ['mode', 'enforce_stacks', 'num_lineups']
    
    for key in required_keys:
        if key not in config:
            return False
    
    return True


# ============================================================================
# EXPORT ALL SETTINGS
# ============================================================================

__all__ = [
    'GA_SETTINGS',
    'FITNESS_WEIGHTS',
    'STACKING_SETTINGS',
    'LEVERAGE_SETTINGS',
    'CONTEST_PRESETS',
    'PORTFOLIO_SETTINGS',
    'CONSTRAINTS',
    'PERFORMANCE_SETTINGS',
    'VALIDATION_SETTINGS',
    'UI_SETTINGS',
    'LOGGING_SETTINGS',
    'get_ga_settings',
    'get_contest_preset',
    'get_fitness_weights',
    'get_target_correlation',
    'validate_lineup_config'
]
