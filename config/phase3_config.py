"""
Module 3: Portfolio Optimization Configuration
Settings for exposure management, portfolio optimization, and filtering
"""

# ============================================================================
# EXPOSURE MANAGEMENT SETTINGS
# ============================================================================

EXPOSURE_SETTINGS = {
    'global_defaults': {
        'max_exposure_gpp': 40.0,        # 40% max for GPPs
        'max_exposure_cash': 70.0,       # 70% max for cash games
        'max_exposure_single': 100.0,    # 100% for single entry
        'min_exposure': 0.0,             # Minimum exposure
        'captain_weight': 1.5            # Weight captain appearances higher
    },
    'position_limits': {
        'QB': {'min': 0.0, 'max': 50.0},
        'RB': {'min': 0.0, 'max': 45.0},
        'WR': {'min': 0.0, 'max': 40.0},
        'TE': {'min': 0.0, 'max': 45.0}
    },
    'rule_priorities': {
        'player_specific': 10,    # Highest priority
        'position_based': 5,
        'team_based': 3,
        'global': 1              # Lowest priority
    }
}


# ============================================================================
# PORTFOLIO OPTIMIZATION SETTINGS
# ============================================================================

PORTFOLIO_SETTINGS = {
    'batch_generation': {
        'default_batch_size': 50,
        'min_batch_size': 10,
        'max_batch_size': 100,
        'candidate_pool_multiplier': 3   # Generate 3x candidates per batch
    },
    'diversity': {
        'min_unique_players_gpp': 3,     # Min different players between lineups (GPP)
        'min_unique_players_cash': 2,    # Min different players between lineups (Cash)
        'target_captain_diversity': 0.30, # Target 30% unique captains
        'min_captain_diversity': 0.20,   # Minimum 20% unique captains
        'max_captain_exposure': 15.0     # Max 15% per captain
    },
    'optimization': {
        'generations_per_batch': 100,
        'use_exposure_aware': True,
        'enforce_hard_caps': True,
        'allow_soft_cap_violations': False
    }
}


# ============================================================================
# CONTEST-SPECIFIC PORTFOLIO PRESETS
# ============================================================================

CONTEST_PORTFOLIO_PRESETS = {
    '150_ENTRY_GPP': {
        'num_lineups': 150,
        'max_player_exposure': 35.0,
        'min_unique_players': 3,
        'target_captain_diversity': 0.25,
        'batch_size': 50,
        'mode': 'GENETIC_GPP'
    },
    '20_ENTRY_GPP': {
        'num_lineups': 20,
        'max_player_exposure': 40.0,
        'min_unique_players': 3,
        'target_captain_diversity': 0.35,
        'batch_size': 20,
        'mode': 'GENETIC_GPP'
    },
    '3_ENTRY_MAX': {
        'num_lineups': 3,
        'max_player_exposure': 67.0,
        'min_unique_players': 4,
        'target_captain_diversity': 0.67,
        'batch_size': 3,
        'mode': 'GENETIC_CONTRARIAN'
    },
    'SINGLE_ENTRY': {
        'num_lineups': 1,
        'max_player_exposure': 100.0,
        'min_unique_players': 6,
        'target_captain_diversity': 1.0,
        'batch_size': 1,
        'mode': 'LEVERAGE_FIRST'
    },
    'CASH_MULTI': {
        'num_lineups': 20,
        'max_player_exposure': 60.0,
        'min_unique_players': 2,
        'target_captain_diversity': 0.40,
        'batch_size': 20,
        'mode': 'GENETIC_CASH'
    }
}


# ============================================================================
# FILTERING SETTINGS
# ============================================================================

FILTER_SETTINGS = {
    'deduplication': {
        'remove_exact_duplicates': True,
        'min_unique_players': 2,        # For similarity filtering
        'enable_similarity_check': True
    },
    'quality_thresholds': {
        'min_projection_gpp': 110.0,
        'min_projection_cash': 115.0,
        'min_ceiling_gpp': 140.0,
        'min_correlation': 50.0,
        'max_ownership_contrarian': 220.0
    },
    'stack_requirements': {
        'require_qb_stack_gpp': True,
        'require_qb_stack_cash': False,
        'min_stacks': 1
    },
    'captain_constraints': {
        'max_captain_exposure_default': 20.0,
        'enforce_captain_diversity': True
    }
}


# ============================================================================
# TIERED PORTFOLIO SETTINGS
# ============================================================================

TIER_DISTRIBUTIONS = {
    'conservative': {
        'safe': 0.50,        # 50% safe lineups
        'balanced': 0.40,    # 40% balanced
        'contrarian': 0.10   # 10% contrarian
    },
    'balanced': {
        'safe': 0.30,
        'balanced': 0.50,
        'contrarian': 0.20
    },
    'aggressive': {
        'safe': 0.20,
        'balanced': 0.30,
        'contrarian': 0.50
    },
    'contrarian_heavy': {
        'safe': 0.10,
        'balanced': 0.30,
        'contrarian': 0.60
    }
}

TIER_SETTINGS = {
    'safe': {
        'mode': 'GENETIC_CASH',
        'max_exposure': 50.0,
        'max_ownership': 200.0,
        'min_projection': 115.0
    },
    'balanced': {
        'mode': 'GENETIC_GPP',
        'max_exposure': 40.0,
        'max_ownership': None,
        'min_correlation': 55.0
    },
    'contrarian': {
        'mode': 'GENETIC_CONTRARIAN',
        'max_exposure': 30.0,
        'max_ownership': 180.0,
        'min_leverage': 2.0
    }
}


# ============================================================================
# PORTFOLIO METRICS SETTINGS
# ============================================================================

METRICS_SETTINGS = {
    'top_exposures_count': 20,
    'exposure_report_precision': 1,
    'calculate_similarity_matrix': False,  # Can be slow for large portfolios
    'track_evolution_stats': True
}


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

PERFORMANCE_SETTINGS = {
    'max_portfolio_size': 150,
    'enable_parallel_batch': False,     # Future: parallel batch generation
    'cache_exposure_calcs': True,
    'max_filter_iterations': 10
}


# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

VALIDATION_SETTINGS = {
    'enforce_lineup_count': True,
    'min_portfolio_size': 1,
    'max_portfolio_size': 150,
    'validate_exposure_compliance': True,
    'strict_mode': False                 # Treat soft caps as hard caps
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_contest_portfolio_preset(contest_type: str) -> dict:
    """Get portfolio preset for a contest type"""
    return CONTEST_PORTFOLIO_PRESETS.get(
        contest_type.upper(),
        CONTEST_PORTFOLIO_PRESETS['20_ENTRY_GPP']
    )


def get_exposure_limit(contest_type: str) -> float:
    """Get appropriate exposure limit for contest type"""
    exposure_map = {
        '150_ENTRY_GPP': 35.0,
        '20_ENTRY_GPP': 40.0,
        '3_ENTRY_MAX': 67.0,
        'SINGLE_ENTRY': 100.0,
        'CASH_MULTI': 60.0
    }
    
    return exposure_map.get(contest_type.upper(), 40.0)


def get_tier_distribution(style: str) -> dict:
    """Get tier distribution for a portfolio style"""
    return TIER_DISTRIBUTIONS.get(
        style.lower(),
        TIER_DISTRIBUTIONS['balanced']
    )


def get_filter_preset(contest_type: str) -> list:
    """Get recommended filter preset for contest type"""
    
    if 'CASH' in contest_type.upper():
        return [
            {'type': 'duplicates'},
            {'type': 'projection', 'min': 115.0},
            {'type': 'ownership', 'max': 220.0},
            {'type': 'similarity', 'min_unique_players': 2}
        ]
    
    elif 'CONTRARIAN' in contest_type.upper() or '3_ENTRY' in contest_type.upper():
        return [
            {'type': 'duplicates'},
            {'type': 'ownership', 'max': 200.0},
            {'type': 'correlation', 'min': 55.0},
            {'type': 'similarity', 'min_unique_players': 4}
        ]
    
    else:  # GPP default
        return [
            {'type': 'duplicates'},
            {'type': 'projection', 'min': 110.0},
            {'type': 'ceiling', 'min': 140.0},
            {'type': 'correlation', 'min': 50.0},
            {'type': 'stacks', 'require_qb_stack': True},
            {'type': 'similarity', 'min_unique_players': 3}
        ]


def validate_portfolio_config(config: dict) -> bool:
    """Validate portfolio configuration"""
    required_keys = ['num_lineups', 'mode']
    
    for key in required_keys:
        if key not in config:
            return False
    
    # Validate num_lineups
    if not (VALIDATION_SETTINGS['min_portfolio_size'] <= 
            config['num_lineups'] <= 
            VALIDATION_SETTINGS['max_portfolio_size']):
        return False
    
    return True


def get_recommended_batch_size(total_lineups: int) -> int:
    """Get recommended batch size for total lineups"""
    if total_lineups <= 20:
        return total_lineups
    elif total_lineups <= 50:
        return 20
    elif total_lineups <= 100:
        return 50
    else:
        return 50


# ============================================================================
# EXPORT ALL SETTINGS
# ============================================================================

__all__ = [
    'EXPOSURE_SETTINGS',
    'PORTFOLIO_SETTINGS',
    'CONTEST_PORTFOLIO_PRESETS',
    'FILTER_SETTINGS',
    'TIER_DISTRIBUTIONS',
    'TIER_SETTINGS',
    'METRICS_SETTINGS',
    'PERFORMANCE_SETTINGS',
    'VALIDATION_SETTINGS',
    'get_contest_portfolio_preset',
    'get_exposure_limit',
    'get_tier_distribution',
    'get_filter_preset',
    'validate_portfolio_config',
    'get_recommended_batch_size'
]
