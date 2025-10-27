"""
Module 5: Configuration
Advanced Simulation & Analysis settings
"""

from typing import Dict


# ============================================================================
# MONTE CARLO SIMULATION CONFIGURATION
# ============================================================================

SIMULATION_CONFIG = {
    # Default simulation settings
    'default_simulations': 10000,      # Number of simulations to run
    'fast_simulations': 5000,          # Quick simulations
    'thorough_simulations': 20000,     # Detailed simulations
    
    # Player variance estimation
    'variance_by_position': {
        'QB': 0.40,     # QBs have 40% variance
        'RB': 0.35,     # RBs have 35% variance
        'WR': 0.35,     # WRs have 35% variance
        'TE': 0.30,     # TEs have 30% variance (more consistent)
        'DST': 0.50     # DST has highest variance
    },
    
    # Minimum variance
    'min_std_dev': 2.0,  # Minimum 2 pts std deviation
    
    # Field simulation
    'default_field_avg': 140.0,  # Average field score
    'default_field_std': 15.0,   # Field score std deviation
    
    # Contest sizes (for quick testing)
    'contest_sizes': {
        'small': 1000,
        'medium': 10000,
        'large': 50000,
        'milly_maker': 150000
    }
}


# ============================================================================
# LINEUP EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    # Default scoring weights
    'default_weights': {
        'projection': 1.0,
        'ceiling': 1.0,
        'leverage': 1.0,
        'correlation': 1.0,
        'uniqueness': 0.5,
        'efficiency': 0.5
    },
    
    # GPP weights (prioritize ceiling and leverage)
    'gpp_weights': {
        'projection': 0.8,
        'ceiling': 1.5,
        'leverage': 1.5,
        'correlation': 1.2,
        'uniqueness': 0.8,
        'efficiency': 0.3
    },
    
    # Cash weights (prioritize floor and consistency)
    'cash_weights': {
        'projection': 1.5,
        'ceiling': 0.5,
        'leverage': 0.3,
        'correlation': 0.8,
        'uniqueness': 0.2,
        'efficiency': 1.0
    },
    
    # Correlation scoring
    'correlation_points': {
        'qb_pass_catcher': 20,    # QB + same team WR/TE
        'rb_dst': 15,              # RB + same team DST
        'game_stack_4plus': 10     # 4+ players from same game
    },
    
    # Salary cap
    'salary_cap': 50000,
    
    # Selection strategies
    'balanced_selection': {
        'high_ceiling_pct': 0.40,
        'high_leverage_pct': 0.30,
        'high_correlation_pct': 0.20,
        'safe_pct': 0.10
    }
}


# ============================================================================
# STRATEGY OPTIMIZATION CONFIGURATION
# ============================================================================

OPTIMIZATION_CONFIG = {
    # Simulation counts for optimization
    'quick_test_sims': 3000,      # Quick parameter test
    'standard_test_sims': 5000,   # Standard test
    'thorough_test_sims': 10000,  # Thorough test
    
    # Grid search limits
    'max_combinations': 50,  # Max parameter combinations to test
    
    # Common parameter ranges for testing
    'exposure_range': [20, 25, 30, 35, 40],
    'stack_rate_range': [0.5, 0.6, 0.7, 0.8, 0.9],
    
    # Optimization objectives
    'objectives': [
        'expected_roi',
        'win_rate',
        'cash_rate',
        'top_10_rate'
    ],
    
    # Contest-specific objectives
    'contest_objectives': {
        'CASH': 'cash_rate',
        'GPP': 'expected_roi',
        'MILLY_MAKER': 'win_rate',
        'SATELLITE': 'cash_rate'
    }
}


# ============================================================================
# VARIANCE ANALYSIS CONFIGURATION
# ============================================================================

VARIANCE_CONFIG = {
    # Variance thresholds
    'high_variance_threshold': 0.40,    # CV > 0.4 = boom-or-bust
    'medium_variance_threshold': 0.25,  # CV 0.25-0.4 = medium
    # Low variance = CV < 0.25
    
    # Risk tolerance multipliers
    'risk_tolerance_values': {
        'conservative': 0.2,
        'balanced': 0.5,
        'aggressive': 0.8
    },
    
    # Portfolio recommendations
    'portfolio_variance_mix': {
        'conservative': {
            'high_variance': 0.20,
            'medium_variance': 0.40,
            'low_variance': 0.40
        },
        'balanced': {
            'high_variance': 0.40,
            'medium_variance': 0.40,
            'low_variance': 0.20
        },
        'aggressive': {
            'high_variance': 0.60,
            'medium_variance': 0.30,
            'low_variance': 0.10
        }
    },
    
    # Simulation settings for variance analysis
    'variance_sims': 1000  # Simulations for single lineup analysis
}


# ============================================================================
# CONTEST SELECTION CONFIGURATION
# ============================================================================

CONTEST_CONFIG = {
    # Field strength estimation
    'field_strength_by_size': {
        'small': 50,      # <1K entries
        'medium': 60,     # 1K-10K entries
        'large': 70,      # 10K-50K entries
        'massive': 80,    # 50K-100K entries
        'milly': 85       # 100K+ entries
    },
    
    'field_strength_by_fee': {
        'low': 0,         # <$10 entry
        'medium': 5,      # $10-$50 entry
        'high': 10        # $50+ entry
    },
    
    # EV calculation defaults
    'default_portfolio_strength': 75,  # Assume 75/100 skill level
    
    # Cash multipliers
    'cash_multiplier': 1.8,  # Cash games pay 1.8x entry
    
    # Kelly Criterion settings
    'default_kelly_fraction': 0.25,  # Use quarter Kelly (conservative)
    'max_bankroll_pct': 0.10,        # Never risk >10% of bankroll
    
    # Win probability caps
    'max_win_probability': 0.05,  # Cap at 5% win rate
    'min_win_probability': 0.0001,  # Floor at 0.01%
    
    # Cash probability ranges
    'cash_probability_range': {
        'CASH': {'base': 0.50, 'skill_factor': 200},
        'GPP': {'base': 0.20, 'skill_factor': 250},
        'SATELLITE': {'base': 0.45, 'skill_factor': 200}
    }
}


# ============================================================================
# RESULTS TRACKING CONFIGURATION
# ============================================================================

TRACKING_CONFIG = {
    # Ownership accuracy targets
    'ownership_accuracy_targets': {
        'excellent': 3.0,   # MAE <= 3%
        'good': 5.0,        # MAE <= 5%
        'acceptable': 8.0,  # MAE <= 8%
        'poor': 999.9       # MAE > 8%
    },
    
    # Performance thresholds
    'roi_thresholds': {
        'excellent': 15.0,  # 15%+ ROI
        'good': 10.0,       # 10-15% ROI
        'break_even': 0.0,  # 0-10% ROI
        'losing': -999.9    # <0% ROI
    },
    
    # Export settings
    'auto_export': False,
    'export_path': '/mnt/user-data/outputs/results/',
    'export_format': 'csv'
}


# ============================================================================
# INTEGRATION SETTINGS
# ============================================================================

INTEGRATION_CONFIG = {
    # Which modules are enabled
    'modules_enabled': {
        'monte_carlo': True,
        'lineup_evaluator': True,
        'strategy_optimizer': True,
        'variance_analyzer': True,
        'contest_selector': True,
        'results_tracker': True
    },
    
    # Performance settings
    'parallel_processing': False,  # Enable multiprocessing (advanced)
    'num_workers': 4,              # Worker processes if parallel enabled
    
    # Caching
    'cache_simulations': False,    # Cache simulation results
    'cache_ttl': 3600,             # Cache time-to-live (seconds)
    
    # Logging
    'logging': {
        'level': 'INFO',
        'log_file': None,
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    'QUICK_TEST': {
        'simulations': SIMULATION_CONFIG['fast_simulations'],
        'optimization_sims': OPTIMIZATION_CONFIG['quick_test_sims'],
        'description': 'Fast testing mode (5K sims)'
    },
    
    'STANDARD': {
        'simulations': SIMULATION_CONFIG['default_simulations'],
        'optimization_sims': OPTIMIZATION_CONFIG['standard_test_sims'],
        'description': 'Standard mode (10K sims)'
    },
    
    'THOROUGH': {
        'simulations': SIMULATION_CONFIG['thorough_simulations'],
        'optimization_sims': OPTIMIZATION_CONFIG['thorough_test_sims'],
        'description': 'Thorough analysis (20K sims)'
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_preset(preset_name: str) -> Dict:
    """
    Get preset configuration.
    
    Args:
        preset_name: 'QUICK_TEST', 'STANDARD', or 'THOROUGH'
    
    Returns:
        Preset configuration dict
    """
    return PRESETS.get(preset_name, PRESETS['STANDARD'])


def get_evaluation_weights(contest_type: str) -> Dict[str, float]:
    """
    Get evaluation weights for contest type.
    
    Args:
        contest_type: 'GPP' or 'CASH'
    
    Returns:
        Weights dictionary
    """
    if contest_type == 'GPP':
        return EVALUATION_CONFIG['gpp_weights']
    elif contest_type == 'CASH':
        return EVALUATION_CONFIG['cash_weights']
    else:
        return EVALUATION_CONFIG['default_weights']


def get_variance_threshold(risk_level: str) -> float:
    """
    Get variance threshold for risk level.
    
    Args:
        risk_level: 'high', 'medium', or 'low'
    
    Returns:
        Coefficient of variation threshold
    """
    if risk_level == 'high':
        return VARIANCE_CONFIG['high_variance_threshold']
    elif risk_level == 'medium':
        return VARIANCE_CONFIG['medium_variance_threshold']
    else:
        return 0.0


def get_portfolio_variance_mix(risk_profile: str) -> Dict[str, float]:
    """
    Get recommended portfolio variance mix.
    
    Args:
        risk_profile: 'conservative', 'balanced', or 'aggressive'
    
    Returns:
        Mix percentages dict
    """
    return VARIANCE_CONFIG['portfolio_variance_mix'].get(
        risk_profile,
        VARIANCE_CONFIG['portfolio_variance_mix']['balanced']
    )
