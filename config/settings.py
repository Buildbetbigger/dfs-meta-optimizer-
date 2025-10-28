"""
Configuration Settings for DFS Meta-Optimizer
Version 5.0.0

Centralized configuration for all optimizer settings including:
- DFS sport constraints
- Optimization modes and weights
- Claude AI integration
- Feature flags and module toggles
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
# CORE DFS CONSTRAINTS
# ==============================================================================

SALARY_CAP = 50000
ROSTER_SIZE = 9
MIN_SALARY_PCT = 0.96  # Use at least 96% of salary cap

# Position requirements (DraftKings NFL standard)
POSITION_REQUIREMENTS = {
    'QB': 1,
    'RB': 2,
    'WR': 3,
    'TE': 1,
    'FLEX': 1,  # RB/WR/TE
    'DST': 1
}

# ==============================================================================
# CLAUDE AI SETTINGS
# ==============================================================================

# Enable/disable Claude AI features
ENABLE_CLAUDE_AI = True
AI_OWNERSHIP_PREDICTION = True
AI_STRATEGIC_ANALYSIS = True
AI_NEWS_ANALYSIS = False  # Future feature

# API Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Try Streamlit secrets if environment variable not found
try:
    import streamlit as st
    if not ANTHROPIC_API_KEY and hasattr(st, 'secrets'):
        ANTHROPIC_API_KEY = st.secrets.get('ANTHROPIC_API_KEY', '')
except ImportError:
    pass

# Claude model settings
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 4000
CLAUDE_TEMPERATURE = 0.7

# Cost tracking
ESTIMATED_COST_PER_REQUEST = 0.006  # Approximately $0.006 per API call

# ==============================================================================
# OPTIMIZATION MODES
# ==============================================================================

OPTIMIZATION_MODES = {
    'conservative': {
        'description': 'Safe cash game builds - high floor, consistent scoring',
        'projection_weight': 1.5,
        'ceiling_weight': 0.3,
        'leverage_weight': 0.1,
        'ownership_penalty': 0.05,
        'floor_weight': 0.8,
        'correlation_weight': 0.2,
        'population_size': 50,
        'generations': 30,
        'mutation_rate': 0.15
    },
    
    'balanced': {
        'description': 'Balanced GPP approach - mix of safety and upside',
        'projection_weight': 1.0,
        'ceiling_weight': 0.7,
        'leverage_weight': 0.6,
        'ownership_penalty': 0.15,
        'floor_weight': 0.4,
        'correlation_weight': 0.4,
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.20
    },
    
    'aggressive': {
        'description': 'High-risk GPP builds - maximum ceiling and leverage',
        'projection_weight': 0.5,
        'ceiling_weight': 1.5,
        'leverage_weight': 1.2,
        'ownership_penalty': 0.3,
        'floor_weight': 0.1,
        'correlation_weight': 0.5,
        'population_size': 150,
        'generations': 75,
        'mutation_rate': 0.25
    },
    
    'contrarian': {
        'description': 'Anti-chalk strategy with low-owned plays',
        'projection_weight': 0.6,
        'ceiling_weight': 1.0,
        'leverage_weight': 1.0,
        'ownership_penalty': 0.5,  # Heavy ownership fade
        'floor_weight': 0.2,
        'correlation_weight': 0.3,
        'population_size': 100,
        'generations': 60,
        'mutation_rate': 0.22
    },
    
    'leverage': {
        'description': 'Leverage-first approach - beat the field',
        'projection_weight': 0.7,
        'ceiling_weight': 1.0,
        'leverage_weight': 1.5,  # Maximum leverage focus
        'ownership_penalty': 0.2,
        'floor_weight': 0.3,
        'correlation_weight': 0.4,
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.20
    },
    
    'cash': {
        'description': 'Cash game optimized - prioritize floor and safety',
        'projection_weight': 1.8,
        'ceiling_weight': 0.2,
        'leverage_weight': 0.05,
        'ownership_penalty': 0.03,
        'floor_weight': 1.0,
        'correlation_weight': 0.3,
        'population_size': 50,
        'generations': 30,
        'mutation_rate': 0.12
    }
}

# Default mode
DEFAULT_MODE = 'balanced'

# ==============================================================================
# OPPONENT MODELING SETTINGS
# ==============================================================================

# Ownership classification thresholds
HIGH_OWNERSHIP_THRESHOLD = 30.0  # >30% = chalk
LOW_OWNERSHIP_THRESHOLD = 10.0   # <10% = contrarian

# Leverage calculation settings
MIN_LEVERAGE_THRESHOLD = 1.5     # projection / ownership ratio
HIGH_LEVERAGE_THRESHOLD = 3.0

# Field construction assumptions
AVERAGE_FIELD_OWNERSHIP = 15.0   # Default if no AI predictions
FIELD_SIZE_ASSUMPTION = 10000    # Assume 10K entries for leverage calcs

# ==============================================================================
# LINEUP GENERATION SETTINGS
# ==============================================================================

# Portfolio settings
MAX_PLAYER_EXPOSURE = 0.50       # Max 50% exposure per player across portfolio
MIN_PLAYER_DIFFERENCE = 2        # At least 2 different players between lineups
DEFAULT_LINEUP_COUNT = 20
MAX_LINEUP_COUNT = 150

# Diversity settings
ENFORCE_UNIQUENESS = True
MIN_UNIQUE_PLAYERS = 0.60        # 60% of players should be unique across portfolio

# ==============================================================================
# GENETIC ALGORITHM SETTINGS (for Module 2+)
# ==============================================================================

GA_DEFAULT_SETTINGS = {
    'population_size': 200,
    'generations': 100,
    'mutation_rate': 0.15,
    'crossover_rate': 0.7,
    'elite_size': 20,
    'tournament_size': 5
}

# ==============================================================================
# STACKING SETTINGS (for Module 2+)
# ==============================================================================

STACKING_ENABLED = True
MIN_STACK_CORRELATION = 0.70     # Minimum correlation for valid stack
QB_STACK_WEIGHT = 1.15           # 15% bonus for QB stacks

# Correlation coefficients
CORRELATIONS = {
    'qb_wr_same_team': 0.85,
    'qb_te_same_team': 0.80,
    'qb_rb_same_team': 0.25,
    'same_team_general': 0.20,
    'opposing_offense': 0.30,
    'defense_opposing': -0.40
}

# ==============================================================================
# VALIDATION SETTINGS
# ==============================================================================

# Player data validation
MIN_PROJECTION = 0.0
MAX_PROJECTION = 100.0
MIN_OWNERSHIP = 0.0
MAX_OWNERSHIP = 100.0
MIN_SALARY = 3000
MAX_SALARY = 12000

# Lineup validation
REQUIRE_VALID_POSITIONS = True
REQUIRE_VALID_TEAMS = True
ALLOW_DUPLICATE_PLAYERS = False

# ==============================================================================
# PERFORMANCE SETTINGS
# ==============================================================================

MAX_GENERATION_TIME = 300        # 5 minutes max
ENABLE_CACHING = True
CACHE_SIZE = 1000

# Parallel processing (disabled by default for Streamlit compatibility)
ENABLE_MULTIPROCESSING = False
MAX_WORKERS = 4

# ==============================================================================
# UI SETTINGS
# ==============================================================================

SHOW_PROGRESS = True
SHOW_DETAILED_STATS = True
DEFAULT_DECIMAL_PLACES = {
    'projection': 1,
    'ownership': 1,
    'salary': 0,
    'leverage': 2,
    'correlation': 1
}

# Color scheme for UI
COLORS = {
    'high_leverage': '#90EE90',  # Light green
    'chalk': '#FFB6C1',          # Light red
    'contrarian': '#ADD8E6',     # Light blue
    'balanced': '#FFFFCC'        # Light yellow
}

# ==============================================================================
# FEATURE FLAGS
# ==============================================================================

# Module toggles
ENABLE_MODULE_2 = True   # Genetic optimization + stacking
ENABLE_MODULE_3 = True   # Portfolio optimization
ENABLE_MODULE_4 = False  # Real-time data (not implemented yet)
ENABLE_MODULE_5 = False  # Monte Carlo simulation (not implemented yet)

# Advanced features
ENABLE_VARIANCE_ANALYSIS = True
ENABLE_EXPOSURE_MANAGEMENT = True
ENABLE_CORRELATION_OPTIMIZATION = True

# ==============================================================================
# DEBUG & LOGGING
# ==============================================================================

DEBUG_MODE = False
VERBOSE_LOGGING = False
LOG_FILE = "dfs_optimizer.log"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def validate_api_key() -> bool:
    """Check if API key is configured"""
    return bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.startswith('sk-ant-'))


def get_optimization_mode(mode_name: str) -> dict:
    """
    Get optimization mode configuration
    
    Args:
        mode_name: Name of the mode
        
    Returns:
        Mode configuration dict
    """
    return OPTIMIZATION_MODES.get(mode_name, OPTIMIZATION_MODES[DEFAULT_MODE])


def get_required_salary() -> int:
    """Calculate minimum required salary based on MIN_SALARY_PCT"""
    return int(SALARY_CAP * MIN_SALARY_PCT)


def validate_settings() -> List[str]:
    """
    Validate critical settings
    
    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    
    # Check AI configuration
    if ENABLE_CLAUDE_AI and not validate_api_key():
        errors.append("ANTHROPIC_API_KEY not set or invalid, but AI is enabled")
    
    # Check salary constraints
    if MIN_SALARY_PCT < 0.90 or MIN_SALARY_PCT > 1.0:
        errors.append(f"MIN_SALARY_PCT should be 0.90-1.0, got {MIN_SALARY_PCT}")
    
    # Check optimization modes have required keys
    required_keys = ['projection_weight', 'ceiling_weight', 'leverage_weight']
    for mode_name, mode_config in OPTIMIZATION_MODES.items():
        for key in required_keys:
            if key not in mode_config:
                errors.append(f"Mode '{mode_name}' missing required key: {key}")
    
    return errors


def print_config_summary():
    """Print summary of current configuration"""
    print("=" * 70)
    print("DFS META-OPTIMIZER CONFIGURATION")
    print("=" * 70)
    print(f"Salary Cap: ${SALARY_CAP:,}")
    print(f"Roster Size: {ROSTER_SIZE}")
    print(f"Min Salary Usage: {MIN_SALARY_PCT*100:.0f}% (${get_required_salary():,})")
    print()
    print("Optimization Modes Available:")
    for mode, config in OPTIMIZATION_MODES.items():
        print(f"  • {mode}: {config['description']}")
    print()
    print(f"Claude AI: {'✅ Enabled' if ENABLE_CLAUDE_AI and validate_api_key() else '❌ Disabled'}")
    print(f"Module 2 (Genetic/Stacking): {'✅ Enabled' if ENABLE_MODULE_2 else '❌ Disabled'}")
    print(f"Module 3 (Portfolio): {'✅ Enabled' if ENABLE_MODULE_3 else '❌ Disabled'}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("=" * 70)


# Run validation on import (optional - can comment out)
_validation_errors = validate_settings()
if _validation_errors and DEBUG_MODE:
    print("⚠️  Configuration validation warnings:")
    for error in _validation_errors:
        print(f"   - {error}")


# Export all settings
__all__ = [
    'SALARY_CAP', 'ROSTER_SIZE', 'MIN_SALARY_PCT',
    'POSITION_REQUIREMENTS',
    'ENABLE_CLAUDE_AI', 'AI_OWNERSHIP_PREDICTION', 'AI_STRATEGIC_ANALYSIS',
    'ANTHROPIC_API_KEY', 'CLAUDE_MODEL', 'CLAUDE_MAX_TOKENS',
    'OPTIMIZATION_MODES', 'DEFAULT_MODE',
    'HIGH_OWNERSHIP_THRESHOLD', 'LOW_OWNERSHIP_THRESHOLD',
    'MAX_PLAYER_EXPOSURE', 'DEFAULT_LINEUP_COUNT',
    'validate_api_key', 'get_optimization_mode', 'get_required_salary',
    'print_config_summary'
]
