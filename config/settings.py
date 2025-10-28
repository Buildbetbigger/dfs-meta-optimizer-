"""
Configuration Settings for DFS Meta-Optimizer

All application configuration in one place.
"""

import os
from typing import Dict

# ============================================================================
# DRAFTKINGS SHOWDOWN SETTINGS
# ============================================================================

# Salary cap for DraftKings Showdown
SALARY_CAP = 50000

# Roster size (1 Captain + 5 FLEX)
ROSTER_SIZE = 6

# Captain point multiplier
CAPTAIN_MULTIPLIER = 1.5

# Minimum salary cap usage (as percentage)
MIN_SALARY_USAGE = 0.96  # 96%

# ============================================================================
# OPTIMIZATION MODES
# ============================================================================

# Define different optimization strategies
# Each mode has different weight configurations for:
# - projection_weight: How much to value projected points
# - leverage_weight: How much to value leverage score
# - ceiling_weight: How much to value upside potential
# - ownership_weight: How much to fade high ownership (higher = more fade)

OPTIMIZATION_MODES = {
    'conservative': {
        'projection_weight': 1.0,    # Focus on projections
        'leverage_weight': 0.2,      # Light leverage consideration
        'ceiling_weight': 0.1,       # Minimal ceiling focus
        'ownership_weight': 0.02,    # Slight chalk fade
        'description': 'Projections-first with high-owned players'
    },
    
    'balanced': {
        'projection_weight': 0.8,    # Strong projection focus
        'leverage_weight': 0.6,      # Moderate leverage
        'ceiling_weight': 0.4,       # Moderate ceiling
        'ownership_weight': 0.01,    # Light ownership consideration
        'description': 'Balanced approach with moderate leverage'
    },
    
    'leverage': {
        'projection_weight': 0.6,    # Moderate projection focus
        'leverage_weight': 1.0,      # Maximum leverage focus
        'ceiling_weight': 0.8,       # High ceiling focus
        'ownership_weight': 0.005,   # Minimal ownership fade
        'description': 'Leverage-first with high upside plays'
    },
    
    'contrarian': {
        'projection_weight': 0.5,    # Lower projection focus
        'leverage_weight': 1.0,      # Maximum leverage
        'ceiling_weight': 0.9,       # Maximum ceiling focus
        'ownership_weight': 0.001,   # Almost no ownership penalty
        'description': 'Anti-chalk strategy with contrarian picks'
    },
    
    'cash': {
        'projection_weight': 1.0,    # Maximum projection focus
        'leverage_weight': 0.1,      # Minimal leverage
        'ceiling_weight': 0.0,       # No ceiling consideration
        'ownership_weight': 0.03,    # Moderate chalk fade for safety
        'description': 'Cash game focused - high floor, safe plays'
    }
}

# Default optimization mode
DEFAULT_MODE = 'balanced'

# ============================================================================
# LINEUP GENERATION SETTINGS
# ============================================================================

# Maximum player exposure across portfolio (0-1)
MAX_PLAYER_EXPOSURE = 0.40  # 40%

# Minimum player difference between lineups
MIN_PLAYER_DIFFERENCE = 2  # At least 2 different players

# Default number of lineups to generate
DEFAULT_LINEUP_COUNT = 20

# Maximum lineups per generation (prevent excessive API usage)
MAX_LINEUP_COUNT = 150

# ============================================================================
# CLAUDE AI SETTINGS
# ============================================================================

# Enable Claude AI features
ENABLE_CLAUDE_AI = True

# Anthropic API key (from environment or Streamlit secrets)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# Try Streamlit secrets if environment variable not found
try:
    import streamlit as st
    if not ANTHROPIC_API_KEY and hasattr(st, 'secrets'):
        ANTHROPIC_API_KEY = st.secrets.get('ANTHROPIC_API_KEY', '')
except ImportError:
    pass

# Claude model to use
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Maximum tokens per Claude API call
CLAUDE_MAX_TOKENS = 2000

# Temperature for Claude responses (0-1, higher = more random)
CLAUDE_TEMPERATURE = 0.7

# ============================================================================
# OPPONENT MODELING SETTINGS
# ============================================================================

# Ownership thresholds for player categorization
CHALK_THRESHOLD = 0.75      # Top 25% ownership = chalk
LEVERAGE_THRESHOLD = 0.75   # Top 25% leverage = good leverage

# Contrarian ownership threshold (below this = contrarian)
CONTRARIAN_THRESHOLD = 0.25  # Bottom 25% ownership

# ============================================================================
# PORTFOLIO OPTIMIZATION SETTINGS
# ============================================================================

# Exposure management
SOFT_EXPOSURE_CAP = 0.35    # 35% soft cap (warning)
HARD_EXPOSURE_CAP = 0.45    # 45% hard cap (prevent)

# Position exposure limits (for classic contests)
POSITION_EXPOSURE_LIMITS = {
    'QB': 0.50,   # 50% max
    'RB': 0.40,   # 40% max
    'WR': 0.40,   # 40% max
    'TE': 0.40,   # 40% max
    'DST': 0.30   # 30% max
}

# Team exposure limits
MAX_TEAM_EXPOSURE = 0.40    # 40% max from single team

# ============================================================================
# CORRELATION SETTINGS
# ============================================================================

# QB stack preferences
QB_STACK_TARGETS = ['WR', 'TE']  # Positions to stack with QB

# Minimum correlation threshold for stacking
MIN_CORRELATION_SCORE = 0.3

# Bring-back recommendations (opponents of stacked team)
ENABLE_BRINGBACK = True
BRINGBACK_POSITIONS = ['WR', 'TE', 'RB']

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================

# Monte Carlo simulation settings
MONTE_CARLO_SIMULATIONS = 10000

# Percentiles to calculate for projections
PROJECTION_PERCENTILES = [10, 25, 50, 75, 90]

# Contest simulation settings
SIMULATE_FIELD_SIZE = 10000  # Number of opponents to simulate

# ============================================================================
# DATA VALIDATION SETTINGS
# ============================================================================

# Required CSV columns
REQUIRED_CSV_COLUMNS = ['name', 'team', 'position', 'salary', 'projection']

# Optional CSV columns (will be calculated if missing)
OPTIONAL_CSV_COLUMNS = ['ownership', 'ceiling', 'floor']

# Valid positions
VALID_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'DST']

# Salary range validation
MIN_PLAYER_SALARY = 2000
MAX_PLAYER_SALARY = 15000

# Projection range validation
MIN_PROJECTION = 0
MAX_PROJECTION = 100

# ============================================================================
# UI SETTINGS
# ============================================================================

# Streamlit page configuration
PAGE_TITLE = "DFS Meta-Optimizer"
PAGE_ICON = "🎯"
LAYOUT = "wide"

# Display settings
SHOW_DEBUG_INFO = False
VERBOSE_LOGGING = True

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

# Default export filename
DEFAULT_EXPORT_FILENAME = "lineups.csv"

# Export format
EXPORT_FORMAT = "draftkings"  # Options: "draftkings", "fanduel", "yahoo"

# Include metadata in export
INCLUDE_METADATA = True

# ============================================================================
# ADVANCED SETTINGS (rarely need to change)
# ============================================================================

# Genetic algorithm settings (if using Module 2)
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 50
GA_MUTATION_RATE = 0.15
GA_CROSSOVER_RATE = 0.80
GA_ELITE_SIZE = 10

# Real-time data refresh interval (seconds)
DATA_REFRESH_INTERVAL = 300  # 5 minutes

# API rate limiting
MAX_API_CALLS_PER_MINUTE = 50
API_TIMEOUT_SECONDS = 30

# ============================================================================
# ENVIRONMENT-SPECIFIC SETTINGS
# ============================================================================

# Development vs Production
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')

if ENVIRONMENT == 'development':
    DEBUG_MODE = True
    VERBOSE_LOGGING = True
    SHOW_DEBUG_INFO = True
else:
    DEBUG_MODE = False
    VERBOSE_LOGGING = False
    SHOW_DEBUG_INFO = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_mode_config(mode: str = None) -> Dict:
    """
    Get configuration for optimization mode
    
    Args:
        mode: Mode name (defaults to DEFAULT_MODE)
    
    Returns:
        Mode configuration dictionary
    """
    if mode is None:
        mode = DEFAULT_MODE
    
    if mode not in OPTIMIZATION_MODES:
        print(f"Warning: Mode '{mode}' not found, using '{DEFAULT_MODE}'")
        mode = DEFAULT_MODE
    
    return OPTIMIZATION_MODES[mode]


def validate_api_key() -> bool:
    """
    Validate that Anthropic API key is configured
    
    Returns:
        True if API key is set, False otherwise
    """
    if not ANTHROPIC_API_KEY:
        return False
    
    if len(ANTHROPIC_API_KEY) < 50:
        return False
    
    if not ANTHROPIC_API_KEY.startswith('sk-ant-'):
        return False
    
    return True


def get_salary_range() -> tuple:
    """
    Get valid salary range
    
    Returns:
        Tuple of (min_salary, max_salary)
    """
    return (MIN_PLAYER_SALARY, MAX_PLAYER_SALARY)


def get_required_salary() -> int:
    """
    Get minimum required salary usage
    
    Returns:
        Minimum salary to use
    """
    return int(SALARY_CAP * MIN_SALARY_USAGE)


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config() -> bool:
    """
    Validate all configuration settings
    
    Returns:
        True if configuration is valid
    """
    errors = []
    
    # Validate salary cap
    if SALARY_CAP <= 0:
        errors.append("SALARY_CAP must be positive")
    
    # Validate roster size
    if ROSTER_SIZE < 2:
        errors.append("ROSTER_SIZE must be at least 2")
    
    # Validate captain multiplier
    if CAPTAIN_MULTIPLIER <= 1:
        errors.append("CAPTAIN_MULTIPLIER must be greater than 1")
    
    # Validate optimization modes
    for mode, config in OPTIMIZATION_MODES.items():
        required_keys = ['projection_weight', 'leverage_weight', 'ceiling_weight', 'ownership_weight']
        for key in required_keys:
            if key not in config:
                errors.append(f"Mode '{mode}' missing required key: {key}")
    
    # Validate exposure settings
    if MAX_PLAYER_EXPOSURE <= 0 or MAX_PLAYER_EXPOSURE > 1:
        errors.append("MAX_PLAYER_EXPOSURE must be between 0 and 1")
    
    # Print errors if any
    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    return True


# Validate configuration on import
if __name__ != '__main__':
    if not validate_config():
        print("⚠️ Warning: Configuration validation failed!")

# ============================================================================
# EXPORT SETTINGS SUMMARY
# ============================================================================

def print_config_summary():
    """Print summary of current configuration"""
    print("=" * 60)
    print("DFS META-OPTIMIZER CONFIGURATION")
    print("=" * 60)
    print(f"Salary Cap: ${SALARY_CAP:,}")
    print(f"Roster Size: {ROSTER_SIZE} (1 CPT + {ROSTER_SIZE-1} FLEX)")
    print(f"Captain Multiplier: {CAPTAIN_MULTIPLIER}x")
    print(f"Min Salary Usage: {MIN_SALARY_USAGE*100:.0f}% (${get_required_salary():,})")
    print()
    print("Optimization Modes:")
    for mode, config in OPTIMIZATION_MODES.items():
        print(f"  {mode}: {config.get('description', 'No description')}")
    print()
    print(f"Claude AI: {'Enabled' if ENABLE_CLAUDE_AI and validate_api_key() else 'Disabled'}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("=" * 60)


# Uncomment to see config summary on import
# print_config_summary()
