"""
Configuration Settings for DFS Meta-Optimizer

All settings for the optimizer including:
- Optimization modes
- DFS constraints
- AI settings
- Feature flags
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ====================
# CORE DFS CONSTRAINTS
# ====================

SALARY_CAP = 50000
ROSTER_SIZE = 9
MIN_SALARY_PCT = 0.96  # Use 96% of salary cap minimum

# Position requirements (DraftKings standard)
POSITION_REQUIREMENTS = {
    'QB': 1,
    'RB': 2,
    'WR': 3,
    'TE': 1,
    'FLEX': 1,  # RB/WR/TE
    'DST': 1
}

# ====================
# CLAUDE AI SETTINGS
# ====================

# Enable/disable Claude AI features
ENABLE_CLAUDE_AI = True
AI_OWNERSHIP_PREDICTION = True
AI_STRATEGIC_ANALYSIS = True

# API Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 4000
CLAUDE_TEMPERATURE = 0.7

# Cost tracking
ESTIMATED_COST_PER_REQUEST = 0.006  # ~$0.006 per request

# ====================
# OPTIMIZATION MODES
# ====================

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
        'leverage_weight': 0.4,
        'ownership_penalty': 0.15,
        'floor_weight': 0.4,
        'correlation_weight': 0.5,
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.2
    },
    
    'leverage': {
        'description': 'High-leverage tournament plays - maximize ceiling/ownership ratio',
        'projection_weight': 0.7,
        'ceiling_weight': 1.2,
        'leverage_weight': 1.0,
        'ownership_penalty': 0.25,
        'floor_weight': 0.2,
        'correlation_weight': 0.7,
        'population_size': 150,
        'generations': 75,
        'mutation_rate': 0.25
    },
    
    'contrarian': {
        'description': 'Maximum differentiation - fade the chalk, unique builds',
        'projection_weight': 0.5,
        'ceiling_weight': 1.0,
        'leverage_weight': 1.2,
        'ownership_penalty': 0.4,
        'floor_weight': 0.1,
        'correlation_weight': 0.8,
        'population_size': 150,
        'generations': 75,
        'mutation_rate': 0.3
    },
    
    'cash': {
        'description': 'Cash game optimization - maximize floor and consistency',
        'projection_weight': 1.8,
        'ceiling_weight': 0.2,
        'leverage_weight': 0.0,
        'ownership_penalty': 0.0,
        'floor_weight': 1.2,
        'correlation_weight': 0.3,
        'population_size': 40,
        'generations': 25,
        'mutation_rate': 0.1
    }
}

# Default mode
DEFAULT_MODE = 'balanced'

# ====================
# LINEUP GENERATION
# ====================

# Diversity settings
DEFAULT_DIVERSITY_FACTOR = 0.3
MIN_UNIQUE_PLAYERS = 3  # Minimum different players between lineups

# Duplicate detection threshold
DUPLICATE_THRESHOLD = 7  # Max matching players before considering duplicate

# ====================
# STACKING PREFERENCES
# ====================

# Correlation thresholds
MIN_QB_STACK_CORRELATION = 0.5
MIN_GAME_STACK_CORRELATION = 0.4

# Stack enforcement
ENFORCE_QB_STACKS = True
MIN_STACK_SIZE = 2  # QB + at least 1 pass catcher

# ====================
# EXPOSURE MANAGEMENT
# ====================

# Global exposure limits
GLOBAL_MAX_EXPOSURE = 40.0  # 40% max for any player
GLOBAL_MIN_EXPOSURE = 0.0

# Captain exposure (for showdown)
MAX_CAPTAIN_EXPOSURE = 30.0

# ====================
# PORTFOLIO SETTINGS
# ====================

# Multi-entry defaults
DEFAULT_PORTFOLIO_SIZE = 20
MAX_PORTFOLIO_SIZE = 150

# Tiered portfolio distribution
TIERED_DISTRIBUTION = {
    'safe': 0.30,      # 30% conservative plays
    'balanced': 0.50,  # 50% balanced builds
    'contrarian': 0.20 # 20% high-risk/reward
}

# ====================
# SIMULATION SETTINGS
# ====================

# Monte Carlo defaults
DEFAULT_SIMULATIONS = 10000
QUICK_SIMULATIONS = 1000
THOROUGH_SIMULATIONS = 50000

# Contest simulation
DEFAULT_CONTEST_SIZE = 150000  # Milly Maker size
DEFAULT_FIELD_AVG = 140.0
DEFAULT_FIELD_STD = 15.0

# ====================
# REAL-TIME DATA
# ====================

# News impact thresholds
CRITICAL_NEWS_IMPACT = 80
HIGH_NEWS_IMPACT = 60
MEDIUM_NEWS_IMPACT = 40

# Vegas line movement thresholds
SIGNIFICANT_SPREAD_MOVE = 1.0  # Points
SIGNIFICANT_TOTAL_MOVE = 2.0   # Points

# Ownership thresholds
CHALK_THRESHOLD = 25.0    # 25%+ is chalk
LEVERAGE_THRESHOLD = 15.0  # <15% is leverage play

# ====================
# DISPLAY SETTINGS
# ====================

# Number formatting
CURRENCY_FORMAT = "${:,.0f}"
PERCENTAGE_FORMAT = "{:.1f}%"
DECIMAL_FORMAT = "{:.2f}"

# Lineup display
SHOW_CAPTAIN_INDICATOR = True
SHOW_LEVERAGE_SCORES = True
SHOW_CORRELATION_SCORES = True

# ====================
# PERFORMANCE TUNING
# ====================

# Genetic algorithm limits
MAX_GENERATIONS = 200
MAX_POPULATION_SIZE = 300

# Lineup generation timeout
GENERATION_TIMEOUT_SECONDS = 300  # 5 minutes max

# Memory management
MAX_CACHED_LINEUPS = 1000

# ====================
# FEATURE FLAGS
# ====================

# Module toggles
ENABLE_MODULE_2 = True  # Genetic optimization
ENABLE_MODULE_3 = True  # Portfolio optimization
ENABLE_MODULE_4 = True  # Real-time data
ENABLE_MODULE_5 = True  # Monte Carlo simulation

# Advanced features
ENABLE_STACKING = True
ENABLE_EXPOSURE_MANAGEMENT = True
ENABLE_VARIANCE_ANALYSIS = True
ENABLE_CONTEST_SELECTION = True

# ====================
# DEBUG & LOGGING
# ====================

DEBUG_MODE = False
VERBOSE_LOGGING = False
SHOW_GENERATION_PROGRESS = True

# Log file
LOG_FILE = "dfs_optimizer.log"

# ====================
# VALIDATION
# ====================

def validate_settings():
    """Validate critical settings"""
    errors = []
    
    # Check AI key if AI enabled
    if ENABLE_CLAUDE_AI and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY not set but AI is enabled")
    
    # Check salary constraints
    if MIN_SALARY_PCT < 0.9 or MIN_SALARY_PCT > 1.0:
        errors.append(f"MIN_SALARY_PCT should be 0.9-1.0, got {MIN_SALARY_PCT}")
    
    # Check optimization modes
    for mode_name, mode_config in OPTIMIZATION_MODES.items():
        required_keys = ['projection_weight', 'ceiling_weight', 'leverage_weight']
        for key in required_keys:
            if key not in mode_config:
                errors.append(f"Mode '{mode_name}' missing required key: {key}")
    
    return errors

# Run validation on import
_validation_errors = validate_settings()
if _validation_errors:
    print("⚠️  Configuration validation warnings:")
    for error in _validation_errors:
        print(f"   - {error}")
