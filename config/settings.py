"""
Configuration settings for DFS Meta-Optimizer
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONTEST SETTINGS
# ============================================================================

SALARY_CAP = 50000
ROSTER_SIZE = 6  # 1 Captain + 5 Flex
CAPTAIN_MULTIPLIER = 1.5  # Captain gets 1.5x points and costs 1.5x salary

# ============================================================================
# OPPONENT MODELING SETTINGS
# ============================================================================

# Ownership thresholds
HIGH_OWNERSHIP_THRESHOLD = 40  # % - Flag as "chalk" above this
LOW_OWNERSHIP_THRESHOLD = 10   # % - Flag as "contrarian" below this
CHALK_PENALTY_WEIGHT = 0.3     # How much to penalize high ownership

# Leverage calculation
MIN_ACCEPTABLE_LEVERAGE = 1.5  # Minimum leverage score to consider
LEVERAGE_WEIGHT = 0.4          # How much to weight leverage vs projection

# Field distribution modeling
FIELD_SIZE_DEFAULT = 10000     # Assumed contest size for modeling
WINNING_THRESHOLD_PERCENTILE = 99  # Top 1% target for GPP

# ============================================================================
# OPTIMIZATION MODES
# ============================================================================

OPTIMIZATION_MODES = {
    'anti_chalk': {
        'description': 'Maximum differentiation from field',
        'leverage_weight': 0.6,
        'projection_weight': 0.4,
        'max_avg_ownership': 25,
        'prioritize_contrarian': True
    },
    'leverage': {
        'description': 'Optimize ceiling/ownership ratio',
        'leverage_weight': 0.5,
        'projection_weight': 0.5,
        'max_avg_ownership': 30,
        'prioritize_contrarian': False
    },
    'balanced': {
        'description': 'Mix of projection and leverage',
        'leverage_weight': 0.3,
        'projection_weight': 0.7,
        'max_avg_ownership': 35,
        'prioritize_contrarian': False
    },
    'safe': {
        'description': 'High floor, lower risk',
        'leverage_weight': 0.1,
        'projection_weight': 0.9,
        'max_avg_ownership': 50,
        'prioritize_contrarian': False
    }
}

# ============================================================================
# LINEUP GENERATION SETTINGS
# ============================================================================

DEFAULT_LINEUP_COUNT = 20
MAX_LINEUP_COUNT = 150
MIN_PLAYER_DIFFERENCE = 3  # Minimum players different between lineups
MAX_PLAYER_EXPOSURE = 0.7  # Maximum % of lineups a player can appear in

# Stacking preferences
ENABLE_STACKING = True
QB_STACK_CORRELATION_BONUS = 0.15  # Bonus for QB + pass catcher

# ============================================================================
# REAL-TIME ADAPTATION SETTINGS
# ============================================================================

# Update intervals (in seconds)
SENTIMENT_UPDATE_INTERVAL = 300  # 5 minutes
VEGAS_UPDATE_INTERVAL = 600      # 10 minutes
NEWS_CHECK_INTERVAL = 180        # 3 minutes

# Sentiment analysis
SENTIMENT_KEYWORDS = [
    'start', 'sit', 'play', 'avoid', 'chalk', 'fade',
    'smash', 'love', 'hate', 'lock', 'bust', 'boom'
]

SENTIMENT_WEIGHT = 0.2  # How much sentiment impacts ownership prediction

# Vegas monitoring
VEGAS_LINE_MOVE_THRESHOLD = 1.5  # Points - significant line movement
VEGAS_TOTAL_MOVE_THRESHOLD = 3.0  # Points - significant total movement

# ============================================================================
# DATA VALIDATION
# ============================================================================

REQUIRED_COLUMNS = [
    'name', 'team', 'position', 'salary', 
    'projection', 'ownership', 'ceiling', 'floor'
]

VALID_POSITIONS = ['QB', 'RB', 'WR', 'TE']

SALARY_RANGE = {
    'min': 1000,
    'max': 15000
}

PROJECTION_RANGE = {
    'min': 0,
    'max': 60
}

# ============================================================================
# API KEYS (Load from Streamlit secrets or .env file)
# ============================================================================

# Claude API (Phase 1.5) - Try Streamlit secrets first, then .env
ANTHROPIC_API_KEY = "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"

try:
    import streamlit as st
    if hasattr(st, 'secrets') and "ANTHROPIC_API_KEY" in st.secrets:
        ANTHROPIC_API_KEY = str(st.secrets["ANTHROPIC_API_KEY"]).strip().strip('"').strip("'")
except:
    pass

# Fallback to .env if not in Streamlit secrets
if not ANTHROPIC_API_KEY:
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '').strip().strip('"').strip("'")

# The Odds API (Vegas lines)
ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')

# Twitter API (Sentiment analysis)
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET', '')

# Reddit API (Sentiment analysis)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'DFS-Meta-Optimizer')

# ============================================================================
# UI SETTINGS
# ============================================================================

STREAMLIT_CONFIG = {
    'page_title': 'DFS Meta-Optimizer',
    'page_icon': '🎯',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color scheme for visualizations
COLORS = {
    'chalk': '#FF6B6B',      # High ownership - Red
    'balanced': '#4ECDC4',   # Medium ownership - Teal
    'contrarian': '#95E1D3', # Low ownership - Light green
    'leverage': '#F38181',   # High leverage - Coral
    'primary': '#6C5CE7',    # Primary accent
    'background': '#2D3436'  # Dark background
}

# ============================================================================
# FILE PATHS
# ============================================================================

DATA_DIR = 'data'
MODELS_DIR = 'models'
EXPORTS_DIR = 'exports'

SAMPLE_DATA_PATH = f'{DATA_DIR}/sample_players.csv'
HISTORICAL_OWNERSHIP_PATH = f'{DATA_DIR}/historical_ownership.csv'

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# FEATURE FLAGS
# ============================================================================

ENABLE_CLAUDE_AI = True  # Phase 1.5 - AI Assistant
ENABLE_SENTIMENT_ANALYSIS = False  # Enable in Phase 2
ENABLE_VEGAS_INTEGRATION = False   # Enable in Phase 2
ENABLE_ML_OWNERSHIP_PREDICTION = False  # Enable in Phase 3
ENABLE_BACKTESTING = False  # Enable in Phase 3

# ============================================================================
# CLAUDE API SETTINGS (Phase 1.5)
# ============================================================================

CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 2000
CLAUDE_TEMPERATURE = 0.7  # Balanced creativity/consistency

# Token usage limits (to manage costs)
CLAUDE_DAILY_REQUEST_LIMIT = 500  # Max requests per day
CLAUDE_COST_PER_REQUEST_ESTIMATE = 0.006  # USD

# AI feature toggles
AI_OWNERSHIP_PREDICTION = True
AI_NEWS_ANALYSIS = True
AI_STRATEGIC_ADVICE = True
AI_LINEUP_SCORING = True
