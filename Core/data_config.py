"""
DFS Meta-Optimizer - Data Configuration v6.3.0

Configuration for real-time data integration modules:
- News Feed Monitor
- Vegas Lines Tracker  
- Ownership Tracker
"""

from typing import Dict


# ============================================================================
# NEWS FEED CONFIGURATION
# ============================================================================

NEWS_CONFIG = {
    # News source priorities (higher = more trusted)
    'source_priorities': {
        'rotoworld': 90,
        'fantasypros': 85,
        'nfl_official': 95,
        'beat_reporter': 80,
        'fantasy_expert': 75,
        'manual': 50
    },
    
    # Impact thresholds for alerts
    'alert_thresholds': {
        'critical': 70.0,  # Impact score >= 70 triggers critical alert
        'high': 50.0,
        'medium': 30.0
    },
    
    # Projection adjustment limits
    'projection_adjustments': {
        'max_increase': 0.20,  # Max 20% increase
        'max_decrease': 0.30,  # Max 30% decrease
        'default_factor': 0.15  # Default adjustment factor
    },
    
    # News retention
    'retention_hours': 168,  # Keep news for 1 week
    
    # Auto-refresh settings
    'auto_fetch_interval': 30  # Minutes between auto-fetches
}


# ============================================================================
# VEGAS LINES CONFIGURATION
# ============================================================================

VEGAS_CONFIG = {
    # Line movement thresholds
    'movement_thresholds': {
        'spread': {
            'minor': 0.5,      # 0.5 point = minor movement
            'moderate': 1.0,    # 1 point = moderate
            'major': 2.0        # 2+ points = major
        },
        'total': {
            'minor': 1.0,
            'moderate': 2.0,
            'major': 3.0
        },
        'ml': {
            'minor': 10,        # Money line changes
            'moderate': 25,
            'major': 50
        }
    },
    
    # Sharp money detection
    'sharp_indicators': {
        'min_spread_movement': 1.0,      # 1+ point movement
        'min_total_movement': 2.0,       # 2+ point movement
        'reverse_line_threshold': 0.5    # Line moved opposite direction
    },
    
    # Implied total calculation
    'implied_total_boost': {
        'high_total': 28.0,      # Team totals >= 28 = elite
        'medium_total': 24.0,    # 24-28 = good
        'low_total': 17.0        # <= 17 = poor
    },
    
    # Data retention
    'retention_hours': 168  # Keep lines for 1 week
}


# ============================================================================
# OWNERSHIP PREDICTION CONFIGURATION
# ============================================================================

OWNERSHIP_CONFIG = {
    # Base ownership by value tiers
    'value_tiers': {
        'elite': {'min': 4.0, 'base_ownership': 30.0},
        'great': {'min': 3.5, 'base_ownership': 20.0},
        'good': {'min': 3.0, 'base_ownership': 12.0},
        'decent': {'min': 2.5, 'base_ownership': 7.0},
        'poor': {'min': 0.0, 'base_ownership': 3.0}
    },
    
    # Salary tier multipliers
    'salary_multipliers': {
        'stud': {'min': 9000, 'mult': 1.3},
        'high': {'min': 7000, 'mult': 1.0},
        'mid': {'min': 5000, 'mult': 0.8},
        'value': {'min': 0, 'mult': 0.6}
    },
    
    # Vegas total impact on ownership
    'vegas_multipliers': {
        'high_scoring': {'min': 28.0, 'mult': 1.4},
        'moderate': {'min': 24.0, 'mult': 1.2},
        'average': {'min': 20.0, 'mult': 1.0},
        'low_scoring': {'min': 0.0, 'mult': 0.7}
    },
    
    # Injury status impact
    'injury_multipliers': {
        'ACTIVE': 1.0,
        'PROBABLE': 0.9,
        'QUESTIONABLE': 0.6,
        'DOUBTFUL': 0.2,
        'OUT': 0.0
    },
    
    # Recent performance (recency bias)
    'performance_multipliers': {
        'hot_streak': {'min_ratio': 1.3, 'mult': 1.3},
        'trending_up': {'min_ratio': 1.1, 'mult': 1.15},
        'average': {'min_ratio': 0.9, 'mult': 1.0},
        'trending_down': {'min_ratio': 0.7, 'mult': 0.8}
    },
    
    # Contest type adjustments
    'contest_adjustments': {
        'GPP': 1.0,
        'CASH': 1.5,  # Value plays more owned in cash
        'TOURNAMENT': 0.9,
        'SATELLITE': 1.2
    },
    
    # Ownership thresholds
    'thresholds': {
        'chalk': 25.0,       # 25%+ = chalk
        'popular': 15.0,     # 15-25% = popular
        'medium': 8.0,       # 8-15% = medium
        'low': 0.0           # < 8% = contrarian
    },
    
    # Prediction accuracy tracking
    'accuracy_targets': {
        'excellent': 3.0,    # MAE <= 3%
        'good': 5.0,         # MAE <= 5%
        'acceptable': 8.0,   # MAE <= 8%
        'poor': 999.9        # MAE > 8%
    }
}


# ============================================================================
# DATA REFRESH CONFIGURATION
# ============================================================================

REFRESH_CONFIG = {
    # Automatic refresh settings
    'auto_refresh': {
        'enabled': False,              # Auto-refresh disabled by default
        'interval_minutes': 30,        # Refresh every 30 minutes
        'max_refreshes': 10            # Max number of auto-refreshes
    },
    
    # What to update on refresh
    'update_settings': {
        'update_projections': True,     # Update projections from news
        'update_ownership': True,       # Update ownership predictions
        'update_vegas': True            # Update Vegas line impacts
    },
    
    # Alert settings
    'alerts': {
        'critical_news_hours': 2,       # Alert on critical news from last 2 hours
        'line_movement_hours': 2,       # Alert on line movements from last 2 hours
        'min_ownership_shift': 10.0,    # Alert on 10%+ ownership changes
        'min_projection_change': 2.0    # Alert on 2+ point projection changes
    },
    
    # Performance settings
    'performance': {
        'batch_size': 50,               # Process players in batches of 50
        'timeout_seconds': 30,          # Timeout for external API calls
        'retry_attempts': 3             # Retry failed requests 3 times
    },
    
    # Data export
    'export': {
        'auto_export': False,           # Auto-export updated data
        'export_path': '/mnt/user-data/outputs/',
        'export_format': 'csv'          # 'csv' or 'json'
    }
}


# ============================================================================
# CONTEST-SPECIFIC PRESETS
# ============================================================================

CONTEST_PRESETS = {
    'MILLY_MAKER': {
        'ownership_threshold_chalk': 30.0,
        'ownership_threshold_leverage': 12.0,
        'projection_adjustment_factor': 0.20,
        'vegas_weight': 1.4,
        'news_weight': 1.3
    },
    
    'GPP_LARGE': {
        'ownership_threshold_chalk': 25.0,
        'ownership_threshold_leverage': 15.0,
        'projection_adjustment_factor': 0.15,
        'vegas_weight': 1.2,
        'news_weight': 1.1
    },
    
    'GPP_SMALL': {
        'ownership_threshold_chalk': 20.0,
        'ownership_threshold_leverage': 18.0,
        'projection_adjustment_factor': 0.12,
        'vegas_weight': 1.1,
        'news_weight': 1.0
    },
    
    'CASH_GAME': {
        'ownership_threshold_chalk': 35.0,
        'ownership_threshold_leverage': 8.0,
        'projection_adjustment_factor': 0.10,
        'vegas_weight': 1.5,
        'news_weight': 1.4
    },
    
    'SHOWDOWN': {
        'ownership_threshold_chalk': 40.0,
        'ownership_threshold_leverage': 10.0,
        'projection_adjustment_factor': 0.18,
        'vegas_weight': 1.6,
        'news_weight': 1.5
    }
}


# ============================================================================
# INTEGRATION SETTINGS
# ============================================================================

INTEGRATION_CONFIG = {
    # Which modules to use
    'modules_enabled': {
        'news_monitor': True,
        'vegas_tracker': True,
        'ownership_tracker': True
    },
    
    # External API settings (for future expansion)
    'external_apis': {
        'rotoworld': {
            'enabled': False,
            'api_key': None,
            'base_url': 'https://api.rotoworld.com'
        },
        'odds_api': {
            'enabled': False,
            'api_key': None,
            'base_url': 'https://api.the-odds-api.com'
        }
    },
    
    # Logging
    'logging': {
        'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        'log_file': None,  # None = console only
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_contest_preset(contest_type: str) -> Dict:
    """
    Get preset configuration for contest type.
    
    Args:
        contest_type: Contest type identifier
    
    Returns:
        Configuration dict
    """
    return CONTEST_PRESETS.get(contest_type, CONTEST_PRESETS['GPP_LARGE'])


def get_ownership_threshold(tier: str) -> float:
    """
    Get ownership threshold for tier.
    
    Args:
        tier: 'chalk', 'popular', 'medium', or 'low'
    
    Returns:
        Ownership percentage threshold
    """
    return OWNERSHIP_CONFIG['thresholds'].get(tier, 0.0)


def get_value_tier_ownership(value: float) -> float:
    """
    Get base ownership for value tier.
    
    Args:
        value: Points per $1000 salary
    
    Returns:
        Base ownership percentage
    """
    for tier_name, tier_config in OWNERSHIP_CONFIG['value_tiers'].items():
        if value >= tier_config['min']:
            return tier_config['base_ownership']
    
    return 3.0  # Default for poor value
