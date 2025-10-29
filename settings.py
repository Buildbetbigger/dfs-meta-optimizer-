"""
Advanced Configuration System for DFS Meta-Optimizer
Version 6.0.0 - MOST ADVANCED STATE

Revolutionary features:
- Dataclass-based structured configuration
- Dynamic hot-reload without restart
- Contest-specific auto-detection and presets
- Historical performance tracking for modes
- Configuration profiles (save/load custom configs)
- Advanced cross-field validation
- Environment-specific configs (dev/prod)
- Configuration versioning and migration
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "dev"
    PRODUCTION = "prod"
    TESTING = "test"

CURRENT_ENV = Environment(os.getenv('DFS_ENV', 'prod'))

# ==============================================================================
# CONTEST TYPE DETECTION
# ==============================================================================

class ContestType(Enum):
    """DFS contest types with auto-detection"""
    GPP_LARGE = "gpp_large"      # 100K+ entries
    GPP_MEDIUM = "gpp_medium"    # 10K-100K entries
    GPP_SMALL = "gpp_small"      # <10K entries
    DOUBLE_UP = "double_up"      # 50/50, Double Ups
    CASH_GAME = "cash"           # Head to head, 50/50
    SATELLITE = "satellite"      # Qualifier tournaments
    SINGLE_ENTRY = "single"      # Single entry GPP
    THREE_MAX = "three_max"      # 3-max entry
    SHOWDOWN = "showdown"        # Captain mode

def detect_contest_type(entry_fee: float, max_entries: int, field_size: int) -> ContestType:
    """
    Auto-detect contest type from parameters
    
    Args:
        entry_fee: Entry fee amount
        max_entries: Max entries per user
        field_size: Total field size
        
    Returns:
        Detected ContestType
    """
    # Single entry detection
    if max_entries == 1:
        return ContestType.SINGLE_ENTRY
    
    # 3-max detection
    if max_entries == 3:
        return ContestType.THREE_MAX
    
    # Cash game detection (typically smaller fields)
    if field_size < 100:
        return ContestType.CASH_GAME
    
    # Double up detection (50% payout)
    if field_size < 1000 and entry_fee < 50:
        return ContestType.DOUBLE_UP
    
    # GPP size detection
    if field_size >= 100000:
        return ContestType.GPP_LARGE
    elif field_size >= 10000:
        return ContestType.GPP_MEDIUM
    else:
        return ContestType.GPP_SMALL

# ==============================================================================
# OPTIMIZATION MODE CONFIGURATION
# ==============================================================================

@dataclass
class OptimizationMode:
    """
    Structured optimization mode configuration
    """
    name: str
    description: str
    
    # Scoring weights
    projection_weight: float
    ceiling_weight: float
    leverage_weight: float
    ownership_penalty: float
    floor_weight: float
    correlation_weight: float
    
    # Algorithm parameters
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float = 0.7
    elite_size: int = 20
    
    # Contest suitability
    best_for_contests: List[ContestType] = field(default_factory=list)
    
    # Performance tracking
    historical_roi: Optional[float] = None
    times_used: int = 0
    avg_score: Optional[float] = None
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = asdict(self)
        # Handle enum serialization
        result['best_for_contests'] = [c.value for c in self.best_for_contests]
        if self.last_updated:
            result['last_updated'] = self.last_updated.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationMode':
        """Create from dictionary"""
        # Handle enum deserialization
        if 'best_for_contests' in data:
            data['best_for_contests'] = [
                ContestType(c) for c in data['best_for_contests']
            ]
        if 'last_updated' in data and data['last_updated']:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

# ==============================================================================
# PRESET OPTIMIZATION MODES
# ==============================================================================

OPTIMIZATION_MODES = {
    'conservative': OptimizationMode(
        name='conservative',
        description='Safe cash game builds - high floor, consistent scoring',
        projection_weight=1.5,
        ceiling_weight=0.3,
        leverage_weight=0.1,
        ownership_penalty=0.05,
        floor_weight=0.8,
        correlation_weight=0.2,
        population_size=50,
        generations=30,
        mutation_rate=0.15,
        best_for_contests=[ContestType.CASH_GAME, ContestType.DOUBLE_UP]
    ),
    
    'balanced': OptimizationMode(
        name='balanced',
        description='Balanced GPP approach - mix of safety and upside',
        projection_weight=1.0,
        ceiling_weight=0.7,
        leverage_weight=0.6,
        ownership_penalty=0.15,
        floor_weight=0.4,
        correlation_weight=0.4,
        population_size=100,
        generations=50,
        mutation_rate=0.20,
        best_for_contests=[ContestType.GPP_SMALL, ContestType.GPP_MEDIUM]
    ),
    
    'aggressive': OptimizationMode(
        name='aggressive',
        description='High-risk GPP builds - maximum ceiling and leverage',
        projection_weight=0.5,
        ceiling_weight=1.5,
        leverage_weight=1.2,
        ownership_penalty=0.3,
        floor_weight=0.1,
        correlation_weight=0.5,
        population_size=150,
        generations=75,
        mutation_rate=0.25,
        best_for_contests=[ContestType.GPP_LARGE, ContestType.GPP_MEDIUM]
    ),
    
    'contrarian': OptimizationMode(
        name='contrarian',
        description='Anti-chalk strategy with low-owned plays',
        projection_weight=0.6,
        ceiling_weight=1.0,
        leverage_weight=1.0,
        ownership_penalty=0.5,
        floor_weight=0.2,
        correlation_weight=0.3,
        population_size=100,
        generations=60,
        mutation_rate=0.22,
        best_for_contests=[ContestType.GPP_LARGE, ContestType.SINGLE_ENTRY]
    ),
    
    'leverage': OptimizationMode(
        name='leverage',
        description='Leverage-first approach - beat the field',
        projection_weight=0.7,
        ceiling_weight=1.0,
        leverage_weight=1.5,
        ownership_penalty=0.2,
        floor_weight=0.3,
        correlation_weight=0.4,
        population_size=100,
        generations=50,
        mutation_rate=0.20,
        best_for_contests=[ContestType.GPP_MEDIUM, ContestType.THREE_MAX]
    ),
    
    'cash': OptimizationMode(
        name='cash',
        description='Cash game optimized - prioritize floor and safety',
        projection_weight=1.8,
        ceiling_weight=0.2,
        leverage_weight=0.05,
        ownership_penalty=0.03,
        floor_weight=1.0,
        correlation_weight=0.3,
        population_size=50,
        generations=30,
        mutation_rate=0.12,
        best_for_contests=[ContestType.CASH_GAME, ContestType.DOUBLE_UP]
    ),
    
    'single_entry': OptimizationMode(
        name='single_entry',
        description='Single-entry GPP - pure leverage play',
        projection_weight=0.4,
        ceiling_weight=1.3,
        leverage_weight=1.8,
        ownership_penalty=0.6,
        floor_weight=0.1,
        correlation_weight=0.6,
        population_size=200,
        generations=100,
        mutation_rate=0.30,
        best_for_contests=[ContestType.SINGLE_ENTRY]
    ),
    
    'showdown': OptimizationMode(
        name='showdown',
        description='Showdown captain mode optimization',
        projection_weight=1.0,
        ceiling_weight=1.2,
        leverage_weight=0.8,
        ownership_penalty=0.2,
        floor_weight=0.3,
        correlation_weight=0.9,  # High correlation for game stacks
        population_size=150,
        generations=80,
        mutation_rate=0.25,
        best_for_contests=[ContestType.SHOWDOWN]
    )
}

# ==============================================================================
# MAIN CONFIGURATION CLASS
# ==============================================================================

@dataclass
class DFSConfig:
    """
    Main configuration class with hot-reload support
    """
    
    # Core DFS constraints
    salary_cap: int = 50000
    roster_size: int = 9
    min_salary_pct: float = 0.96
    
    # Position requirements
    position_requirements: Dict[str, int] = field(default_factory=lambda: {
        'QB': 1,
        'RB': 2,
        'WR': 3,
        'TE': 1,
        'FLEX': 1,
        'DST': 1
    })
    
    # Claude AI settings
    enable_claude_ai: bool = True
    ai_ownership_prediction: bool = True
    ai_strategic_analysis: bool = True
    ai_news_analysis: bool = False
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 4000
    claude_temperature: float = 0.7
    
    # API optimization
    enable_prompt_caching: bool = True  # 90% cost reduction
    enable_response_caching: bool = True  # Cache predictions 15min
    enable_batch_predictions: bool = True  # Batch API calls
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Cost tracking
    estimated_cost_per_request: float = 0.006
    daily_budget_limit: Optional[float] = None  # Set to limit spending
    
    # Opponent modeling
    high_ownership_threshold: float = 30.0
    low_ownership_threshold: float = 10.0
    min_leverage_threshold: float = 1.5
    high_leverage_threshold: float = 3.0
    
    # Contest-specific adjustments
    contest_size_multiplier: Dict[ContestType, float] = field(default_factory=lambda: {
        ContestType.GPP_LARGE: 1.5,    # Increase leverage importance
        ContestType.GPP_MEDIUM: 1.2,
        ContestType.GPP_SMALL: 1.0,
        ContestType.CASH_GAME: 0.5,    # Decrease leverage importance
        ContestType.DOUBLE_UP: 0.6,
        ContestType.SINGLE_ENTRY: 2.0,  # Maximum leverage
        ContestType.THREE_MAX: 1.3
    })
    
    # Portfolio settings
    max_player_exposure: float = 0.50
    min_player_difference: int = 2
    default_lineup_count: int = 20
    max_lineup_count: int = 150
    enforce_uniqueness: bool = True
    min_unique_players: float = 0.60
    
    # Performance optimization
    max_generation_time: int = 300
    enable_caching: bool = True
    cache_size: int = 1000
    enable_multiprocessing: bool = False
    max_workers: int = 4
    enable_gpu_acceleration: bool = False  # For future ML models
    
    # Validation ranges
    min_projection: float = 0.0
    max_projection: float = 100.0
    min_ownership: float = 0.0
    max_ownership: float = 100.0
    min_salary: int = 3000
    max_salary: int = 12000
    
    # Feature flags
    enable_module_2: bool = True   # Genetic + stacking
    enable_module_3: bool = True   # Portfolio optimization
    enable_module_4: bool = False  # Real-time data
    enable_module_5: bool = False  # Monte Carlo
    enable_variance_analysis: bool = True
    enable_exposure_management: bool = True
    enable_correlation_optimization: bool = True
    
    # Advanced features
    enable_ownership_correlation: bool = True
    enable_leverage_decay: bool = True
    enable_adaptive_diversity: bool = True
    enable_lineup_explanation: bool = True
    enable_historical_learning: bool = True
    
    # Logging
    debug_mode: bool = False
    verbose_logging: bool = False
    log_file: str = "dfs_optimizer.log"
    log_api_calls: bool = True
    
    # Configuration metadata
    config_version: str = "6.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    profile_name: str = "default"
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Load API key from environment if not set
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
            
            # Try Streamlit secrets
            if not self.anthropic_api_key:
                try:
                    import streamlit as st
                    if hasattr(st, 'secrets'):
                        self.anthropic_api_key = st.secrets.get('ANTHROPIC_API_KEY', '')
                except ImportError:
                    pass
    
    def validate(self) -> List[str]:
        """
        Comprehensive validation with cross-field checks
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # API key validation
        if self.enable_claude_ai:
            if not self.anthropic_api_key:
                errors.append("ANTHROPIC_API_KEY not set but AI is enabled")
            elif not self.anthropic_api_key.startswith('sk-ant-'):
                errors.append("ANTHROPIC_API_KEY has invalid format")
            elif len(self.anthropic_api_key) < 50:
                errors.append(f"ANTHROPIC_API_KEY too short ({len(self.anthropic_api_key)} chars)")
        
        # Salary constraints
        if self.min_salary_pct < 0.90 or self.min_salary_pct > 1.0:
            errors.append(f"min_salary_pct should be 0.90-1.0, got {self.min_salary_pct}")
        
        if self.salary_cap <= 0:
            errors.append(f"salary_cap must be positive, got {self.salary_cap}")
        
        # Roster size
        if self.roster_size < 1:
            errors.append(f"roster_size must be positive, got {self.roster_size}")
        
        # Position requirements validation
        total_positions = sum(self.position_requirements.values())
        if total_positions != self.roster_size:
            errors.append(f"Position requirements ({total_positions}) don't match roster_size ({self.roster_size})")
        
        # Threshold validation
        if self.low_ownership_threshold >= self.high_ownership_threshold:
            errors.append("low_ownership_threshold must be < high_ownership_threshold")
        
        # Exposure validation
        if self.max_player_exposure < 0 or self.max_player_exposure > 1:
            errors.append(f"max_player_exposure should be 0-1, got {self.max_player_exposure}")
        
        if self.min_unique_players < 0 or self.min_unique_players > 1:
            errors.append(f"min_unique_players should be 0-1, got {self.min_unique_players}")
        
        # Lineup count validation
        if self.default_lineup_count > self.max_lineup_count:
            errors.append("default_lineup_count cannot exceed max_lineup_count")
        
        # Budget validation
        if self.daily_budget_limit is not None and self.daily_budget_limit < 0:
            errors.append("daily_budget_limit cannot be negative")
        
        return errors
    
    def get_required_salary(self) -> int:
        """Calculate minimum required salary"""
        return int(self.salary_cap * self.min_salary_pct)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        
        # Handle enum serialization
        if 'contest_size_multiplier' in result:
            result['contest_size_multiplier'] = {
                k.value: v for k, v in self.contest_size_multiplier.items()
            }
        
        # Handle datetime serialization
        result['created_at'] = self.created_at.isoformat()
        result['last_modified'] = self.last_modified.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DFSConfig':
        """Create from dictionary"""
        # Handle enum deserialization
        if 'contest_size_multiplier' in data:
            data['contest_size_multiplier'] = {
                ContestType(k): v for k, v in data['contest_size_multiplier'].items()
            }
        
        # Handle datetime deserialization
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_modified' in data and isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        
        return cls(**data)
    
    def save_profile(self, filepath: Optional[str] = None):
        """
        Save configuration profile to file
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            filepath = f"config_profiles/{self.profile_name}.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved configuration profile to {filepath}")
    
    @classmethod
    def load_profile(cls, profile_name: str) -> 'DFSConfig':
        """
        Load configuration profile from file
        
        Args:
            profile_name: Name of profile to load
            
        Returns:
            DFSConfig instance
        """
        filepath = f"config_profiles/{profile_name}.json"
        
        if not Path(filepath).exists():
            logger.warning(f"Profile {profile_name} not found, using default")
            return cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded configuration profile from {filepath}")
        return cls.from_dict(data)

# ==============================================================================
# CONFIGURATION MANAGER
# ==============================================================================

class ConfigManager:
    """
    Manages configuration with hot-reload and performance tracking
    """
    
    def __init__(self, config: Optional[DFSConfig] = None):
        """Initialize with optional config"""
        self.config = config or DFSConfig()
        self.performance_history: Dict[str, List[float]] = {}
        self.mode_usage_count: Dict[str, int] = {}
        
        # Load performance history if exists
        self._load_performance_history()
    
    def get_optimal_mode_for_contest(self, contest_type: ContestType) -> OptimizationMode:
        """
        Get optimal optimization mode for contest type
        
        Args:
            contest_type: Type of contest
            
        Returns:
            Best optimization mode
        """
        # Filter modes suitable for this contest
        suitable_modes = [
            mode for mode in OPTIMIZATION_MODES.values()
            if contest_type in mode.best_for_contests
        ]
        
        if not suitable_modes:
            # Fallback to balanced
            return OPTIMIZATION_MODES['balanced']
        
        # If we have historical performance data, use it
        if self.performance_history:
            best_mode = max(
                suitable_modes,
                key=lambda m: m.historical_roi or 0
            )
            return best_mode
        
        # Otherwise return first suitable mode
        return suitable_modes[0]
    
    def record_mode_performance(self, mode_name: str, roi: float, score: float):
        """
        Record performance for a mode
        
        Args:
            mode_name: Name of optimization mode
            roi: Return on investment
            score: Lineup score
        """
        if mode_name not in OPTIMIZATION_MODES:
            return
        
        mode = OPTIMIZATION_MODES[mode_name]
        
        # Update usage count
        mode.times_used += 1
        
        # Update ROI (exponential moving average)
        if mode.historical_roi is None:
            mode.historical_roi = roi
        else:
            mode.historical_roi = 0.7 * mode.historical_roi + 0.3 * roi
        
        # Update average score
        if mode.avg_score is None:
            mode.avg_score = score
        else:
            mode.avg_score = 0.7 * mode.avg_score + 0.3 * score
        
        mode.last_updated = datetime.now()
        
        # Save to history
        if mode_name not in self.performance_history:
            self.performance_history[mode_name] = []
        self.performance_history[mode_name].append(roi)
        
        # Save history to disk
        self._save_performance_history()
        
        logger.info(f"Recorded performance for {mode_name}: ROI={roi:.2f}, Score={score:.2f}")
    
    def _load_performance_history(self):
        """Load performance history from disk"""
        history_file = "config_profiles/performance_history.json"
        
        if not Path(history_file).exists():
            return
        
        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
            
            self.performance_history = data.get('history', {})
            
            # Update modes with historical data
            for mode_name, history in self.performance_history.items():
                if mode_name in OPTIMIZATION_MODES and history:
                    mode = OPTIMIZATION_MODES[mode_name]
                    mode.historical_roi = sum(history[-10:]) / min(len(history), 10)
                    mode.times_used = len(history)
            
            logger.info("Loaded performance history")
        except Exception as e:
            logger.error(f"Failed to load performance history: {e}")
    
    def _save_performance_history(self):
        """Save performance history to disk"""
        history_file = "config_profiles/performance_history.json"
        Path(history_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                'history': self.performance_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved performance history")
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
    
    def get_mode_rankings(self) -> List[Tuple[str, float]]:
        """
        Get modes ranked by historical performance
        
        Returns:
            List of (mode_name, roi) tuples sorted by ROI
        """
        rankings = []
        for name, mode in OPTIMIZATION_MODES.items():
            if mode.historical_roi is not None:
                rankings.append((name, mode.historical_roi))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def hot_reload(self, new_config: DFSConfig) -> bool:
        """
        Hot reload configuration without restart
        
        Args:
            new_config: New configuration
            
        Returns:
            True if reload successful
        """
        # Validate new config
        errors = new_config.validate()
        if errors:
            logger.error(f"Config validation failed: {errors}")
            return False
        
        # Swap config
        old_config = self.config
        self.config = new_config
        self.config.last_modified = datetime.now()
        
        logger.info(f"Hot reloaded configuration (v{new_config.config_version})")
        return True

# ==============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ==============================================================================

# Create global config manager
_config_manager = ConfigManager()

def get_config() -> DFSConfig:
    """Get current configuration"""
    return _config_manager.config

def get_config_manager() -> ConfigManager:
    """Get configuration manager"""
    return _config_manager

def get_optimization_mode(mode_name: str) -> OptimizationMode:
    """Get optimization mode by name"""
    return OPTIMIZATION_MODES.get(mode_name, OPTIMIZATION_MODES['balanced'])

def validate_api_key() -> bool:
    """Check if API key is valid"""
    config = get_config()
    if not config.anthropic_api_key:
        return False
    return config.anthropic_api_key.startswith('sk-ant-') and len(config.anthropic_api_key) >= 50

def print_config_summary():
    """Print configuration summary"""
    config = get_config()
    
    print("=" * 70)
    print("DFS META-OPTIMIZER CONFIGURATION v6.0.0")
    print("=" * 70)
    print(f"Environment: {CURRENT_ENV.value.upper()}")
    print(f"Profile: {config.profile_name}")
    print(f"Salary Cap: ${config.salary_cap:,}")
    print(f"Roster Size: {config.roster_size}")
    print(f"Min Salary Usage: {config.min_salary_pct*100:.0f}% (${config.get_required_salary():,})")
    print()
    print("Optimization Modes Available:")
    for mode in OPTIMIZATION_MODES.values():
        status = ""
        if mode.historical_roi:
            status = f" (ROI: {mode.historical_roi:.1f}x, Used: {mode.times_used}x)"
        print(f"  • {mode.name}: {mode.description}{status}")
    print()
    print(f"Claude AI: {'✅ Enabled' if config.enable_claude_ai and validate_api_key() else '❌ Disabled'}")
    print(f"  - Prompt Caching: {'✅' if config.enable_prompt_caching else '❌'}")
    print(f"  - Response Caching: {'✅' if config.enable_response_caching else '❌'}")
    print(f"  - Batch Predictions: {'✅' if config.enable_batch_predictions else '❌'}")
    print()
    print(f"Advanced Features:")
    print(f"  - Ownership Correlation: {'✅' if config.enable_ownership_correlation else '❌'}")
    print(f"  - Leverage Decay: {'✅' if config.enable_leverage_decay else '❌'}")
    print(f"  - Adaptive Diversity: {'✅' if config.enable_adaptive_diversity else '❌'}")
    print(f"  - Lineup Explanation: {'✅' if config.enable_lineup_explanation else '❌'}")
    print(f"  - Historical Learning: {'✅' if config.enable_historical_learning else '❌'}")
    print()
    print(f"Modules:")
    print(f"  - Module 2 (Genetic/Stacking): {'✅' if config.enable_module_2 else '❌'}")
    print(f"  - Module 3 (Portfolio): {'✅' if config.enable_module_3 else '❌'}")
    print(f"  - Module 4 (Real-time Data): {'✅' if config.enable_module_4 else '❌'}")
    print(f"  - Module 5 (Monte Carlo): {'✅' if config.enable_module_5 else '❌'}")
    print()
    print(f"Debug Mode: {'✅' if config.debug_mode else '❌'}")
    print("=" * 70)

# ==============================================================================
# COMPATIBILITY ALIASES (for import flexibility)
# ==============================================================================

# Alias for alternative import names
Settings = DFSConfig

def get_settings() -> DFSConfig:
    """Get current settings (alias for get_config)"""
    return get_config()

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'DFSConfig',
    'Settings',  # Alias for DFSConfig
    'OptimizationMode',
    'ContestType',
    'ConfigManager',
    'OPTIMIZATION_MODES',
    'get_config',
    'get_settings',  # Alias for get_config
    'get_config_manager',
    'get_optimization_mode',
    'validate_api_key',
    'detect_contest_type',
    'print_config_summary',
]
