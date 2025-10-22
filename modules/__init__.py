"""Core modules for DFS Meta-Optimizer"""
from .opponent_modeling import OpponentModel
from .optimization_engine import LineupOptimizer

try:
    from .claude_assistant import ClaudeAssistant
except ImportError:
    ClaudeAssistant = None
