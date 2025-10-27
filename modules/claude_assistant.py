"""
Claude AI Assistant Module

Uses Claude API to provide intelligent analysis:
- Ownership prediction
- News impact analysis
- Strategic recommendations
- Lineup quality scoring
"""

import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")

from config.settings import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE
)


class ClaudeAssistant:
    """
    AI-powered DFS analysis using Claude API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Claude assistant
        
        Args:
            api_key: Anthropic API key (defaults to config)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        # Get and thoroughly clean API key
        raw_key = api_key or ANTHROPIC_API_KEY
        
        if not raw_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in .env file or Streamlit secrets")
        
        # Clean the key - remove ALL whitespace, quotes, and newlines
        self.api_key = str(raw_key).strip()
        self.api_key = self.api_key.strip('"').strip("'")
        self.api_key = self.api_key.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Validate key format
        if len(self.api_key) < 50:
            raise ValueError(
                f"ANTHROPIC_API_KEY too short. Got {len(self.api_key)} chars, need 50+. "
                f"Check your secrets configuration."
            )
        
        if not self.api_key.startswith('sk-ant-'):
            raise ValueError(
                f"ANTHROPIC_API_KEY format invalid. Should start with 'sk-ant-'. "
                f"Yours starts with: {self.api_key[:10]}"
            )
        
        print(f"🔑 API Key validated: {self.api_key[:15]}...{self.api_key[-10:]} ({len(self.api_key)} chars)")
        
        try:
            # Create client with the cleaned key
            self.client = Anthropic(api_key=self.api_key)
            self.request_count = 0
            self.total_cost = 0.0
            print("✅ Claude API client initialized successfully")
            
        except Exception as e:
            raise ValueError(f"Failed to create Anthropic client: {str(e)}")
    
    def _call_claude(self, prompt: str, system: str = None) -> str:
        """
        Make a call to Claude API
        
        Args:
            prompt: User prompt
            system: Optional system prompt
        
        Returns:
            Claude's response text
        """
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "temperature": CLAUDE_TEMPERATURE,
            "messages": messages
        }
        
        if system:
            kwargs["system"] = system
        
        try:
            # Debug output
            print(f"🤖 Making Claude API call (Request #{self.request_count + 1})...")
            
            response = self.client.messages.create(**kwargs)
            
            # Track usage
            self.request_count += 1
            self.total_cost += 0.006  # Rough estimate
            
            print(f"✅ API call successful")
            return response.content[0].text
            
        except Exception as e:
            print(f"❌ Claude API call failed: {str(e)}")
            print(f"   API Key (first 15 chars): {self.api_key[:15]}...")
            print(f"   API Key (last 10 chars): ...{self.api_key[-10:]}")
            print(f"   Key length: {len(self.api_key)}")
            raise
    
    def predict_ownership(self, 
                         player_data: Dict,
                         context: Dict = None) -> Dict:
        """
        Use Claude to predict player ownership percentage
        
        Args:
            player_data: Dictionary with player information
            context: Additional context (news, vegas, recent performance)
        
        Returns:
            Dictionary with ownership prediction and reasoning
        """
        context = context or {}
        
        system = """You are an expert DFS analyst specializing in ownership prediction.
You understand player psychology, recency bias, narrative angles, and market behavior.
IMPORTANT: Provide SPECIFIC, DIFFERENTIATED predictions based on each player's unique situation."""

        # Build detailed prompt
        prompt = f"""Predict the ownership percentage for this player in a DraftKings tournament:

**Player:** {player_data.get('name', 'Unknown')}
**Team:** {player_data.get('team', 'Unknown')}
**Position:** {player_data.get('position', 'Unknown')}
**Salary:** ${player_data.get('salary', 0):,}
**Projection:** {player_data.get('projection', 0):.1f} points
**Current Ownership Estimate:** {player_data.get('ownership', 0):.1f}%

**Additional Context:**
{json.dumps(context, indent=2)}

**Analysis Required:**
1. What is your ownership prediction? (Be specific - don't just say "moderate")
2. What factors make this player more/less popular than projection alone suggests?
3. Is there a narrative or recency bias affecting this player?
4. How does the salary position influence ownership?
5. Rate your confidence in this prediction (0-100%)

Respond ONLY in valid JSON format:
{{
    "ownership": <number 0-100>,
    "confidence": <number 0-100>,
    "reasoning": "<2-3 sentence explanation>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "leverage_opportunity": "<yes|no|maybe>"
}}"""

        try:
            response = self._call_claude(prompt, system)
            
            # Clean and parse JSON
            response_clean = response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1])
            
            result = json.loads(response_clean)
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"❌ Ownership prediction failed: {str(e)}")
            return {
                'ownership': player_data.get('ownership', 10.0),
                'confidence': 0,
                'reasoning': f'Error: {str(e)}',
                'key_factors': [],
                'leverage_opportunity': 'unknown',
                'error': str(e)
            }
    
    def analyze_news_impact(self, news: str, affected_players: List[str]) -> Dict:
        """
        Analyze how breaking news affects player values and ownership
        
        Args:
            news: News text to analyze
            affected_players: List of potentially affected player names
        
        Returns:
            Impact analysis with adjustments
        """
        system = """You are an expert DFS analyst who understands how news impacts
player value, projections, and ownership in real-time."""

        prompt = f"""Analyze this breaking news and its DFS impact:

**NEWS:** {news}

**POTENTIALLY AFFECTED PLAYERS:**
{json.dumps(affected_players, indent=2)}

**Analysis Required:**
1. Which specific players are impacted (positively or negatively)?
2. How should their projections change? (numerical adjustment)
3. How will public ownership react? (numerical adjustment)
4. What's the leverage opportunity here?
5. Any secondary effects or correlations to consider?

**Think through:**
- Direct impacts (player mentioned in news)
- Indirect impacts (teammates benefit, opponents affected)
- Public overreaction vs actual impact
- Leverage opportunities (ownership shifts more than projection)

Respond in ONLY valid JSON format:
{{
    "impacted_players": [
        {{
            "name": "<player name>",
            "impact_type": "<positive|negative|neutral>",
            "projection_change": <number, can be negative>,
            "ownership_change": <number, can be negative>,
            "new_leverage": "<increase|decrease|same>",
            "reasoning": "<brief explanation>"
        }}
    ],
    "overall_strategy": "<how this changes your approach>",
    "urgency": "<high|medium|low>",
    "key_takeaway": "<most important thing to know>",
    "leverage_opportunities": ["<player1>", "<player2>"]
}}"""

        try:
            response = self._call_claude(prompt, system)
            
            # Clean and parse JSON
            response_clean = response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1])
            
            result = json.loads(response_clean)
            result['timestamp'] = datetime.now().isoformat()
            result['news_analyzed'] = news[:100]
            
            return result
            
        except Exception as e:
            print(f"❌ News analysis failed: {str(e)}")
            return {
                'impacted_players': [],
                'overall_strategy': f'Error analyzing news: {str(e)}',
                'urgency': 'unknown',
                'key_takeaway': 'Unable to analyze',
                'leverage_opportunities': [],
                'error': str(e)
            }
    
    def get_strategic_advice(self,
                            field_analysis: Dict,
                            player_metrics: pd.DataFrame,
                            contest_info: Dict) -> Dict:
        """
        Get strategic recommendations based on field analysis
        
        Args:
            field_analysis: Field distribution metrics
            player_metrics: DataFrame with player leverage metrics
            contest_info: Contest type and details
        
        Returns:
            Strategic recommendations
        """
        system = """You are an elite DFS tournament strategist and game theorist.
You understand leverage, field dynamics, and optimal tournament strategy."""

        # Convert metrics to dict for prompt
        # Calculate leverage_score if it doesn't exist
        if 'leverage_score' not in player_metrics.columns:
            player_metrics = player_metrics.copy()
            if 'ceiling' in player_metrics.columns and 'ownership' in player_metrics.columns:
                # Leverage = ceiling / ownership (higher is better)
                player_metrics['leverage_score'] = player_metrics['ceiling'] / (player_metrics['ownership'] + 0.1)
            elif 'projection' in player_metrics.columns and 'ownership' in player_metrics.columns:
                # Fallback to projection / ownership
                player_metrics['leverage_score'] = player_metrics['projection'] / (player_metrics['ownership'] + 0.1)
            else:
                # Last resort: just use projection
                player_metrics['leverage_score'] = player_metrics.get('projection', 10.0)
        
        # Get top leverage and high owned players
        top_leverage = player_metrics.nlargest(5, 'leverage_score')[['name', 'leverage_score', 'ownership']].to_dict('records')
        high_owned = player_metrics.nlargest(5, 'ownership')[['name', 'ownership', 'projection']].to_dict('records')
        
        # Convert field_analysis to JSON-serializable format (handle numpy types)
        def make_serializable(obj):
            """Convert numpy/pandas types to native Python types"""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return obj
        
        field_analysis_clean = make_serializable(field_analysis)
        contest_info_clean = make_serializable(contest_info)
        top_leverage_clean = make_serializable(top_leverage)
        high_owned_clean = make_serializable(high_owned)

        prompt = f"""Given this field analysis and player metrics, provide strategic guidance:

**CONTEST INFO:**
{json.dumps(contest_info_clean, indent=2)}

**FIELD ANALYSIS:**
{json.dumps(field_analysis_clean, indent=2)}

**TOP LEVERAGE PLAYS:**
{json.dumps(top_leverage_clean, indent=2)}

**HIGHEST OWNED PLAYERS:**
{json.dumps(high_owned_clean, indent=2)}

**Strategic Questions:**
1. What should the contrarian threshold be for this contest?
2. Which chalk plays should I fade vs play through?
3. What's the optimal captain strategy?
4. Should I employ naked fades? If so, on whom?
5. How correlated should my builds be?
6. What's my overall lineup construction philosophy for this field?

Provide actionable strategic recommendations in ONLY valid JSON format:
{{
    "contrarian_threshold": <0-100>,
    "chalk_to_fade": ["<player1>", "<player2>"],
    "chalk_to_play": ["<player1>", "<player2>"],
    "leverage_targets": ["<player1>", "<player2>"],
    "captain_philosophy": "<brief description>",
    "correlation_advice": "<brief description>",
    "key_insight": "<most important strategic takeaway>",
    "optimal_mode": "<CASH|GPP|CONTRARIAN|BALANCED>",
    "confidence": <0-100>
}}"""

        try:
            response = self._call_claude(prompt, system)
            
            # Clean and parse JSON
            response_clean = response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1])
            
            result = json.loads(response_clean)
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"❌ Strategic advice failed: {str(e)}")
            return {
                'contrarian_threshold': 50,
                'chalk_to_fade': [],
                'chalk_to_play': [],
                'leverage_targets': [],
                'captain_philosophy': f'Error: {str(e)}',
                'correlation_advice': 'Unable to provide',
                'key_insight': 'Error getting advice',
                'optimal_mode': 'BALANCED',
                'confidence': 0,
                'error': str(e)
            }
    
    def score_lineup_quality(self, lineup: Dict, field_context: Dict) -> Dict:
        """
        Score a lineup's quality across multiple dimensions
        
        Args:
            lineup: Lineup dictionary with players
            field_context: Field ownership and correlation data
        
        Returns:
            Multi-dimensional quality scores
        """
        system = """You are an expert DFS lineup evaluator who understands
tournament equity, game theory, and winning lineup construction."""

        prompt = f"""Evaluate this DFS lineup across multiple dimensions:

**LINEUP:**
{json.dumps(lineup, indent=2)}

**FIELD CONTEXT:**
{json.dumps(field_context, indent=2)}

**Evaluate on these dimensions:**
1. **Tournament Equity (0-100)**: Probability of winning vs the field
2. **Correlation Quality (0-100)**: How well do players work together?
3. **Leverage vs Field (0-100)**: Differentiation value
4. **Uniqueness Value (0-100)**: Is being different helping or hurting?
5. **Game Theory Score (0-100)**: Strategic soundness

**Also Identify:**
- Biggest strength of this lineup
- Biggest weakness of this lineup
- One specific improvement suggestion
- Game scripts where this lineup thrives
- Game scripts where this lineup struggles

Provide scores and brief explanations for each dimension in ONLY valid JSON format:
{{
    "tournament_equity": <0-100>,
    "correlation_quality": <0-100>,
    "leverage_score": <0-100>,
    "uniqueness_value": <0-100>,
    "game_theory_score": <0-100>,
    "overall_grade": "<A+|A|A-|B+|B|B-|C+|C|C-|D|F>",
    "biggest_strength": "<brief description>",
    "biggest_weakness": "<brief description>",
    "improvement_suggestion": "<specific actionable advice>",
    "ideal_game_script": "<description>",
    "poor_game_script": "<description>"
}}"""

        try:
            response = self._call_claude(prompt, system)
            
            # Clean and parse JSON
            response_clean = response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1])
            
            result = json.loads(response_clean)
            result['timestamp'] = datetime.now().isoformat()
            result['lineup_id'] = lineup.get('lineup_id')
            
            return result
            
        except Exception as e:
            print(f"❌ Lineup scoring failed: {str(e)}")
            return {
                'tournament_equity': 0,
                'correlation_quality': 0,
                'leverage_score': 0,
                'uniqueness_value': 0,
                'game_theory_score': 0,
                'overall_grade': 'F',
                'biggest_strength': 'Unknown',
                'biggest_weakness': f'Error: {str(e)}',
                'improvement_suggestion': 'Unable to evaluate',
                'ideal_game_script': 'Unknown',
                'poor_game_script': 'Unknown',
                'error': str(e)
            }
    
    def batch_predict_ownership(self,
                               players_df: pd.DataFrame,
                               context: Dict = None) -> pd.DataFrame:
        """
        Predict ownership for all players in the pool
        
        Args:
            players_df: DataFrame with all players
            context: Shared context (vegas, game environment)
        
        Returns:
            Updated DataFrame with AI predictions
        """
        context = context or {}
        updated_df = players_df.copy()
        
        print(f"🤖 Starting AI ownership prediction for {len(players_df)} players...")
        
        for idx, player in players_df.iterrows():
            player_dict = player.to_dict()
            player_name = str(player_dict.get('name', 'Unknown'))
            
            # Add player-specific context
            player_context = context.copy()
            
            try:
                prediction = self.predict_ownership(player_dict, player_context)
                
                # Update dataframe
                updated_df.at[idx, 'ownership'] = prediction['ownership']
                updated_df.at[idx, 'ai_confidence'] = prediction['confidence']
                updated_df.at[idx, 'ai_reasoning'] = prediction['reasoning']
                
                print(f"  ✓ {player_name}: {prediction['ownership']:.1f}% (confidence: {prediction['confidence']}%)")
                
            except Exception as e:
                print(f"  ⚠️ {player_name}: Failed - {str(e)}")
                # Keep original values on error
                continue
        
        print(f"✅ AI ownership prediction complete!")
        print(f"📊 Requests made: {self.request_count}")
        print(f"💰 Estimated cost: ${self.total_cost:.3f}")
        
        return updated_df
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'requests': self.request_count,
            'estimated_cost': self.total_cost,
            'model': CLAUDE_MODEL
        }
