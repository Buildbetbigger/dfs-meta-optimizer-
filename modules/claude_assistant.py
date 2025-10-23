"""
Claude AI Assistant Module

Uses Claude API to provide intelligent analysis:
- Ownership prediction
- News impact analysis
- Strategic recommendations
- Lineup quality scoring

This replaces the need for Twitter/Reddit scraping and provides
more intelligent analysis than rule-based systems.
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

        # Clean and validate API key
        self.api_key = api_key or ANTHROPIC_API_KEY
        if self.api_key:
            self.api_key = str(self.api_key).strip().strip('"').strip("'")

        if not self.api_key or len(self.api_key) < 20:
            raise ValueError(
                f"ANTHROPIC_API_KEY must be set in .env file or Streamlit secrets. "
                f"Current key length: {len(self.api_key) if self.api_key else 0} chars"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.request_count = 0
        self.total_cost = 0.0
    
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
        
        response = self.client.messages.create(**kwargs)
        
        # Track usage
        self.request_count += 1
        self.total_cost += 0.006  # Rough estimate
        
        return response.content[0].text
    
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
        
        # Calculate value metrics to help AI differentiate
        value_score = (player_data['projection'] / player_data['salary']) * 1000
        
        prompt = f"""Predict the projected ownership for this DFS player in a SHOWDOWN contest:

**Player Information:**
- Name: {player_data['name']}
- Position: {player_data['position']}
- Team: {player_data['team']}
- Salary: ${player_data['salary']:,}
- Current Projection: {player_data['projection']} points
- Value Score: {value_score:.2f} pts per $1K

**Additional Context:**
- Recent Games: {context.get('recent_games', 'Not provided')}
- Breaking News: {context.get('news', 'None')}
- Vegas Lines: {context.get('vegas', 'Not provided')}
- Game Environment: {context.get('environment', 'Standard')}
- Contest Type: {context.get('contest_type', 'GPP')}

**Consider These DFS Psychology Factors:**

1. **Salary Tier Analysis:**
   - Is this player expensive ($9K+), mid-tier ($6-9K), or value ($<6K)?
   - Expensive players: typically 25-40% owned
   - Mid-tier: typically 15-30% owned
   - Value players: typically 5-20% owned

2. **Position-Specific Ownership:**
   - QBs: Generally highest owned (30-45% for top options)
   - Top TEs/WRs: Moderate-high owned (20-35%)
   - RBs in showdown: Varies widely (10-30%)
   - Value plays: Low owned (5-15%)

3. **Team and Matchup Context:**
   - Is this team favored or an underdog?
   - High-scoring game expected or defensive struggle?

4. **Value Score Impact:**
   - High value (>2.5 pts/$1K) = typically attracts more ownership
   - Low value (<2.0 pts/$1K) = typically lower owned

**CRITICAL:** Your prediction must be SPECIFIC to this player's situation. 
DO NOT give generic predictions. Consider their exact salary, position, team, and value.

Respond in ONLY valid JSON format:
{{
    "ownership": <number between 5-50 based on factors above>,
    "confidence": "<high|medium|low>",
    "reasoning": "<2-3 sentence explanation specific to THIS player>",
    "factors": ["<factor1>", "<factor2>", "<factor3>"],
    "risk": "<what could change this prediction>"
}}"""

        try:
            response = self._call_claude(prompt, system)
            
            # Parse JSON from response
            response_clean = response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1])
            
            result = json.loads(response_clean)
            
            # Validate and add metadata
            result['timestamp'] = datetime.now().isoformat()
            result['player'] = player_data['name']
            
            # Ensure ownership is reasonable
            if result['ownership'] < 5:
                result['ownership'] = 5
            elif result['ownership'] > 50:
                result['ownership'] = 50
            
            return result
            
        except json.JSONDecodeError as e:
            # Fallback: Use salary-based heuristic
            base_ownership = 15
            if player_data['salary'] > 9000:
                base_ownership = 30
            elif player_data['salary'] > 7000:
                base_ownership = 20
            else:
                base_ownership = 12
                
            return {
                'ownership': base_ownership,
                'confidence': 'low',
                'reasoning': f'Fallback prediction based on salary tier (${player_data["salary"]})',
                'factors': ['salary_based'],
                'risk': 'Generic prediction - add context for better accuracy',
                'error': str(e),
                'raw_response': response if 'response' in locals() else None
            }
        except Exception as e:
            return {
                'ownership': 15,
                'confidence': 'low',
                'reasoning': f'Error during prediction: {str(e)}',
                'factors': ['error'],
                'risk': 'Unknown',
                'error': str(e)
            }
    
    def analyze_news_impact(self,
                           news: str,
                           players_df: pd.DataFrame) -> Dict:
        """
        Analyze how breaking news impacts the DFS slate
        
        Args:
            news: Breaking news text
            players_df: DataFrame with player data
        
        Returns:
            Dictionary with impact analysis
        """
        system = """You are an expert DFS analyst specializing in breaking news analysis.
You understand how news impacts projections, ownership, and leverage."""
        
        # Create a simplified player summary
        player_summary = players_df[['name', 'team', 'position', 'salary', 'projection', 'ownership']].to_string()
        
        prompt = f"""Analyze how this breaking news impacts the DFS slate:

**Breaking News:**
{news}

**Current Player Pool:**
{player_summary[:1500]}  

**Your Analysis Should Cover:**
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
            result['news_analyzed'] = news[:100]  # Store snippet
            
            return result
            
        except Exception as e:
            return {
                'impacted_players': [],
                'overall_strategy': f'Error analyzing news: {str(e)}',
                'urgency': 'unknown',
                'key_takeaway': 'Unable to analyze',
                'leverage_opportunities': [],
                'error': str(e),
                'raw_response': response if 'response' in locals() else None
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
        
        # Format top players
        top_chalk = player_metrics.nlargest(5, 'ownership')[['name', 'ownership', 'projection', 'leverage']].to_string()
        top_leverage = player_metrics.nlargest(8, 'leverage')[['name', 'ownership', 'projection', 'leverage']].to_string()
        
        prompt = f"""Analyze this DFS field and provide strategic recommendations:

**Field Analysis:**
- Average Ownership: {field_analysis['avg_ownership']:.1f}%
- Chalk Players: {field_analysis['chalk_count']}
- Field Concentration: {field_analysis['field_concentration']:.3f}
- Average Leverage: {field_analysis['avg_leverage']:.2f}

**Top Chalk Players (High Ownership):**
{top_chalk}

**Top Leverage Plays:**
{top_leverage}

**Contest Information:**
- Type: {contest_info.get('type', 'GPP')}
- Field Size: {contest_info.get('entries', 'Unknown')}
- Payout Structure: {contest_info.get('payout', 'Top-heavy')}

**Provide Strategic Analysis:**

1. **Contrarian Level**: On a scale of 1-10, how contrarian should we be?
   - 1 = Play the chalk, safe approach
   - 10 = Maximum differentiation, very contrarian

2. **Chalk Fades**: Which highly-owned players should we fade and why?

3. **Leverage Emphasis**: Which leverage plays deserve increased exposure?

4. **Stacking Strategy**: What's the optimal stacking approach for this field?
   - Game stacks (QB + pass catchers)
   - Bring-back strategies
   - Correlation plays

5. **Captain Strategy**: How should we approach captain selection?

6. **Game Theory**: What does the field NOT see that we should exploit?

Think through: If 40% of the field builds similar lineups, what do we need to do to beat them?

Provide detailed analysis in clear sections."""

        try:
            response = self._call_claude(prompt, system)
            
            return {
                'recommendation': response,
                'timestamp': datetime.now().isoformat(),
                'field_state': field_analysis,
                'contest_type': contest_info.get('type')
            }
            
        except Exception as e:
            return {
                'recommendation': f'Error getting strategic advice: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def score_lineup_quality(self,
                            lineup: Dict,
                            field_context: Dict,
                            opponent_model) -> Dict:
        """
        Get AI analysis of lineup quality and suggestions
        
        Args:
            lineup: Lineup dictionary with players and metrics
            field_context: Current field state
            opponent_model: OpponentModel instance for data
        
        Returns:
            Quality score and recommendations
        """
        system = """You are an expert DFS lineup evaluator.
You assess tournament equity, correlation, leverage, and game theory."""
        
        metrics = lineup['metrics']
        
        prompt = f"""Evaluate this DFS Showdown lineup:

**Lineup Construction:**
- Captain: {lineup['captain']}
- Flex Players: {', '.join(lineup['flex'])}

**Lineup Metrics:**
- Total Projection: {metrics['total_projection']:.1f} points
- Total Ceiling: {metrics['total_ceiling']:.1f} points
- Average Ownership: {metrics['avg_ownership']:.1f}%
- Uniqueness: {metrics['uniqueness']:.1f}%
- Total Salary: ${metrics['total_salary']:,}
- Salary Remaining: ${metrics['salary_remaining']:,}

**Field Context:**
- Average Field Ownership: {field_context.get('avg_ownership', 'Unknown')}%
- Field Concentration: {field_context.get('field_concentration', 'Unknown')}

**Evaluate This Lineup (0-100 score) On:**

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

Provide scores and brief explanations for each dimension."""

        try:
            response = self._call_claude(prompt, system)
            
            return {
                'analysis': response,
                'timestamp': datetime.now().isoformat(),
                'lineup_id': lineup.get('lineup_id')
            }
            
        except Exception as e:
            return {
                'analysis': f'Error scoring lineup: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
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
            
            # Add player-specific context
            player_context = context.copy()
            
            prediction = self.predict_ownership(player_dict, player_context)
            
            # Update dataframe
            updated_df.at[idx, 'ownership'] = prediction['ownership']
            updated_df.at[idx, 'ai_confidence'] = prediction['confidence']
            updated_df.at[idx, 'ai_reasoning'] = prediction['reasoning']
            
            print(f"  ✓ {player['name']}: {prediction['ownership']:.1f}% (was {player['ownership']:.1f}%)")
        
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
