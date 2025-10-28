"""
AI Assistant Module

Uses Claude API for:
- Ownership prediction
- News impact analysis
- Strategic recommendations
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  anthropic package not installed. Install with: pip install anthropic")


class AIAssistant:
    """
    AI-powered DFS analysis using Claude API
    
    BUG FIX #1: Properly implements ownership prediction
    - Actually calls Claude API
    - Returns varied ownership percentages
    - Not just 15% for everyone
    """
    
    def __init__(self, api_key: str):
        """
        Initialize AI assistant
        
        Args:
            api_key: Anthropic API key
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic --break-system-packages")
        
        if not api_key:
            raise ValueError("API key is required")
        
        # Clean and validate API key
        self.api_key = str(api_key).strip().strip('"').strip("'")
        self.api_key = self.api_key.replace('\n', '').replace('\r', '').replace(' ', '')
        
        if len(self.api_key) < 50:
            raise ValueError(f"API key too short ({len(self.api_key)} chars). Expected 50+")
        
        if not self.api_key.startswith('sk-ant-'):
            raise ValueError(f"Invalid API key format. Should start with 'sk-ant-'")
        
        # Initialize client
        try:
            self.client = Anthropic(api_key=self.api_key)
            self.request_count = 0
            self.total_cost = 0.0
            print(f"✅ AI Assistant initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to create Anthropic client: {str(e)}")
    
    def _call_claude(self, prompt: str, system: str = None, max_tokens: int = 4000) -> str:
        """
        Make API call to Claude
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Max response tokens
            
        Returns:
            Claude's response text
        """
        try:
            kwargs = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system:
                kwargs["system"] = system
            
            response = self.client.messages.create(**kwargs)
            
            self.request_count += 1
            self.total_cost += 0.006  # Rough estimate
            
            return response.content[0].text
            
        except Exception as e:
            print(f"❌ Claude API error: {str(e)}")
            raise
    
    def predict_ownership(self, players_df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict ownership percentages for all players
        
        BUG FIX: This now actually calls the API and returns varied predictions
        
        Args:
            players_df: DataFrame with player data
            
        Returns:
            Dictionary mapping player names to predicted ownership %
        """
        # Prepare player data for prompt
        player_list = []
        for _, player in players_df.iterrows():
            player_info = {
                'name': str(player['name']),
                'position': str(player['position']),
                'team': str(player['team']),
                'salary': int(player['salary']),
                'projection': float(player['projection']),
                'value': float(player['projection'] / (player['salary'] / 1000))
            }
            player_list.append(player_info)
        
        # Create prompt
        system_prompt = """You are an expert DFS analyst specializing in ownership prediction.
You understand player psychology, recency bias, narrative angles, and market behavior.

Your task is to predict what percentage of the field will own each player in a DFS contest."""

        user_prompt = f"""Predict ownership percentages for these players in a GPP contest.

Players:
{json.dumps(player_list, indent=2)}

Consider these factors:
1. Value (points per $1000 salary) - cheap players get overowned
2. Star power - big names get high ownership
3. Position scarcity - fewer good TEs means higher ownership for top ones
4. Salary - expensive players typically have lower ownership
5. Projection - higher projections drive ownership

Return ONLY a JSON object mapping player names to ownership percentages (0-100).
Format: {{"Player Name": 25.5, "Another Player": 18.2}}

DO NOT include any other text. ONLY the JSON object."""

        try:
            # Call Claude API
            response_text = self._call_claude(user_prompt, system_prompt, max_tokens=2000)
            
            # Parse JSON response
            # Remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith('```'):
                # Remove ```json and ``` markers
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            ownership_predictions = json.loads(response_text)
            
            # Validate predictions
            valid_predictions = {}
            for name, ownership in ownership_predictions.items():
                # Clean player name
                clean_name = str(name).strip()
                
                # Validate ownership range
                ownership_val = float(ownership)
                if ownership_val < 0:
                    ownership_val = 5.0
                elif ownership_val > 100:
                    ownership_val = 40.0
                
                valid_predictions[clean_name] = ownership_val
            
            print(f"✅ Predicted ownership for {len(valid_predictions)} players")
            return valid_predictions
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Response was: {response_text[:200]}")
            # Return default predictions on error
            return {str(p['name']): 15.0 for p in player_list}
        except Exception as e:
            print(f"❌ Ownership prediction error: {e}")
            # Return default predictions on error
            return {str(p['name']): 15.0 for p in player_list}
    
    def analyze_news_impact(self, player_name: str, news_text: str) -> Dict:
        """
        Analyze impact of news on player
        
        Args:
            player_name: Player name
            news_text: News content
            
        Returns:
            Impact analysis dictionary
        """
        system_prompt = """You are a DFS expert analyzing news impact on player value."""
        
        user_prompt = f"""Analyze this news about {player_name}:

"{news_text}"

Provide:
1. Impact severity (0-100, where 100 is season-ending injury)
2. Projection adjustment (e.g., -15% for questionable tag)
3. Brief reasoning (1-2 sentences)

Return ONLY a JSON object:
{{
  "impact_score": 45,
  "projection_adjustment": -10,
  "reasoning": "Brief explanation here"
}}"""

        try:
            response = self._call_claude(user_prompt, system_prompt, max_tokens=500)
            
            # Clean response
            response = response.strip().replace('```json', '').replace('```', '').strip()
            result = json.loads(response)
            
            return {
                'impact_score': float(result.get('impact_score', 0)),
                'projection_adjustment': float(result.get('projection_adjustment', 0)),
                'reasoning': str(result.get('reasoning', 'No reasoning provided'))
            }
            
        except Exception as e:
            print(f"❌ News analysis error: {e}")
            return {
                'impact_score': 0,
                'projection_adjustment': 0,
                'reasoning': 'Analysis failed'
            }
    
    def get_strategic_advice(self, player_pool: pd.DataFrame, mode: str = 'GPP') -> str:
        """
        Get strategic advice for lineup construction
        
        Args:
            player_pool: Available players
            mode: Contest type (GPP, Cash, etc.)
            
        Returns:
            Strategic advice text
        """
        # Get top players by value
        top_value = player_pool.nlargest(10, 'projection')
        
        player_summary = []
        for _, player in top_value.iterrows():
            player_summary.append({
                'name': str(player['name']),
                'position': str(player['position']),
                'salary': int(player['salary']),
                'projection': float(player['projection']),
                'ownership': float(player.get('ownership', 15.0))
            })
        
        system_prompt = f"""You are a DFS expert providing strategic advice for {mode} contests."""
        
        user_prompt = f"""Given these top projected players:

{json.dumps(player_summary, indent=2)}

Provide 3-5 strategic insights for building {mode} lineups today.
Focus on:
- Leverage opportunities (high ceiling, low ownership)
- Stacking strategies
- Chalk plays to consider/avoid
- Value plays

Keep response under 200 words."""

        try:
            response = self._call_claude(user_prompt, system_prompt, max_tokens=1000)
            return response
        except Exception as e:
            print(f"❌ Strategic advice error: {e}")
            return "Unable to generate strategic advice at this time."
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'requests_made': self.request_count,
            'estimated_cost': self.total_cost,
            'avg_cost_per_request': self.total_cost / max(self.request_count, 1)
        }
