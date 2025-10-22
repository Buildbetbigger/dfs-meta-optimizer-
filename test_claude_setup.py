"""
Test script to verify Claude API setup

Run this before using the main app to ensure your API key is configured correctly.

Usage:
    python test_claude_setup.py
"""

import os
import sys
from dotenv import load_dotenv

print("🧪 Testing Claude API Setup...\n")

# Test 1: Check if .env file exists
print("Test 1: Checking for .env file...")
if os.path.exists('.env'):
    print("✅ .env file found")
else:
    print("❌ .env file not found")
    print("💡 Create one by copying .env.example:")
    print("   cp .env.example .env")
    sys.exit(1)

# Test 2: Load environment variables
print("\nTest 2: Loading environment variables...")
load_dotenv()
api_key = os.getenv('ANTHROPIC_API_KEY')

if api_key:
    print(f"✅ ANTHROPIC_API_KEY found")
    print(f"   Key preview: {api_key[:15]}...{api_key[-4:]}")
else:
    print("❌ ANTHROPIC_API_KEY not set in .env")
    print("💡 Add your API key to .env:")
    print("   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
    sys.exit(1)

# Test 3: Check if anthropic package is installed
print("\nTest 3: Checking anthropic package...")
try:
    import anthropic
    print(f"✅ anthropic package installed (version {anthropic.__version__})")
except ImportError:
    print("❌ anthropic package not installed")
    print("💡 Install it with:")
    print("   pip install anthropic==0.39.0")
    sys.exit(1)

# Test 4: Test API connection
print("\nTest 4: Testing API connection...")
try:
    from anthropic import Anthropic
    
    client = Anthropic(api_key=api_key)
    
    print("   Sending test request to Claude...")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Say 'API test successful' and nothing else."}
        ]
    )
    
    response_text = response.content[0].text
    print(f"✅ API connection successful!")
    print(f"   Claude response: {response_text}")
    
    # Test 5: Estimate cost
    print("\nTest 5: Calculating usage...")
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    # Approximate costs for Sonnet 4
    input_cost = (input_tokens / 1_000_000) * 3
    output_cost = (output_tokens / 1_000_000) * 15
    total_cost = input_cost + output_cost
    
    print(f"   Input tokens: {input_tokens}")
    print(f"   Output tokens: {output_tokens}")
    print(f"   Estimated cost: ${total_cost:.6f}")
    
except Exception as e:
    print(f"❌ API connection failed: {str(e)}")
    print("\n💡 Common issues:")
    print("   - Invalid API key (check for typos)")
    print("   - No credits/billing set up at console.anthropic.com")
    print("   - Network connectivity issues")
    sys.exit(1)

# Test 6: Test DFS-specific functionality
print("\nTest 6: Testing DFS analysis capability...")
try:
    test_prompt = """You are a DFS analyst. A player scored 30 points last week. 
Will their ownership be high this week? Respond with just 'Yes' or 'No' and a one-sentence reason."""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": test_prompt}]
    )
    
    analysis = response.content[0].text
    print(f"✅ DFS analysis working!")
    print(f"   Test analysis: {analysis}")
    
except Exception as e:
    print(f"❌ DFS analysis test failed: {str(e)}")
    sys.exit(1)

# All tests passed
print("\n" + "="*50)
print("🎉 All tests passed! Your Claude API is ready to use.")
print("="*50)
print("\n📊 Summary:")
print(f"   ✅ .env file configured")
print(f"   ✅ API key valid")
print(f"   ✅ anthropic package installed")
print(f"   ✅ API connection working")
print(f"   ✅ DFS analysis capable")
print("\n🚀 You're ready to run the optimizer:")
print("   streamlit run app.py")
print("\n💡 Tips:")
print("   - Each ownership prediction costs ~$0.006")
print("   - 20 players = ~$0.12 per contest")
print("   - Monitor usage at console.anthropic.com")
print("\n📚 See PHASE_1.5_GUIDE.md for usage instructions")
