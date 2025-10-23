"""
Test script to verify Claude API is working end-to-end

Run this in your Streamlit environment to test the API key
"""
import streamlit as st

st.title("🔧 Claude API Key Test")

# Step 1: Read the key
api_key = st.secrets.get("ANTHROPIC_API_KEY", "NOT FOUND")

st.write("### Step 1: Reading Key from Secrets")
st.write(f"✅ Key exists: {api_key != 'NOT FOUND'}")
st.write(f"✅ Key starts correctly: {api_key.startswith('sk-ant-') if api_key != 'NOT FOUND' else False}")
st.write(f"✅ Key length: {len(api_key) if api_key != 'NOT FOUND' else 0}")

# Step 2: Clean the key
if api_key != "NOT FOUND":
    cleaned_key = str(api_key).strip().strip('"').strip("'").replace('\n', '').replace('\r', '').replace(' ', '')
    
    st.write("### Step 2: Cleaning Key")
    st.write(f"Cleaned key first 15 chars: {cleaned_key[:15]}...")
    st.write(f"Cleaned key last 10 chars: ...{cleaned_key[-10:]}")
    st.write(f"Cleaned key length: {len(cleaned_key)}")
    
    # Step 3: Test actual API call
    if st.button("🧪 Test API Call"):
        st.write("### Step 3: Making Test API Call")
        
        try:
            from anthropic import Anthropic
            
            st.write("✅ Anthropic module imported")
            
            # Create client
            client = Anthropic(api_key=cleaned_key)
            st.write("✅ Client created")
            
            # Make test call
            st.write("📡 Calling API...")
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{"role": "user", "content": "Say 'API test successful' and nothing else"}]
            )
            
            st.write("✅ API call successful!")
            st.success(f"Response: {response.content[0].text}")
            
        except Exception as e:
            st.error(f"❌ API call failed: {str(e)}")
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error details: {str(e)}")
            
            # Show what key was used (safely)
            st.write(f"Key used (first 15): {cleaned_key[:15]}...")
            st.write(f"Key used (last 10): ...{cleaned_key[-10:]}")
            st.write(f"Key length: {len(cleaned_key)}")
