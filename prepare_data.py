"""
Data Preparation Script

Prepares raw DFS data for the optimizer by adding ceiling, floor, and ownership columns.

Usage:
    python prepare_data.py your_raw_data.csv
"""

import pandas as pd
import sys

def prepare_data(input_file):
    """Prepare raw CSV for optimizer"""
    
    print(f"📂 Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Verify required columns
    required = ['name', 'team', 'position', 'salary', 'projection']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"✅ Found {len(df)} players")
    
    # Calculate ceiling and floor if missing
    if 'ceiling' not in df.columns:
        df['ceiling'] = df['projection'] * 1.7
        print("✅ Calculated ceiling (projection × 1.7)")
    
    if 'floor' not in df.columns:
        df['floor'] = df['projection'] * 0.4
        print("✅ Calculated floor (projection × 0.4)")
    
    # Add placeholder ownership (AI will predict actual values)
    if 'ownership' not in df.columns:
        df['ownership'] = 15
        print("✅ Added placeholder ownership (AI will predict)")
    
    # Save prepared file
    output_file = 'data/prepared_players.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n🎉 Data prepared successfully!")
    print(f"📊 Saved to: {output_file}")
    print(f"\n📋 Next steps:")
    print(f"   1. Run: streamlit run app.py")
    print(f"   2. Upload: {output_file}")
    print(f"   3. Use AI to predict ownership")
    print(f"   4. Generate lineups!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_data.py your_raw_data.csv")
        sys.exit(1)
    
    prepare_data(sys.argv[1])
