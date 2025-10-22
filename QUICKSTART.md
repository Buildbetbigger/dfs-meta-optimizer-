# 🚀 Quick Start Guide

Get your DFS Meta-Optimizer running in 5 minutes!

## ⚡ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/dfs-meta-optimizer.git
cd dfs-meta-optimizer
```

### Step 2: Set Up Python Environment

**Option A: Using venv (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Option B: Using conda**
```bash
conda create -n dfs-optimizer python=3.9
conda activate dfs-optimizer
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment (Optional for Phase 1)
```bash
cp .env.example .env
# Edit .env if you want to add API keys (not needed for Phase 1)
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 📊 Using the Optimizer

### Quick Workflow

1. **Load Data**
   - Click "Use Sample Data" for a KC vs BUF example
   - Or upload your own CSV with player data

2. **Analyze Field**
   - Click "Analyze Field" button
   - Review chalk players and leverage plays
   - Check the strategy recommendation

3. **Generate Lineups**
   - Choose number of lineups (recommend 20 for testing)
   - Select optimization mode:
     - **Anti-Chalk**: Maximum differentiation (GPPs)
     - **Leverage**: Ceiling/ownership optimization
     - **Balanced**: Good for most contests
     - **Safe**: High floor for cash games
   - Click "Generate Lineups"

4. **Export**
   - Review generated lineups
   - Click "Export to CSV"
   - Upload to DraftKings

---

## 📁 CSV Format for Player Data

Your CSV must include these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| name | string | Player name | Patrick Mahomes |
| team | string | Team abbreviation | KC |
| position | string | QB, RB, WR, or TE | QB |
| salary | integer | DraftKings salary | 11200 |
| projection | float | Projected points | 24.5 |
| ownership | float | Projected ownership % | 35.0 |
| ceiling | float | 80th percentile outcome | 42.0 |
| floor | float | 20th percentile outcome | 15.0 |

### Example CSV:
```csv
name,team,position,salary,projection,ownership,ceiling,floor
Patrick Mahomes,KC,QB,11200,24.5,35,42,15
Travis Kelce,KC,TE,9800,18.2,28,32,10
Josh Allen,BUF,QB,11400,25.1,38,44,16
```

### Where to Get Data:

- **Projections**: FantasyPros, Establish The Run, 4for4
- **Ownership**: RotoGrinders, FantasyLabs, DFS Army
- **Ceiling/Floor**: Use projection ± standard deviation

---

## 🎯 Understanding the Metrics

### Leverage Score
```
Leverage = (Ceiling Points / Ownership %) × 100
```
**What it means**: Tournament-winning upside relative to ownership  
**Good score**: > 2.0 for GPPs  
**Why it matters**: High leverage = better than field when hits ceiling

### Uniqueness
```
Uniqueness = 100 - Average Ownership %
```
**What it means**: How different your lineup is from the field  
**Good score**: > 70% for GPPs, > 50% for double-ups  
**Why it matters**: Need differentiation to win large tournaments

### Chalk Flag
**Threshold**: > 40% projected ownership  
**Strategy**: Usually fade in GPPs, okay in cash games  
**Why it matters**: High-owned players have negative leverage

---

## 🔧 Troubleshooting

### Application won't start
```bash
# Make sure you're in the right directory
cd dfs-meta-optimizer

# Verify Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Import errors
```bash
# Make sure you activated the virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### CSV won't load
- Check that all required columns are present
- Ensure no special characters in player names
- Verify numeric columns are actually numbers, not text

### No lineups generated
- Check that you have enough players (minimum 12-15 recommended)
- Verify salary cap isn't too restrictive
- Try lowering max exposure threshold

---

## 📈 Next Steps

### Phase 2 Features (Coming Soon)
- [ ] Real-time Vegas odds integration
- [ ] Twitter sentiment analysis
- [ ] Reddit buzz tracking
- [ ] Automated ownership prediction
- [ ] News alert system

### Phase 3 Features (Roadmap)
- [ ] Multi-agent AI ensemble
- [ ] Historical backtest engine
- [ ] ML-based ownership model
- [ ] Contest ROI tracking
- [ ] Mobile app

---

## 💡 Pro Tips

1. **Start with Sample Data**
   - Get familiar with the interface
   - Understand the metrics
   - Test different optimization modes

2. **Compare Modes**
   - Generate 10 lineups in each mode
   - See how they differ
   - Understand the tradeoffs

3. **Monitor Leverage**
   - Sort by leverage score
   - High leverage players are key in GPPs
   - Balance leverage with projection

4. **Check Uniqueness**
   - Aim for 60-80% in large-field GPPs
   - Lower (40-50%) okay for smaller contests
   - Too high (>85%) = might be too contrarian

5. **Diversify Your Portfolio**
   - Don't use same players in every lineup
   - Mix modes (some anti-chalk, some balanced)
   - Ensure 3+ player differences between lineups

---

## 🆘 Need Help?

- **Issues**: Open an issue on GitHub
- **Questions**: Check the README.md
- **Updates**: Watch the repository for new releases

---

## ✅ Quick Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Requirements installed
- [ ] Application runs without errors
- [ ] Sample data loads successfully
- [ ] Can generate lineups
- [ ] Can export to CSV
- [ ] Understand leverage and uniqueness metrics

If you've checked all these boxes, you're ready to start optimizing! 🎉

---

**Good luck, and may your leverage plays hit their ceiling!** 🚀
