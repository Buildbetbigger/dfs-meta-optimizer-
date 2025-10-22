# DFS Meta-Optimizer: Core Engine

## 🎯 Project Vision

A revolutionary DFS optimizer that doesn't just find the "best" lineups—it finds lineups optimized to **beat the field**. By combining opponent modeling with real-time adaptation, this tool provides a true competitive advantage in daily fantasy sports.

## 🧠 Core Philosophy

Traditional optimizers maximize projected points. This meta-optimizer maximizes **probability of beating opponents** by:

1. **Opponent Modeling** - Predicting and exploiting field behavior
2. **Real-Time Adaptation** - Continuously updating until contest lock
3. **Leverage Optimization** - Balancing projection with ownership
4. **Anti-Chalk Strategy** - Building differentiated portfolios
5. **🆕 AI-Powered Analysis** - Claude API for intelligent insights (Phase 1.5)

## 🏗️ Architecture

### Core Modules

- **Opponent Modeling Engine** - Predicts field ownership, calculates leverage scores
- **Real-Time Adaptation** - Monitors sentiment, Vegas lines, and breaking news
- **Optimization Engine** - Generates lineups optimized for competitive advantage
- **Sentiment Analysis** - Tracks social media buzz to predict chalk formation

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/dfs-meta-optimizer.git
cd dfs-meta-optimizer

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your Claude API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Test your Claude API setup (recommended)
python test_claude_setup.py
```

### Get Claude API Key (Phase 1.5)

1. Visit https://console.anthropic.com/
2. Sign up and navigate to "API Keys"
3. Create a new key
4. Add it to your `.env` file

**Cost:** ~$0.30 per contest session (cheaper than any DFS data service!)

### Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Data Format

### Player CSV Format
```csv
name,team,position,salary,projection,ownership,ceiling,floor
Patrick Mahomes,KC,QB,11200,24.5,35,42,15
Travis Kelce,KC,TE,9800,18.2,28,32,10
```

**Required columns:**
- `name` - Player name
- `team` - Team abbreviation
- `position` - Position (QB, RB, WR, TE)
- `salary` - DraftKings salary
- `projection` - Projected fantasy points
- `ownership` - Projected ownership % (0-100)
- `ceiling` - 80th percentile outcome
- `floor` - 20th percentile outcome

## 🎮 Usage

### Basic Workflow

1. **Upload Player Pool** - Import CSV or enter manually
2. **Configure Settings** - Set optimization mode, lineup count
3. **Generate Lineups** - Run opponent-modeled optimization
4. **Monitor Real-Time** - Update as news breaks (manual for MVP)
5. **Export Lineups** - Download CSV for DraftKings upload

### Optimization Modes

- **Anti-Chalk** - Maximum differentiation from field
- **Leverage** - Optimize for ceiling/ownership ratio
- **Balanced** - Mix of projection and leverage
- **Scenario-Based** - Build for specific game scripts

## 🔧 Configuration

Edit `config/settings.py` to adjust:
```python
SALARY_CAP = 50000
ROSTER_SIZE = 6
CAPTAIN_MULTIPLIER = 1.5
MAX_OWNERSHIP_THRESHOLD = 40  # Flag chalk above this %
MIN_LEVERAGE_SCORE = 1.5      # Minimum acceptable leverage
```

## 📈 Key Metrics

### Leverage Score
```
Leverage = (Ceiling Points / Ownership %) × 100
```

High leverage = tournament-winning upside relative to ownership

### Uniqueness Score
```
Uniqueness = 100 - Average Ownership %
```

Higher = more differentiated from field

### Anti-Chalk Score
```
Anti-Chalk = % of lineup that fades high-owned (>40%) players
```

## 🛣️ Roadmap

### Phase 1: Core Engine ✅ COMPLETE
- [x] Opponent modeling framework
- [x] Leverage calculation
- [x] Basic optimization engine
- [x] Streamlit dashboard
- [x] CSV import/export

### Phase 1.5: AI Integration ✅ COMPLETE (NEW!)
- [x] Claude API integration
- [x] AI ownership prediction
- [x] Breaking news analysis
- [x] Strategic recommendations
- [x] Cost-effective real-time insights

### Phase 2: Full Automation (Future)
- [ ] Automated Twitter sentiment (if needed)
- [ ] Automated Reddit tracking (if needed)
- [ ] Vegas odds API integration
- [ ] Continuous re-optimization
- [ ] Scheduled updates

### Phase 3: Advanced Features
- [ ] Multi-agent AI ensemble
- [ ] Historical win rate analysis
- [ ] ML ownership prediction models
- [ ] Portfolio theory optimization
- [ ] Backtest engine

### Phase 4: Scale & Polish
- [ ] Authentication system
- [ ] Contest history tracking
- [ ] ROI analytics dashboard
- [ ] Mobile responsive UI
- [ ] Public API

## 🧪 Testing
```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_opponent_modeling.py

# Run with coverage
pytest --cov=modules tests/
```

## 📚 Resources

### DFS Strategy
- [Leverage in DFS](https://rotogrinders.com/articles/leverage-in-dfs)
- [Ownership and Game Theory](https://www.fantasylabs.com/articles/ownership-game-theory)

### APIs Used
- [The Odds API](https://the-odds-api.com/) - Vegas lines
- [Twitter API](https://developer.twitter.com/) - Sentiment analysis
- Reddit API (PRAW) - Community buzz

## 🤝 Contributing

This is a personal project in active development. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Disclaimer

This tool is for research and educational purposes. DFS involves real money and risk. Use responsibly and within your means. Past performance does not guarantee future results.

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built on the shoulders of giants in DFS strategy:
- RotoGrinders community
- FantasyLabs research
- Establish The Run analysis

---

**Questions?** Open an issue or reach out on Twitter [@yourusername]

**Good luck!** 🍀 May your leverage plays hit their ceiling.
