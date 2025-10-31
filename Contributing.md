# Contributing to DFS Meta-Optimizer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

**Before submitting:**
- Check existing issues to avoid duplicates
- Verify you're using the latest version
- Test with minimal configuration

**Include in bug report:**
- Python version and OS
- Complete error message and traceback
- Minimal code to reproduce
- Expected vs actual behavior

### Suggesting Features

**Good feature requests include:**
- Clear problem statement
- Proposed solution
- Use cases and benefits
- Alternatives considered

### Pull Requests

**Process:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes following code standards
4. Add/update tests
5. Update documentation
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open Pull Request

**PR Guidelines:**
- One feature/fix per PR
- Clear description of changes
- Link related issues
- Pass all tests
- Update CHANGELOG.md

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/dfs-meta-optimizer.git
cd dfs-meta-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest black mypy flake8

# Run tests
pytest tests/

# Format code
black .

# Type checking
mypy .

# Linting
flake8 .
```

## Code Standards

### Style Guide
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for all functions/classes

### Example:
```python
def optimize_lineup(
    players: pd.DataFrame,
    salary_cap: int = 50000,
    positions: Dict[str, int] = None
) -> List[Dict]:
    """
    Generate optimal DFS lineup.
    
    Args:
        players: Player pool DataFrame
        salary_cap: Maximum salary (default: 50000)
        positions: Position requirements
    
    Returns:
        List of optimized lineups
    
    Raises:
        ValueError: If salary_cap invalid
    """
    pass
```

### Testing
- Write tests for new features
- Maintain 80%+ code coverage
- Use pytest fixtures
- Mock external APIs

### Documentation
- Update README.md for major features
- Add docstrings to all public functions
- Update INTEGRATION_GUIDE.md for usage changes
- Comment complex algorithms

## Project Structure

```
dfs-meta-optimizer/
├── Core modules: optimization_engine.py, app.py
├── Phase 2 (Math): sharpe_optimizer.py, advanced_kelly.py
├── Phase 3 (Data/AI): phase3_*.py
├── Utils: settings.py, data_config.py
└── Tests: tests/
```

## Commit Messages

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Testing
- `chore`: Maintenance

**Example:**
```
feat(optimizer): Add parallel lineup generation

Implement multi-core processing using ParallelOptimizer class.
Reduces generation time by 70% for 100+ lineups.

Closes #123
```

## Module Guidelines

### Adding New Modules

1. **Planning:**
   - Define clear purpose
   - Design API interface
   - Consider integration points

2. **Implementation:**
   - Follow existing patterns
   - Add comprehensive tests
   - Document thoroughly

3. **Integration:**
   - Update phase3_integration.py
   - Add to quick_start()
   - Update INTEGRATION_GUIDE.md

### Module Template

```python
"""
Module Name - Brief description
Part of DFS Meta-Optimizer v8.0.0
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class ModuleName:
    """Brief class description."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional config."""
        self.config = config or {}
        logger.info("ModuleName initialized")
    
    def main_method(self, data: pd.DataFrame) -> Dict:
        """
        Main functionality.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Results dictionary
        """
        logger.debug(f"Processing {len(data)} records")
        # Implementation
        return {}
```

## Release Process

**Versioning:** Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

**Checklist:**
1. Update version in setup.py
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release tag
6. Build distribution

## Getting Help

- Check existing issues and discussions
- Review INTEGRATION_GUIDE.md
- Ask in GitHub Discussions
- Contact maintainers

## Recognition

Contributors are recognized in:
- CHANGELOG.md
- GitHub Contributors page
- Release notes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Questions?** Open an issue or discussion on GitHub.

**Thank you for contributing to DFS Meta-Optimizer!** 🎯
