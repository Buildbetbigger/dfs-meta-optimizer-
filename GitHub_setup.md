# GitHub Repository Setup Guide

Complete guide for publishing DFS Meta-Optimizer to GitHub.

## Prerequisites

- Git installed locally
- GitHub account
- Repository created on GitHub

## Step 1: Local Setup

```bash
# Navigate to your project directory
cd /path/to/dfs-meta-optimizer

# Initialize git (if not already done)
git init

# Verify all files are present
ls -la
```

## Step 2: Upload Missing Files

**Required Files Checklist:**
- ✅ README.md
- ✅ LICENSE
- ✅ setup.py
- ✅ requirements.txt
- ✅ .gitignore
- ✅ .env.example
- ✅ CONTRIBUTING.md
- ✅ GITHUB_SETUP.md (this file)
- ✅ INTEGRATION_GUIDE.md
- ✅ FIX_SUMMARY.md
- ✅ MANIFEST.in

**Copy files from download:**
```bash
# Copy GitHub setup files to project root
cp /path/to/downloads/.gitignore .
cp /path/to/downloads/.env.example .
cp /path/to/downloads/CONTRIBUTING.md .
cp /path/to/downloads/GITHUB_SETUP.md .
cp /path/to/downloads/MANIFEST.in .
```

## Step 3: Configure Git

```bash
# Set your identity
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Verify .gitignore is working
cat .gitignore
```

## Step 4: Initial Commit

```bash
# Add all files
git add .

# Review what will be committed
git status

# Commit
git commit -m "Initial commit: DFS Meta-Optimizer v8.0.0"
```

## Step 5: Connect to GitHub

**Option A: New Repository**
```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/username/dfs-meta-optimizer.git
git branch -M main
git push -u origin main
```

**Option B: Existing Repository**
```bash
git remote add origin https://github.com/username/dfs-meta-optimizer.git
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Step 6: Verify Upload

**Check on GitHub:**
- ✅ All 11 core files visible
- ✅ All 27 Python modules uploaded
- ✅ README.md renders correctly
- ✅ LICENSE file present
- ✅ .gitignore working (no .env file visible)

## Step 7: Repository Settings

**Update repository settings:**

1. **Description:** "Professional-grade DFS lineup optimizer with AI and advanced mathematics"

2. **Topics:** Add tags
   - dfs
   - daily-fantasy-sports
   - optimizer
   - ai
   - machine-learning
   - python
   - streamlit

3. **Social Preview:** Upload screenshot of app.py interface

4. **Enable Features:**
   - ✅ Issues
   - ✅ Discussions
   - ✅ Projects

## Step 8: Create Releases

```bash
# Tag version
git tag -a v8.0.0 -m "Release v8.0.0: Most Advanced State"
git push origin v8.0.0
```

**On GitHub:**
1. Go to Releases → Create new release
2. Tag: v8.0.0
3. Title: "v8.0.0 - Most Advanced State Achieved"
4. Description: Copy from CHANGELOG
5. Attach: Package files (optional)

## Step 9: Documentation

**Update README.md:**
- Replace `<your-repo-url>` with actual URL
- Update installation instructions
- Test all commands

**Create Wiki (Optional):**
- Getting Started
- Advanced Features
- API Reference
- Troubleshooting

## Step 10: Protection Rules

**Protect main branch:**
1. Settings → Branches
2. Add rule for `main`
3. Enable:
   - Require pull request reviews
   - Require status checks
   - Restrict force push

## Maintenance

### Regular Updates

```bash
# Make changes
git add .
git commit -m "feat: Add new feature"
git push

# New version
git tag -a v8.1.0 -m "Release v8.1.0"
git push origin v8.1.0
```

### Issue Management

**Labels to create:**
- bug (red)
- enhancement (blue)
- documentation (yellow)
- good-first-issue (green)
- help-wanted (purple)

## Common Issues

**Issue: Large file rejection**
```bash
# Remove file from git history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/large/file' \
  --prune-empty --tag-name-filter cat -- --all
```

**Issue: Merge conflicts**
```bash
# Pull latest
git pull origin main

# Resolve conflicts manually
# Then:
git add .
git commit -m "Resolve merge conflicts"
git push
```

**Issue: Wrong remote URL**
```bash
# Check current remote
git remote -v

# Update remote
git remote set-url origin https://github.com/username/new-repo.git
```

## Publishing to PyPI (Optional)

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
pip install twine
twine upload dist/*
```

## CI/CD Setup (Optional)

**GitHub Actions workflow (.github/workflows/test.yml):**
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

## Security

**Secrets Management:**
1. Never commit API keys
2. Use GitHub Secrets for CI/CD
3. Enable security alerts
4. Review Dependabot PRs

**Verify .env is ignored:**
```bash
git status  # Should NOT show .env
```

## Checklist Before Going Public

- [ ] All files uploaded
- [ ] README.md complete
- [ ] LICENSE file present
- [ ] API keys removed from code
- [ ] Tests passing
- [ ] Documentation complete
- [ ] Version tagged
- [ ] Release notes written
- [ ] Social preview added
- [ ] Repository description set
- [ ] Topics/tags added

## Support

**Enable GitHub features:**
- ⭐ Star button
- 👁️ Watch notifications
- 🍴 Fork option
- 📝 Issues
- 💬 Discussions

## Success Metrics

**After 1 week:**
- ⭐ 10+ stars
- 🍴 5+ forks
- 📝 0 critical issues

**After 1 month:**
- ⭐ 50+ stars
- 🍴 10+ forks
- 📊 100+ clones

---

## Quick Commands Reference

```bash
# Status check
git status

# Add files
git add .

# Commit
git commit -m "message"

# Push
git push

# Pull latest
git pull

# Create tag
git tag -a v1.0.0 -m "Release v1.0.0"

# Push tag
git push origin v1.0.0

# View remotes
git remote -v

# View branches
git branch -a
```

---

**Questions?** Open an issue or check GitHub's documentation.

**Ready to publish!** 🚀
