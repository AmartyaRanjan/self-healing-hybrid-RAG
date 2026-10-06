# Contributing to Self-Healing Hybrid RAG

Thank you for your interest in contributing! This document outlines the workflow for making contributions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/self-healing-hybrid-RAG.git
   cd self-healing-hybrid-RAG
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/harshitsharmaaaa/self-healing-hybrid-RAG.git
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Set up environment**:
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your API keys
   ```
6. **Start databases**:
   ```bash
   docker-compose up -d
   ```

## Branching Strategy

- `main` — stable, production-ready code
- `feature/<description>` — new features and enhancements
- `fix/<description>` — bug fixes
- `docs/<description>` — documentation changes

Always create a new branch from `main` before making changes:
```bash
git checkout main
git pull upstream main
git checkout -b feature/my-feature
```

## Making Changes

1. Make your changes on your feature branch
2. Write or update tests for any new functionality
3. Ensure all tests pass: `pytest`
4. Commit with a clear message:
   ```bash
   git add .
   git commit -m "feat: add new retrieval strategy"
   ```

## Commit Message Format

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `test:` — test additions or fixes
- `refactor:` — code refactoring
- `chore:` — maintenance tasks

## Pull Request Process

1. Push your branch to your fork:
   ```bash
   git push origin feature/my-feature
   ```
2. Open a Pull Request against `main`
3. Fill in the PR description:
   - What changes were made
   - Why they were needed
   - How to test them
4. Respond to review comments
5. Once approved, your changes will be merged

## Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to all public functions and classes
- Keep functions focused and small

## Testing

- Run tests before submitting: `pytest`
- Add tests for all new features
- Mock external API calls (Gemini, Neo4j, Chroma) in tests

## Questions?

Open an issue for discussion before starting large changes.
