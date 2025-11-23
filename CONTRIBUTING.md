# Contributing to LiveKit Local Plugins

Thanks for your interest in contributing! This project was built to help others run LiveKit voice agents locally, and contributions are welcome.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/livekit-local-plugins.git
   cd livekit-local-plugins
   ```
3. Install [uv](https://docs.astral.sh/uv/) if you haven't:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
4. Install dependencies:
   ```bash
   uv sync --all-extras
   ```

## Development Workflow

### Running Commands

All commands should be run through `uv run` to use the project's virtual environment:

```bash
# Run the example agent
uv run examples/voice_agent.py dev

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest

# Format code
uv run ruff format src/
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

## Code Style

- Use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Follow PEP 8 conventions
- Add type hints to all function signatures
- Write docstrings for all public classes and functions

### Pre-commit Checks

Before submitting a PR, ensure:

```bash
uv run ruff check src/       # No linting errors
uv run ruff format src/      # Code is formatted
uv run mypy src/             # No type errors
```

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Reference issues when applicable: "Fix #123: ..."

## What to Contribute

### High-Value Contributions

- Additional STT plugins (e.g., Vosk, SpeechRecognition)
- Additional TTS plugins (e.g., Coqui, Bark)
- Performance optimizations
- Documentation improvements
- Bug fixes

### Before Starting Large Changes

Please open an issue first to discuss:
- Major architectural changes
- New plugin implementations
- Changes to the public API

## Questions?

Feel free to open an issue for questions or join the discussion on GitHub.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
