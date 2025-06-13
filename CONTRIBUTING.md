# Contributing to Hardware-Data-Parameter Co-Design Framework

Thank you for your interest in contributing to the Hardware-Data-Parameter Co-Design Framework! This document provides guidelines for contributing to this research project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Research Contributions](#research-contributions)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that promotes an inclusive and respectful environment for all contributors. By participating, you agree to uphold these standards:

- Be respectful and inclusive
- Focus on constructive feedback
- Acknowledge different perspectives and experiences
- Show empathy towards other community members
- Respect the research nature of this project

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended: A100 with 40GB+ memory)
- Git for version control
- Basic understanding of deep learning and state space models

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/hardware-data-parameter-codesign.git
   cd hardware-data-parameter-codesign
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Verify Installation**
   ```bash
   python demo_full_scale_validation.py --quick-test
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **Feature Enhancements**: Improve existing functionality
3. **New Features**: Add new capabilities to the framework
4. **Documentation**: Improve or add documentation
5. **Research Extensions**: Extend the research with new techniques
6. **Performance Optimizations**: Improve computational efficiency
7. **Testing**: Add or improve test coverage

### Research Contributions

This is a research project, so we particularly welcome:

- **Novel Optimization Techniques**: New methods for hardware-data-parameter co-design
- **Experimental Validation**: Additional experiments or datasets
- **Theoretical Analysis**: Mathematical analysis of the framework
- **Benchmarking**: Comparisons with other methods
- **Hardware Extensions**: Support for new hardware platforms

## Pull Request Process

### Before Submitting

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create an Issue**: For significant changes, create an issue first
3. **Branch Naming**: Use descriptive branch names
   - `feature/new-optimization-technique`
   - `bugfix/memory-leak-in-sdm`
   - `docs/improve-installation-guide`

### Submission Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Run validation
   python scripts/run_validation_suite.py --quick
   
   # Check code style
   black --check .
   flake8 .
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new optimization technique for CSP"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Requirements

- **Clear Description**: Explain what changes you made and why
- **Testing**: Include tests for new functionality
- **Documentation**: Update relevant documentation
- **Performance**: For performance changes, include benchmarks
- **Research**: For research contributions, include experimental results

### Commit Message Format

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `research`

Examples:
- `feat(csp): add dynamic sparsity pattern detection`
- `fix(sdm): resolve memory leak in matrix operations`
- `research(validation): add comparison with baseline methods`

## Issue Reporting

### Bug Reports

Include the following information:
- **Environment**: OS, Python version, GPU type
- **Steps to Reproduce**: Clear steps to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full error messages and stack traces
- **Configuration**: Relevant configuration files

### Feature Requests

Include:
- **Problem Description**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Research Justification**: Why is this scientifically valuable?
- **Implementation Ideas**: Any thoughts on implementation

### Research Questions

For research-related discussions:
- **Research Context**: Background and motivation
- **Hypothesis**: What you want to investigate
- **Methodology**: Proposed experimental approach
- **Expected Impact**: How this advances the field

## Code Style

### Python Style

- **Formatter**: Use Black with line length 100
- **Linter**: Use Flake8 for style checking
- **Import Sorting**: Use isort with Black profile
- **Type Hints**: Add type hints for public APIs

### Code Organization

- **Modularity**: Keep functions and classes focused
- **Documentation**: Add docstrings for all public functions
- **Comments**: Explain complex algorithms and research concepts
- **Constants**: Use uppercase for constants

### Research Code Standards

- **Reproducibility**: Ensure experiments are reproducible
- **Configuration**: Use YAML configs for hyperparameters
- **Logging**: Add comprehensive logging for experiments
- **Metrics**: Include statistical significance testing

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Validation Tests**: Test research hypotheses
4. **Performance Tests**: Test computational efficiency

### Running Tests

```bash
# All tests
pytest

# Specific test category
pytest -m unit
pytest -m integration
pytest -m slow

# With coverage
pytest --cov=models --cov=scripts --cov=utils
```

### Writing Tests

- **Test Names**: Use descriptive names
- **Test Data**: Use small, synthetic datasets for unit tests
- **Fixtures**: Use pytest fixtures for common setup
- **Mocking**: Mock external dependencies

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and inline comments
2. **API Documentation**: Function and class documentation
3. **Research Documentation**: Experimental methodology
4. **User Documentation**: Installation and usage guides

### Documentation Standards

- **Docstring Format**: Use Google-style docstrings
- **Examples**: Include usage examples
- **Mathematical Notation**: Use LaTeX for equations
- **References**: Cite relevant papers and resources

## Development Workflow

### Typical Workflow

1. **Issue Discussion**: Discuss the change in an issue
2. **Design Review**: For significant changes, discuss design
3. **Implementation**: Implement the change
4. **Testing**: Add comprehensive tests
5. **Documentation**: Update documentation
6. **Review**: Submit PR for review
7. **Integration**: Merge after approval

### Release Process

- **Semantic Versioning**: Follow semver (major.minor.patch)
- **Changelog**: Update CHANGELOG.md
- **Testing**: Run full validation suite
- **Documentation**: Update version-specific docs

## Research Ethics

### Reproducibility

- **Seed Setting**: Use fixed seeds for reproducible results
- **Environment**: Document exact environment specifications
- **Data**: Provide clear data preparation steps
- **Code**: Ensure code can be run by others

### Attribution

- **Citations**: Properly cite related work
- **Acknowledgments**: Acknowledge contributors
- **Licensing**: Respect software licenses
- **Data**: Follow data usage guidelines

## Getting Help

### Resources

- **Documentation**: Check README.md and docs/
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for feedback on draft PRs

### Contact

For questions about contributing:
- Create an issue for technical questions
- Use discussions for general questions
- Email maintainers for sensitive issues

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **Research Papers**: Co-authorship for significant research contributions
- **Acknowledgments**: Recognition in project documentation

Thank you for contributing to advancing the state of hardware-data-parameter co-design research! 