Contributing to 0xplorer MEV Bot

Thank you for considering contributing to the 0xplorer MEV Bot project! We welcome contributions in the form of bug reports, feature suggestions, code improvements, documentation enhancements, and more.

This document provides guidelines to help you contribute effectively and maintain a high-quality codebase.
Table of Contents

    Code of Conduct
    Getting Started
        Prerequisites
        Setting Up the Development Environment
    How to Contribute
        Reporting Bugs
        Suggesting Enhancements
        Pull Requests
    Coding Guidelines
        Style Guide
        Commit Messages
        Branching Model
    Testing
    Documentation
    Issue Tracker
    Community and Support
    License

Code of Conduct

By participating in this project, you agree to abide by the Contributor Covenant Code of Conduct. Please be respectful and considerate in your interactions with others.
Getting Started
Prerequisites

    Python 3.8 or higher
    Git: Version control system
    Ethereum Node: Access to a fully synchronized Ethereum node (e.g., Geth, Nethermind)
    API Keys: For Infura, Etherscan, CoinGecko, CoinMarketCap, and CryptoCompare
    Wallet and Private Key: For testing and signing transactions

Setting Up the Development Environment

    Fork the Repository

    Click on the "Fork" button at the top right of the repository page to create your own fork.

    Clone Your Fork

    bash

git clone https://github.com/yourusername/0xplorer.git
cd 0xplorer

Set Upstream Remote

bash

git remote add upstream https://github.com/original_author/0xplorer.git

Create a Virtual Environment

bash

python3 -m venv venv
source venv/bin/activate  # For Windows use `venv\Scripts\activate`

Install Dependencies

bash

pip install -r requirements.txt

Configure Environment Variables

Copy the example environment file and customize it:

bash

cp .env-example .env

    Fill in your API keys and configuration settings in the .env file.
    Ensure all paths and addresses are correct.

Install Pre-commit Hooks

We use pre-commit hooks to enforce code style and catch errors early.

bash

    pip install pre-commit
    pre-commit install

How to Contribute
Reporting Bugs

If you find a bug, please open an issue on the GitHub repository.

Before Submitting a Bug Report:

    Search Existing Issues: To avoid duplicates, please check if the issue has already been reported.
    Use a Clear and Descriptive Title: Summarize the problem in the title.

Bug Report Content:

    Description: A clear and concise description of the problem.
    Steps to Reproduce: Detailed steps to reproduce the issue.
    Expected Behavior: What you expected to happen.
    Actual Behavior: What actually happened.
    Screenshots or Logs: If applicable, include error messages, stack traces, or screenshots.
    Environment: Include details about your setup, such as operating system, Python version, and Ethereum client.

Suggesting Enhancements

We welcome suggestions for new features or improvements.

Before Submitting a Feature Request:

    Check Existing Issues and Pull Requests: The feature may already be under discussion.

Feature Request Content:

    Description: A clear and concise description of the proposed enhancement.
    Motivation: Explain why this feature would be useful.
    Alternatives: Mention any alternative solutions you've considered.

Pull Requests

We appreciate your contributions! To submit a pull request (PR):

    Create a Branch

    Use a descriptive name for your branch:

    bash

git checkout -b feature/your-feature-name

Make Changes

    Write clear, maintainable code.
    Include comments and docstrings where necessary.
    Ensure your changes do not break existing functionality.

Write Tests

    Add unit tests for new features or bug fixes.
    Ensure all tests pass before submitting.

Commit Your Changes

    Follow the commit message guidelines below.
    Make small, incremental commits.

Push to Your Fork

bash

    git push origin feature/your-feature-name

    Create a Pull Request
        Go to your fork on GitHub and click "New pull request".
        Ensure the PR is against the correct base branch (usually main or develop).
        Provide a clear description of your changes.

    Address Feedback
        Be responsive to comments and requested changes.
        Update your PR with improvements as needed.

Coding Guidelines
Style Guide

    PEP 8: Follow the Python Enhancement Proposal 8 style guidelines.
    Type Hints: Use type annotations for function signatures and variables.
    Imports: Organize imports using isort and group them logically.
    Line Length: Limit lines to a maximum of 88 characters.
    Naming Conventions: Use descriptive and consistent naming for variables, functions, and classes.

Commit Messages

    Format: Use the Conventional Commits specification.

    Structure:

    swift

type(scope): subject

body (optional)

footer (optional)

Types:

    feat: A new feature
    fix: A bug fix
    docs: Documentation changes
    style: Code style changes (formatting, missing semi-colons, etc.)
    refactor: Code changes that neither fix a bug nor add a feature
    test: Adding or updating tests
    chore: Maintenance tasks

Example:

sql

    feat(strategy): add new arbitrage strategy for Uniswap

    Implement a new arbitrage strategy that detects price discrepancies between Uniswap and Sushiswap.

Branching Model

    Feature Branches: Use feature/feature-name for new features.
    Bug Fixes: Use fix/issue-number for bug fixes.
    Develop Branch: Merge your feature branches into develop.
    Main Branch: Stable code ready for release.

Testing

    Unit Tests: Write unit tests for new code using unittest or pytest.

    Test Coverage: Aim for high test coverage, especially for critical components.

    Running Tests:

    bash

    pytest tests/

    Continuous Integration: Ensure your changes pass all CI checks.

Documentation

    Docstrings: Include docstrings for all modules, classes, and functions using the Google style or reStructuredText.
    README and Guides: Update the README.md or other documentation files if your changes affect them.
    Comments: Write clear comments where necessary to explain complex logic.

Issue Tracker

Use the GitHub issue tracker to:

    Report bugs
    Request features
    Ask questions
    Discuss improvements

Labels: Use appropriate labels to categorize issues.
Community and Support

    Discussions: Participate in discussions on GitHub.
    Slack/Discord: Join our community channels (if available).
    Respectful Communication: Be respectful and considerate in all interactions.

License

By contributing to 0xplorer, you agree that your contributions will be licensed under the MIT License.