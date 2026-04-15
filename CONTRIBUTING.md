# Contributing Guide

Thank you for your interest in Fanjing Face Recognition! This project is part of the [FlowElement](https://github.com/FlowElement) open source ecosystem.

For general contribution guidelines (DCO signing, PR process, etc.), please refer to the [FlowElement Contributing Guide](https://github.com/FlowElement/m_flow/blob/main/CONTRIBUTING.md).

Below are development instructions specific to this project.

## Setting Up Development Environment

```bash
git clone <repo-url>
cd fanjing-face-recognition
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-training.txt
```

## Code Style

- Follow PEP 8, line width 120 characters
- Use type annotations (`typing` module)
- Variable names use snake_case, class names use PascalCase

## Reporting Issues

- Feature bugs: Submit via GitHub Issues
- Security vulnerabilities: Please refer to [FlowElement Security Policy](https://github.com/FlowElement/m_flow/blob/main/SECURITY.md)
