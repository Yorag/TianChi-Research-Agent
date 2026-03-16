"""Bridge module: exports run_agent() for agent.py and test_validation.py."""

try:
    from .react_agent import run_react_agent as run_agent
except ImportError:
    from react_agent import run_react_agent as run_agent

__all__ = ["run_agent"]
