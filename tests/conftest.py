"""Shared fixtures for Prometheus tests."""

import pytest
from pathlib import Path
from prometheus.core import ContextEngine, ContextConfig, ContextMode


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def basic_config(tmp_workspace):
    """Create a basic ContextConfig for testing."""
    return ContextConfig(
        workspace_dir=tmp_workspace,
        mode=ContextMode.MANUAL,
        max_context_tokens=10000,
        enable_compression=False,
    )


@pytest.fixture
def engine(basic_config):
    """Create a ContextEngine for testing."""
    return ContextEngine(basic_config)
