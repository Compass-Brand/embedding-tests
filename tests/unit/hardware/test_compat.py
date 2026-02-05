"""Tests for PyTorch compatibility shims."""

from __future__ import annotations

import torch


class TestSetSubmoduleCompat:
    """Tests for set_submodule monkey-patch."""

    def test_set_submodule_exists_on_module(self) -> None:
        """set_submodule method should exist on torch.nn.Module after import."""
        # Import the compat module to apply the patch
        from embedding_tests.hardware import compat  # noqa: F401

        assert hasattr(torch.nn.Module, "set_submodule")

    def test_set_submodule_sets_direct_child(self) -> None:
        """set_submodule should set a direct child module."""
        from embedding_tests.hardware import compat  # noqa: F401

        parent = torch.nn.Module()
        child = torch.nn.Linear(10, 5)

        parent.set_submodule("child", child)

        assert hasattr(parent, "child")
        assert parent.child is child

    def test_set_submodule_sets_nested_child(self) -> None:
        """set_submodule should set a nested module via dot notation."""
        from embedding_tests.hardware import compat  # noqa: F401

        parent = torch.nn.Module()
        intermediate = torch.nn.Module()
        parent.add_module("layer", intermediate)
        leaf = torch.nn.Linear(10, 5)

        parent.set_submodule("layer.fc", leaf)

        assert hasattr(parent.layer, "fc")
        assert parent.layer.fc is leaf

    def test_set_submodule_replaces_existing(self) -> None:
        """set_submodule should replace an existing module."""
        from embedding_tests.hardware import compat  # noqa: F401

        parent = torch.nn.Module()
        old_child = torch.nn.Linear(10, 5)
        new_child = torch.nn.Linear(10, 10)
        parent.add_module("fc", old_child)

        parent.set_submodule("fc", new_child)

        assert parent.fc is new_child
        assert parent.fc is not old_child

    def test_set_submodule_is_idempotent(self) -> None:
        """Importing compat multiple times should not break anything."""
        from embedding_tests.hardware import compat  # noqa: F401
        from embedding_tests.hardware import compat as compat2  # noqa: F401, F811

        parent = torch.nn.Module()
        child = torch.nn.Linear(10, 5)
        parent.set_submodule("child", child)

        assert parent.child is child
