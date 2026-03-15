"""Tests for patch tools: apply_patch."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from cocoindex_code.patch_tools import (
    PatchHunk,
    _apply_hunks,
    _apply_patch_impl,
    _parse_unified_diff,
)


@pytest.fixture()
def sample_codebase(tmp_path: Path) -> Path:
    """Create a sample codebase for testing."""
    (tmp_path / "src").mkdir()

    (tmp_path / "hello.py").write_text(
        "def hello():\n"
        "    print('Hello, world!')\n"
        "\n"
        "def goodbye():\n"
        "    print('Goodbye!')\n"
    )

    (tmp_path / "src" / "app.py").write_text(
        "class App:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    return tmp_path


@pytest.fixture(autouse=True)
def _patch_config(sample_codebase: Path) -> Iterator[None]:
    """Patch config for patch_tools."""
    with patch(
        "cocoindex_code.filesystem_tools.config"
    ) as mock_fs_config, patch(
        "cocoindex_code.patch_tools._root"
    ) as mock_root, patch(
        "cocoindex_code.patch_tools._safe_resolve"
    ) as mock_resolve:
        mock_fs_config.codebase_root_path = sample_codebase
        mock_root.return_value = sample_codebase

        def safe_resolve_side_effect(path_str):
            import os
            root = sample_codebase
            resolved = (root / path_str).resolve()
            if not (
                resolved == root
                or str(resolved).startswith(str(root) + os.sep)
            ):
                msg = f"Path '{path_str}' escapes the codebase root"
                raise ValueError(msg)
            return resolved

        mock_resolve.side_effect = safe_resolve_side_effect
        yield


# === Tests for _parse_unified_diff ===


class TestParseUnifiedDiff:
    def test_single_file_single_hunk(self) -> None:
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('Hello, world!')\n"
            "+    print('Hello, everyone!')\n"
        )
        files = _parse_unified_diff(diff)
        assert len(files) == 1
        assert files[0].old_path == "hello.py"
        assert files[0].new_path == "hello.py"
        assert len(files[0].hunks) == 1
        assert files[0].hunks[0].old_start == 1

    def test_multi_hunk(self) -> None:
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def hello():\n"
            "-    print('Hello, world!')\n"
            "+    print('Hi!')\n"
            "@@ -4,2 +4,2 @@\n"
            " def goodbye():\n"
            "-    print('Goodbye!')\n"
            "+    print('Bye!')\n"
        )
        files = _parse_unified_diff(diff)
        assert len(files) == 1
        assert len(files[0].hunks) == 2

    def test_multi_file(self) -> None:
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def hello():\n"
            "-    print('Hello, world!')\n"
            "+    print('Hi!')\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1,3 +1,3 @@\n"
            " class App:\n"
            "-    def run(self):\n"
            "+    def start(self):\n"
            "         pass\n"
        )
        files = _parse_unified_diff(diff)
        assert len(files) == 2

    def test_new_file(self) -> None:
        diff = (
            "--- /dev/null\n"
            "+++ b/new_file.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def new_func():\n"
            "+    pass\n"
        )
        files = _parse_unified_diff(diff)
        assert len(files) == 1
        assert files[0].old_path == "/dev/null"
        assert files[0].new_path == "new_file.py"

    def test_empty_patch(self) -> None:
        files = _parse_unified_diff("")
        assert files == []


# === Tests for _apply_hunks ===


class TestApplyHunks:
    def test_single_replacement(self) -> None:
        content = (
            "def hello():\n"
            "    print('Hello, world!')\n"
        )
        hunk = PatchHunk(
            old_start=1, old_count=2, new_start=1, new_count=2,
            lines=[
                " def hello():",
                "-    print('Hello, world!')",
                "+    print('Hello, everyone!')",
            ],
        )
        result, applied, rejected = _apply_hunks(content, [hunk])
        assert applied == 1
        assert rejected == 0
        assert "Hello, everyone!" in result

    def test_context_mismatch_rejects(self) -> None:
        content = "def foo():\n    pass\n"
        hunk = PatchHunk(
            old_start=1, old_count=2, new_start=1, new_count=2,
            lines=[
                " def bar():",  # doesn't match
                "-    pass",
                "+    return None",
            ],
        )
        result, applied, rejected = _apply_hunks(content, [hunk])
        assert applied == 0
        assert rejected == 1
        # Content unchanged
        assert result == content

    def test_multiple_hunks(self) -> None:
        content = (
            "line1\n"
            "line2\n"
            "line3\n"
            "line4\n"
            "line5\n"
        )
        hunk1 = PatchHunk(
            old_start=1, old_count=1, new_start=1, new_count=1,
            lines=["-line1", "+LINE1"],
        )
        hunk2 = PatchHunk(
            old_start=5, old_count=1, new_start=5, new_count=1,
            lines=["-line5", "+LINE5"],
        )
        result, applied, rejected = _apply_hunks(
            content, [hunk1, hunk2],
        )
        assert applied == 2
        assert rejected == 0
        assert "LINE1" in result
        assert "LINE5" in result


# === Tests for _apply_patch_impl ===


class TestApplyPatchImpl:
    def test_dry_run(self, sample_codebase: Path) -> None:
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def hello():\n"
            "-    print('Hello, world!')\n"
            "+    print('Hi!')\n"
        )
        result = _apply_patch_impl(diff, sample_codebase, dry_run=True)
        assert result.success
        assert result.dry_run
        assert result.total_applied == 1
        # File should be unchanged
        content = (sample_codebase / "hello.py").read_text()
        assert "Hello, world!" in content

    def test_apply(self, sample_codebase: Path) -> None:
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def hello():\n"
            "-    print('Hello, world!')\n"
            "+    print('Hi!')\n"
        )
        result = _apply_patch_impl(
            diff, sample_codebase, dry_run=False,
        )
        assert result.success
        assert result.total_applied == 1
        content = (sample_codebase / "hello.py").read_text()
        assert "Hi!" in content

    def test_new_file_creation(
        self, sample_codebase: Path,
    ) -> None:
        diff = (
            "--- /dev/null\n"
            "+++ b/new_file.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def new_func():\n"
            "+    pass\n"
        )
        result = _apply_patch_impl(
            diff, sample_codebase, dry_run=False,
        )
        assert result.success
        assert result.total_applied == 1
        new_file = sample_codebase / "new_file.py"
        assert new_file.exists()
        content = new_file.read_text()
        assert "def new_func():" in content

    def test_nonexistent_file(
        self, sample_codebase: Path,
    ) -> None:
        diff = (
            "--- a/missing.py\n"
            "+++ b/missing.py\n"
            "@@ -1,2 +1,2 @@\n"
            " foo\n"
            "-bar\n"
            "+baz\n"
        )
        result = _apply_patch_impl(
            diff, sample_codebase, dry_run=False,
        )
        assert not result.success
        assert result.total_rejected == 1

    def test_path_traversal_rejected(
        self, sample_codebase: Path,
    ) -> None:
        diff = (
            "--- a/../../etc/passwd\n"
            "+++ b/../../etc/passwd\n"
            "@@ -1,1 +1,1 @@\n"
            "-root\n"
            "+hacked\n"
        )
        result = _apply_patch_impl(
            diff, sample_codebase, dry_run=False,
        )
        assert result.total_rejected >= 1

    def test_empty_patch(
        self, sample_codebase: Path,
    ) -> None:
        result = _apply_patch_impl("", sample_codebase)
        assert not result.success
        assert "No files" in (result.message or "")

    def test_multi_file_patch(
        self, sample_codebase: Path,
    ) -> None:
        diff = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def hello():\n"
            "-    print('Hello, world!')\n"
            "+    print('Hi!')\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1,3 +1,3 @@\n"
            " class App:\n"
            "-    def run(self):\n"
            "+    def start(self):\n"
            "         pass\n"
        )
        result = _apply_patch_impl(
            diff, sample_codebase, dry_run=False,
        )
        assert result.success
        assert result.total_applied == 2
        assert len(result.files) == 2
