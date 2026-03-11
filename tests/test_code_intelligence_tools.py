"""Tests for code intelligence tools."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from cocoindex_code.code_intelligence_tools import (
    _classify_usage,
    _compute_metrics,
    _extract_symbols,
    _find_definitions_impl,
    _find_references_impl,
    _rename_symbol_impl,
    _walk_source_files,
)


@pytest.fixture()
def sample_codebase(tmp_path: Path) -> Path:
    """Create a sample codebase for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "utils").mkdir()
    (tmp_path / "lib").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "__pycache__").mkdir()

    (tmp_path / "main.py").write_text(
        "MAX_RETRIES = 3\n"
        "\n"
        "class UserManager:\n"
        '    """Manages users."""\n'
        "\n"
        "    def __init__(self):\n"
        "        self.users = []\n"
        "\n"
        "    def add_user(self, name):\n"
        "        self.users.append(name)\n"
        "\n"
        "    async def fetch_user(self, user_id):\n"
        "        pass\n"
        "\n"
        "\n"
        "def helper():\n"
        "    manager = UserManager()\n"
        "    manager.add_user('alice')\n"
    )

    (tmp_path / "src" / "app.ts").write_text(
        "export function greet(name: string): string {\n"
        "  return `Hello, ${name}!`;\n"
        "}\n"
        "\n"
        "export class Greeter {\n"
        "  private name: string;\n"
        "\n"
        "  constructor(name: string) {\n"
        "    this.name = name;\n"
        "  }\n"
        "\n"
        "  greet(): string {\n"
        "    return greet(this.name);\n"
        "  }\n"
        "}\n"
        "\n"
        "export const DEFAULT_NAME = 'World';\n"
    )

    (tmp_path / "src" / "utils" / "math.ts").write_text(
        "export const add = (a: number, b: number): number => a + b;\n"
        "export const subtract = (a: number, b: number): number => a - b;\n"
    )

    (tmp_path / "lib" / "database.py").write_text(
        "import sqlite3\n"
        "\n"
        "class DatabaseConnection:\n"
        '    """Database connection manager."""\n'
        "\n"
        "    def connect(self) -> None:\n"
        "        pass\n"
        "\n"
        "    def query(self, sql: str):\n"
        "        pass\n"
    )

    (tmp_path / "lib" / "server.rs").write_text(
        "pub async fn start_server(port: u16) -> Result<(), Error> {\n"
        "    let listener = TcpListener::bind(port).await?;\n"
        "    Ok(())\n"
        "}\n"
        "\n"
        "pub struct Config {\n"
        "    pub host: String,\n"
        "    pub port: u16,\n"
        "}\n"
        "\n"
        "impl Config {\n"
        "    pub fn new() -> Self {\n"
        "        Config { host: String::new(), port: 8080 }\n"
        "    }\n"
        "}\n"
    )

    (tmp_path / "lib" / "handler.go").write_text(
        "package main\n"
        "\n"
        "func HandleRequest(w http.ResponseWriter, r *http.Request) {\n"
        "    w.Write([]byte(\"OK\"))\n"
        "}\n"
        "\n"
        "type Server struct {\n"
        "    Port int\n"
        "}\n"
        "\n"
        "func (s *Server) Start() error {\n"
        "    return nil\n"
        "}\n"
    )

    (tmp_path / "README.md").write_text("# Test Project\n\nA test project.\n")

    (tmp_path / "node_modules" / "pkg.js").write_text("module.exports = {};\n")
    (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(
        b"\x00" * 100
    )

    binary_path = tmp_path / "image.png"
    binary_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00" + b"\x00" * 50
    )

    return tmp_path


@pytest.fixture(autouse=True)
def _patch_config(sample_codebase: Path) -> Iterator[None]:
    """Patch config to point at sample_codebase."""
    with patch(
        "cocoindex_code.filesystem_tools.config"
    ) as mock_fs_config, patch(
        "cocoindex_code.code_intelligence_tools._root"
    ) as mock_root, patch(
        "cocoindex_code.code_intelligence_tools._safe_resolve"
    ) as mock_resolve, patch(
        "cocoindex_code.code_intelligence_tools._relative"
    ) as mock_relative:
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

        def relative_side_effect(path):
            try:
                return str(path.relative_to(sample_codebase))
            except ValueError:
                return str(path)

        mock_relative.side_effect = relative_side_effect
        yield


# === Tests for _extract_symbols ===


class TestExtractSymbols:
    def test_python_functions_and_classes(self) -> None:
        content = (
            "def hello():\n"
            "    pass\n"
            "\n"
            "class Foo:\n"
            "    def method(self):\n"
            "        pass\n"
        )
        symbols = _extract_symbols(content, "python")
        names = [s.name for s in symbols]
        assert "hello" in names
        assert "Foo" in names
        assert "method" in names
        # method should be classified as method
        method_sym = next(s for s in symbols if s.name == "method")
        assert method_sym.symbol_type == "method"
        # hello should be function
        hello_sym = next(s for s in symbols if s.name == "hello")
        assert hello_sym.symbol_type == "function"

    def test_python_constants(self) -> None:
        content = "MAX_SIZE = 100\nPI = 3.14\n"
        symbols = _extract_symbols(content, "python")
        names = [s.name for s in symbols]
        assert "MAX_SIZE" in names
        assert "PI" in names
        for s in symbols:
            assert s.symbol_type == "constant"

    def test_python_async_function(self) -> None:
        content = "async def fetch_data():\n    pass\n"
        symbols = _extract_symbols(content, "python")
        assert len(symbols) == 1
        assert symbols[0].name == "fetch_data"
        assert symbols[0].symbol_type == "function"

    def test_typescript_interface_and_enum(self) -> None:
        content = (
            "export interface User {\n"
            "  name: string;\n"
            "}\n"
            "\n"
            "export type ID = string;\n"
            "\n"
            "export enum Color {\n"
            "  Red, Green, Blue\n"
            "}\n"
        )
        symbols = _extract_symbols(content, "typescript")
        names = [s.name for s in symbols]
        assert "User" in names
        assert "ID" in names
        assert "Color" in names
        user = next(s for s in symbols if s.name == "User")
        assert user.symbol_type == "interface"
        color = next(s for s in symbols if s.name == "Color")
        assert color.symbol_type == "enum"

    def test_javascript_functions_and_classes(self) -> None:
        content = (
            "export function greet(name) {\n"
            "  return name;\n"
            "}\n"
            "export class App {}\n"
            "const VERSION = '1.0';\n"
        )
        symbols = _extract_symbols(content, "javascript")
        names = [s.name for s in symbols]
        assert "greet" in names
        assert "App" in names
        assert "VERSION" in names

    def test_rust_symbols(self) -> None:
        content = (
            "pub async fn serve(port: u16) {}\n"
            "pub struct Config { port: u16 }\n"
            "pub enum Status { Ok, Error }\n"
            "pub trait Handler {}\n"
            "mod tests {}\n"
            "impl Config {}\n"
            "const MAX: u32 = 100;\n"
        )
        symbols = _extract_symbols(content, "rust")
        names = [s.name for s in symbols]
        assert "serve" in names
        assert "Config" in names
        assert "Status" in names
        assert "Handler" in names
        assert "tests" in names
        assert "MAX" in names

    def test_go_symbols(self) -> None:
        content = (
            "func HandleRequest(w http.ResponseWriter) {\n"
            "}\n"
            "type Server struct {\n"
            "    Port int\n"
            "}\n"
            "func (s *Server) Start() error {\n"
            "    return nil\n"
            "}\n"
            "const MaxRetries = 3\n"
        )
        symbols = _extract_symbols(content, "go")
        names = [s.name for s in symbols]
        assert "HandleRequest" in names
        assert "Server" in names
        assert "Start" in names
        assert "MaxRetries" in names

    def test_unknown_language(self) -> None:
        symbols = _extract_symbols("hello world", "brainfuck")
        assert symbols == []

    def test_empty_content(self) -> None:
        symbols = _extract_symbols("", "python")
        assert symbols == []

    def test_end_line_computation(self) -> None:
        content = (
            "def foo():\n"
            "    pass\n"
            "\n"
            "def bar():\n"
            "    x = 1\n"
            "    return x\n"
        )
        symbols = _extract_symbols(content, "python")
        foo = next(s for s in symbols if s.name == "foo")
        bar = next(s for s in symbols if s.name == "bar")
        assert foo.end_line == 3  # before bar starts
        assert bar.end_line == 6  # EOF


# === Tests for _walk_source_files ===


class TestWalkSourceFiles:
    def test_walks_all_source_files(
        self, sample_codebase: Path,
    ) -> None:
        files = _walk_source_files(sample_codebase)
        rel_paths = [rel for _, rel, _ in files]
        assert any("main.py" in p for p in rel_paths)
        assert any("app.ts" in p for p in rel_paths)
        # Excluded dirs
        assert not any("node_modules" in p for p in rel_paths)
        assert not any("__pycache__" in p for p in rel_paths)
        # Binary files excluded
        assert not any("image.png" in p for p in rel_paths)

    def test_language_filter(
        self, sample_codebase: Path,
    ) -> None:
        files = _walk_source_files(
            sample_codebase, languages=["python"],
        )
        for _, _, lang in files:
            assert lang == "python"

    def test_path_filter(
        self, sample_codebase: Path,
    ) -> None:
        files = _walk_source_files(
            sample_codebase, paths=["src/*"],
        )
        for _, rel, _ in files:
            assert rel.startswith("src/") or rel.startswith("src\\")


# === Tests for _classify_usage ===


class TestClassifyUsage:
    def test_import(self) -> None:
        assert _classify_usage(
            "from foo import bar", "bar", "python",
        ) == "import"
        assert _classify_usage(
            "import os", "os", "python",
        ) == "import"

    def test_call(self) -> None:
        assert _classify_usage(
            "result = helper()", "helper", "python",
        ) == "call"

    def test_assignment(self) -> None:
        assert _classify_usage(
            "helper = something", "helper", "python",
        ) == "assignment"

    def test_type_annotation(self) -> None:
        assert _classify_usage(
            "x: UserManager = None", "UserManager", "python",
        ) == "type_annotation"

    def test_definition(self) -> None:
        assert _classify_usage(
            "def helper():", "helper", "python",
        ) == "definition"
        assert _classify_usage(
            "class UserManager:", "UserManager", "python",
        ) == "definition"

    def test_other(self) -> None:
        assert _classify_usage(
            "print(helper)", "helper", "python",
        ) == "other"


# === Tests for _find_definitions_impl ===


class TestFindDefinitions:
    def test_find_python_function(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "helper", sample_codebase,
        )
        assert len(defs) >= 1
        assert any(d.name == "helper" for d in defs)

    def test_find_python_class(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "UserManager", sample_codebase,
        )
        assert len(defs) >= 1
        assert defs[0].symbol_type == "class"

    def test_find_typescript_function(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "greet", sample_codebase,
        )
        assert len(defs) >= 1
        assert any(
            d.file_path.endswith("app.ts") for d in defs
        )

    def test_find_rust_function(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "start_server", sample_codebase,
        )
        assert len(defs) >= 1

    def test_find_go_function(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "HandleRequest", sample_codebase,
        )
        assert len(defs) >= 1

    def test_no_match(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "nonexistent_symbol_xyz", sample_codebase,
        )
        assert len(defs) == 0

    def test_filter_by_type(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "UserManager", sample_codebase,
            symbol_type="function",
        )
        assert len(defs) == 0

    def test_filter_by_language(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "greet", sample_codebase,
            languages=["python"],
        )
        # greet is in typescript, not python
        assert len(defs) == 0

    def test_limit(
        self, sample_codebase: Path,
    ) -> None:
        defs = _find_definitions_impl(
            "helper", sample_codebase, limit=1,
        )
        assert len(defs) <= 1


# === Tests for _find_references_impl ===


class TestFindReferences:
    def test_find_references_to_symbol(
        self, sample_codebase: Path,
    ) -> None:
        refs, total, searched, trunc = _find_references_impl(
            "UserManager", sample_codebase,
        )
        assert total >= 2  # class def + usage in helper()

    def test_word_boundary(
        self, sample_codebase: Path,
    ) -> None:
        # "add" should match add_user method AND add const
        refs, total, _, _ = _find_references_impl(
            "add", sample_codebase,
        )
        # Should NOT match "add_user" since \badd\b won't match inside
        for ref in refs:
            # Each match should contain "add" as a word
            assert "add" in ref.line

    def test_context_lines(
        self, sample_codebase: Path,
    ) -> None:
        refs, _, _, _ = _find_references_impl(
            "UserManager", sample_codebase,
            context_lines=2,
        )
        if refs:
            # At least one ref should have context
            has_context = any(
                ref.context_before or ref.context_after
                for ref in refs
            )
            assert has_context

    def test_language_filter(
        self, sample_codebase: Path,
    ) -> None:
        refs, _, _, _ = _find_references_impl(
            "greet", sample_codebase,
            languages=["typescript"],
        )
        for ref in refs:
            assert ref.path.endswith(".ts")

    def test_truncation(
        self, sample_codebase: Path,
    ) -> None:
        refs, total, _, trunc = _find_references_impl(
            "UserManager", sample_codebase, limit=1,
        )
        assert len(refs) <= 1

    def test_usage_type_classification(
        self, sample_codebase: Path,
    ) -> None:
        refs, _, _, _ = _find_references_impl(
            "sqlite3", sample_codebase,
        )
        import_refs = [
            r for r in refs if r.usage_type == "import"
        ]
        assert len(import_refs) >= 1


# === Tests for _compute_metrics ===


class TestComputeMetrics:
    def test_basic_metrics(self) -> None:
        content = (
            "# A comment\n"
            "\n"
            "def foo():\n"
            "    pass\n"
            "\n"
            "def bar():\n"
            "    x = 1\n"
            "    if x > 0:\n"
            "        return x\n"
            "    return 0\n"
        )
        m = _compute_metrics(content, "python")
        assert m.total_lines == 10
        assert m.blank_lines == 2
        assert m.comment_lines == 1
        assert m.code_lines == 7
        assert m.functions == 2
        assert m.complexity_estimate >= 1  # at least the if

    def test_empty_file(self) -> None:
        m = _compute_metrics("", "python")
        assert m.total_lines == 0
        assert m.functions == 0
        assert m.classes == 0

    def test_nesting_depth(self) -> None:
        content = (
            "def foo():\n"
            "    if True:\n"
            "        for i in range(10):\n"
            "            if i > 5:\n"
            "                print(i)\n"
        )
        m = _compute_metrics(content, "python")
        assert m.max_nesting_depth >= 4

    def test_class_count(self) -> None:
        content = (
            "class Foo:\n"
            "    pass\n"
            "\n"
            "class Bar:\n"
            "    pass\n"
        )
        m = _compute_metrics(content, "python")
        assert m.classes == 2

    def test_unknown_language(self) -> None:
        content = "hello world\n"
        m = _compute_metrics(content, "unknown")
        assert m.total_lines == 1
        assert m.functions == 0


# === Tests for _rename_symbol_impl ===


class TestRenameSymbol:
    def test_dry_run_preview(
        self, sample_codebase: Path,
    ) -> None:
        result = _rename_symbol_impl(
            "UserManager", "AccountManager",
            sample_codebase, dry_run=True,
        )
        assert result.success
        assert result.dry_run
        assert result.total_replacements >= 2
        assert result.files_changed >= 1
        # File should NOT be modified
        content = (sample_codebase / "main.py").read_text()
        assert "UserManager" in content

    def test_actual_rename(
        self, sample_codebase: Path,
    ) -> None:
        result = _rename_symbol_impl(
            "UserManager", "AccountManager",
            sample_codebase, dry_run=False,
        )
        assert result.success
        assert not result.dry_run
        assert result.total_replacements >= 2
        content = (sample_codebase / "main.py").read_text()
        assert "AccountManager" in content
        assert "UserManager" not in content

    def test_word_boundary_safety(
        self, sample_codebase: Path,
    ) -> None:
        # Renaming "add" should not affect "add_user"
        _rename_symbol_impl(
            "add", "sum_values",
            sample_codebase, dry_run=False,
        )
        content = (sample_codebase / "main.py").read_text()
        # add_user should still be intact
        assert "add_user" in content

    def test_same_name_error(
        self, sample_codebase: Path,
    ) -> None:
        result = _rename_symbol_impl(
            "foo", "foo", sample_codebase,
        )
        assert not result.success
        assert "identical" in (result.message or "")

    def test_invalid_name_error(
        self, sample_codebase: Path,
    ) -> None:
        result = _rename_symbol_impl(
            "foo", "invalid-name!", sample_codebase,
        )
        assert not result.success
        assert "valid identifier" in (result.message or "")

    def test_scope_filter(
        self, sample_codebase: Path,
    ) -> None:
        result = _rename_symbol_impl(
            "greet", "sayHello",
            sample_codebase,
            scope="src/**",
            dry_run=True,
        )
        assert result.success
        # Should only match files in src/
        for change in result.changes:
            assert change.file_path.startswith("src")

    def test_no_matches(
        self, sample_codebase: Path,
    ) -> None:
        result = _rename_symbol_impl(
            "nonexistent_xyz_abc", "new_name",
            sample_codebase, dry_run=True,
        )
        assert result.success
        assert result.total_replacements == 0
