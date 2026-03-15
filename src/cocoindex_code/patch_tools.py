"""Patch tools for the cocoindex-code MCP server.

Provides apply_patch tool for applying unified diff patches to files
in the codebase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .filesystem_tools import (
    MAX_WRITE_BYTES,
    _root,
    _safe_resolve,
)

# === Internal data structures ===


@dataclass
class PatchHunk:
    """A single hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class PatchFile:
    """Parsed patch data for a single file."""

    old_path: str
    new_path: str
    hunks: list[PatchHunk] = field(default_factory=list)


# === Pydantic result models ===


class PatchFileResult(BaseModel):
    """Result for a single file in a patch."""

    path: str = Field(description="Relative file path")
    hunks_applied: int = Field(default=0, description="Hunks applied")
    hunks_rejected: int = Field(
        default=0, description="Hunks that failed to apply"
    )
    created: bool = Field(
        default=False, description="Whether file was newly created"
    )


class ApplyPatchResult(BaseModel):
    """Result from apply_patch tool."""

    success: bool
    files: list[PatchFileResult] = Field(default_factory=list)
    total_applied: int = 0
    total_rejected: int = 0
    dry_run: bool = True
    message: str | None = None


# === Unified diff parser ===

_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


def _parse_unified_diff(patch_text: str) -> list[PatchFile]:
    """Parse a unified diff into structured PatchFile objects."""
    files: list[PatchFile] = []
    lines = patch_text.splitlines(keepends=True)
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for file header
        if line.startswith("--- "):
            if i + 1 >= len(lines):
                break
            next_line = lines[i + 1]
            if not next_line.startswith("+++ "):
                i += 1
                continue

            old_path = line[4:].strip()
            new_path = next_line[4:].strip()

            # Strip a/ b/ prefixes
            if old_path.startswith("a/"):
                old_path = old_path[2:]
            if new_path.startswith("b/"):
                new_path = new_path[2:]

            pf = PatchFile(old_path=old_path, new_path=new_path)
            i += 2

            # Parse hunks for this file
            while i < len(lines):
                hunk_line = lines[i]
                m = _HUNK_HEADER.match(hunk_line)
                if m is None:
                    # Check if next file starts
                    if hunk_line.startswith("--- "):
                        break
                    if hunk_line.startswith("diff "):
                        break
                    i += 1
                    continue

                old_start = int(m.group(1))
                old_count = int(m.group(2) or "1")
                new_start = int(m.group(3))
                new_count = int(m.group(4) or "1")

                hunk = PatchHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                )
                i += 1

                # Collect hunk lines
                while i < len(lines):
                    hl = lines[i]
                    # Stop if we hit a new file header
                    if hl.startswith("--- ") or hl.startswith("diff "):
                        break
                    if _HUNK_HEADER.match(hl):
                        break
                    if hl.startswith(("+", "-", " ")):
                        hunk.lines.append(hl.rstrip("\n\r"))
                        i += 1
                    elif hl.startswith("\\"):
                        # "\ No newline at end of file"
                        i += 1
                    else:
                        break

                pf.hunks.append(hunk)

            files.append(pf)
        else:
            i += 1

    return files


# === Hunk application ===


def _apply_hunks(
    content: str, hunks: list[PatchHunk],
) -> tuple[str, int, int]:
    """Apply hunks to file content.

    Returns (new_content, applied_count, rejected_count).
    """
    file_lines = content.splitlines(keepends=True)
    applied = 0
    rejected = 0

    # Apply hunks in reverse order to preserve line numbers
    for hunk in reversed(hunks):
        old_lines: list[str] = []
        new_lines: list[str] = []

        for hl in hunk.lines:
            if hl.startswith("-"):
                old_lines.append(hl[1:])
            elif hl.startswith("+"):
                new_lines.append(hl[1:])
            elif hl.startswith(" "):
                old_lines.append(hl[1:])
                new_lines.append(hl[1:])

        # Verify context matches (old lines)
        start_idx = hunk.old_start - 1  # 0-indexed
        match = True

        if start_idx < 0 or start_idx + len(old_lines) > len(file_lines):
            match = False
        else:
            for j, expected in enumerate(old_lines):
                actual = file_lines[start_idx + j].rstrip("\n\r")
                if actual != expected:
                    match = False
                    break

        if match:
            # Replace old lines with new lines
            replacement = [ln + "\n" for ln in new_lines]
            file_lines[start_idx:start_idx + len(old_lines)] = (
                replacement
            )
            applied += 1
        else:
            rejected += 1

    return "".join(file_lines), applied, rejected


def _apply_patch_impl(
    patch_text: str,
    root: Path,
    dry_run: bool = True,
) -> ApplyPatchResult:
    """Apply a unified diff patch."""
    try:
        patch_files = _parse_unified_diff(patch_text)
    except Exception as e:
        return ApplyPatchResult(
            success=False,
            message=f"Failed to parse patch: {e!s}",
        )

    if not patch_files:
        return ApplyPatchResult(
            success=False,
            message="No files found in patch",
        )

    results: list[PatchFileResult] = []
    total_applied = 0
    total_rejected = 0

    for pf in patch_files:
        target_path = pf.new_path
        is_new = pf.old_path == "/dev/null"
        is_delete = pf.new_path == "/dev/null"

        if is_delete:
            target_path = pf.old_path

        try:
            resolved = _safe_resolve(target_path)
        except ValueError:
            results.append(PatchFileResult(
                path=target_path,
                hunks_rejected=len(pf.hunks),
            ))
            total_rejected += len(pf.hunks)
            continue

        if is_new:
            # New file: collect all + lines
            new_content = ""
            for hunk in pf.hunks:
                for hl in hunk.lines:
                    if hl.startswith("+"):
                        new_content += hl[1:] + "\n"

            if not dry_run:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                content_bytes = new_content.encode("utf-8")
                if len(content_bytes) > MAX_WRITE_BYTES:
                    results.append(PatchFileResult(
                        path=target_path,
                        hunks_rejected=len(pf.hunks),
                    ))
                    total_rejected += len(pf.hunks)
                    continue
                resolved.write_text(new_content, encoding="utf-8")

            results.append(PatchFileResult(
                path=target_path,
                hunks_applied=len(pf.hunks),
                created=True,
            ))
            total_applied += len(pf.hunks)
            continue

        if not resolved.is_file():
            results.append(PatchFileResult(
                path=target_path,
                hunks_rejected=len(pf.hunks),
            ))
            total_rejected += len(pf.hunks)
            continue

        try:
            content = resolved.read_text(
                encoding="utf-8", errors="replace",
            )
        except OSError:
            results.append(PatchFileResult(
                path=target_path,
                hunks_rejected=len(pf.hunks),
            ))
            total_rejected += len(pf.hunks)
            continue

        new_content, app, rej = _apply_hunks(content, pf.hunks)

        if not dry_run and app > 0:
            content_bytes = new_content.encode("utf-8")
            if len(content_bytes) > MAX_WRITE_BYTES:
                results.append(PatchFileResult(
                    path=target_path,
                    hunks_rejected=len(pf.hunks),
                ))
                total_rejected += len(pf.hunks)
                continue
            resolved.write_text(new_content, encoding="utf-8")

        results.append(PatchFileResult(
            path=target_path,
            hunks_applied=app,
            hunks_rejected=rej,
        ))
        total_applied += app
        total_rejected += rej

    return ApplyPatchResult(
        success=total_rejected == 0,
        files=results,
        total_applied=total_applied,
        total_rejected=total_rejected,
        dry_run=dry_run,
    )


# === MCP tool registration ===


def register_patch_tools(mcp: FastMCP) -> None:
    """Register patch tools on the MCP server."""

    @mcp.tool(
        name="apply_patch",
        description=(
            "Apply a unified diff patch to one or more files."
            " Accepts standard unified diff format (as produced by"
            " 'git diff' or 'diff -u')."
            " Defaults to dry_run=true so you can preview which hunks"
            " would be applied or rejected before committing changes."
            " Set dry_run=false to actually modify files."
            " Supports new file creation, multi-file patches,"
            " and multi-hunk patches."
        ),
    )
    async def apply_patch(
        patch: str = Field(
            description=(
                "Unified diff text. Must include --- / +++ headers"
                " and @@ hunk markers."
            ),
        ),
        dry_run: bool = Field(
            default=True,
            description=(
                "Preview changes without applying."
                " Set to false to apply the patch."
            ),
        ),
    ) -> ApplyPatchResult:
        """Apply a unified diff patch."""
        try:
            return _apply_patch_impl(
                patch, _root(), dry_run=dry_run,
            )
        except Exception as e:
            return ApplyPatchResult(
                success=False,
                message=f"apply_patch failed: {e!s}",
            )
