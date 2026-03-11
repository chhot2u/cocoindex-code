"""Tests for the plan_optimizer tool."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from cocoindex_code.thinking_tools import (
    PLAN_DIMENSIONS,
    ThinkingEngine,
    ThoughtData,
)


@pytest.fixture()
def thinking_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture(autouse=True)
def _patch_config(thinking_dir: Path) -> Iterator[None]:
    with (
        patch("cocoindex_code.thinking_tools.config") as mock_config,
        patch("cocoindex_code.thinking_tools._engine", None),
    ):
        mock_config.index_dir = thinking_dir
        yield


def _make_thought(
    thought: str = "t",
    thought_number: int = 1,
    total_thoughts: int = 10,
    next_thought_needed: bool = True,
) -> ThoughtData:
    return ThoughtData(
        thought=thought,
        thought_number=thought_number,
        total_thoughts=total_thoughts,
        next_thought_needed=next_thought_needed,
    )


SAMPLE_PLAN = """# Implementation Plan: Add User Authentication

## Phase 1: Database Schema
1. Create users table with email, password_hash, created_at
2. Add sessions table for JWT token tracking
3. Write migration scripts

## Phase 2: API Endpoints
1. POST /api/auth/register - validate input, hash password, create user
2. POST /api/auth/login - verify credentials, issue JWT
3. POST /api/auth/logout - invalidate session
4. GET /api/auth/me - return current user profile

## Phase 3: Middleware
1. Create auth middleware to verify JWT on protected routes
2. Add rate limiting to auth endpoints

## Phase 4: Testing
1. Unit tests for password hashing
2. Integration tests for auth endpoints
3. E2E test for login flow
"""

VAGUE_PLAN = """
Fix the authentication.
Make it work somehow.
Clean up the code and improve stuff.
Handle the edge cases etc.
Figure out the deployment.
"""

NO_STRUCTURE_PLAN = (
    "We need to add a new feature to the application.\n"
    "It should allow users to upload files.\n"
    "The files need to be stored somewhere.\n"
    "We also need to validate the files.\n"
    "Then we deploy it to production.\n"
)


class TestAntiPatternDetection:
    def test_detects_vague_language(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        patterns = engine._detect_anti_patterns(VAGUE_PLAN)
        vague = [
            p for p in patterns
            if p.pattern_type == "vague_language"
        ]
        assert len(vague) >= 3  # "make it work", "somehow", "stuff"

    def test_detects_todo_markers(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        plan = "Step 1: Create model\nStep 2: TODO implement validation\n"
        patterns = engine._detect_anti_patterns(plan)
        todo = [
            p for p in patterns
            if p.pattern_type == "todo_marker"
        ]
        assert len(todo) >= 1

    def test_detects_missing_concerns(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        # Plan that mentions nothing about security
        plan = (
            "1. Create the endpoint\n"
            "2. Add error handling\n"
            "3. Write tests\n"
        )
        patterns = engine._detect_anti_patterns(plan)
        missing = [
            p for p in patterns
            if p.pattern_type == "missing_security"
        ]
        assert len(missing) >= 1

    def test_detects_no_structure(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        patterns = engine._detect_anti_patterns(NO_STRUCTURE_PLAN)
        no_struct = [
            p for p in patterns
            if p.pattern_type == "no_structure"
        ]
        assert len(no_struct) >= 1

    def test_detects_god_step(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        long_step = "x" * 600
        plan = f"1. {long_step}\n2. Short step\n"
        patterns = engine._detect_anti_patterns(plan)
        god = [
            p for p in patterns
            if p.pattern_type == "god_step"
        ]
        assert len(god) >= 1

    def test_clean_plan_has_few_issues(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        patterns = engine._detect_anti_patterns(SAMPLE_PLAN)
        # A well-structured plan should have few anti-patterns
        # It may flag missing concerns (e.g. security) which is valid
        vague = [
            p for p in patterns
            if p.pattern_type == "vague_language"
        ]
        assert len(vague) == 0
        god_steps = [
            p for p in patterns
            if p.pattern_type == "god_step"
        ]
        assert len(god_steps) == 0
        todos = [
            p for p in patterns
            if p.pattern_type == "todo_marker"
        ]
        assert len(todos) == 0


class TestPlanHealthScore:
    def test_perfect_scores(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        scores = {dim: 10.0 for dim in PLAN_DIMENSIONS}
        health = engine._compute_plan_health(scores, 0)
        assert health == 100.0

    def test_zero_scores(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        scores = {dim: 0.0 for dim in PLAN_DIMENSIONS}
        health = engine._compute_plan_health(scores, 0)
        assert health == 0.0

    def test_anti_patterns_reduce_health(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        scores = {dim: 10.0 for dim in PLAN_DIMENSIONS}
        health_clean = engine._compute_plan_health(scores, 0)
        health_dirty = engine._compute_plan_health(scores, 5)
        assert health_dirty < health_clean
        assert health_dirty == 75.0  # 100 - 5*5

    def test_empty_scores(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        health = engine._compute_plan_health({}, 0)
        assert health == 0.0


class TestProcessPlanOptimizer:
    def test_invalid_phase(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        result = engine.process_plan_optimizer(
            "s1", _make_thought(), phase="invalid_phase",
        )
        assert not result.success
        assert "Invalid phase" in (result.message or "")

    def test_submit_plan(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        result = engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan",
            plan_text=SAMPLE_PLAN,
            plan_context="Adding auth to the web app",
        )
        assert result.success
        assert result.plan_text == SAMPLE_PLAN
        assert result.plan_context == "Adding auth to the web app"
        # Anti-patterns auto-detected
        assert isinstance(result.anti_patterns, list)

    def test_submit_plan_requires_text(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        result = engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan",
        )
        assert not result.success
        assert "plan_text is required" in (result.message or "")

    def test_analyze_dimension(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="analyze",
            dimension="clarity", score=8.5,
        )
        assert result.success
        assert result.analysis_scores["clarity"] == 8.5

    def test_analyze_invalid_dimension(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="analyze",
            dimension="nonexistent", score=5.0,
        )
        assert not result.success
        assert "Invalid dimension" in (result.message or "")

    def test_analyze_clamps_score(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="analyze",
            dimension="clarity", score=15.0,
        )
        assert result.success
        assert result.analysis_scores["clarity"] == 10.0

    def test_analyze_adds_issue(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="analyze",
            issue="Missing rollback strategy",
        )
        assert result.success
        assert "Missing rollback strategy" in result.analysis_issues

    def test_add_variant(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="add_variant",
            variant_label="A",
            variant_name="Minimal & Pragmatic",
            variant_summary="Quick implementation",
            variant_pros=["Fast to ship"],
            variant_cons=["Less robust"],
            variant_risk_level="low",
        )
        assert result.success
        assert len(result.variants) == 1
        assert result.variants[0].label == "A"
        assert result.variants[0].name == "Minimal & Pragmatic"

    def test_add_variant_requires_label(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="add_variant",
            variant_name="Test",
        )
        assert not result.success

    def test_add_duplicate_variant_rejected(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="add_variant",
            variant_label="A", variant_name="First",
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=3),
            phase="add_variant",
            variant_label="A", variant_name="Duplicate",
        )
        assert not result.success
        assert "already exists" in (result.message or "")

    def test_score_variant(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="add_variant",
            variant_label="A", variant_name="Minimal",
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=3),
            phase="score_variant",
            variant_label="A",
            dimension="clarity", score=9.0,
        )
        assert result.success
        assert result.variants[0].scores["clarity"] == 9.0
        assert result.variants[0].total == 9.0

    def test_score_variant_not_found(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="score_variant",
            variant_label="Z",
            dimension="clarity", score=5.0,
        )
        assert not result.success
        assert "not found" in (result.message or "")

    def test_recommend_auto_picks_winner(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        # Add two variants with different scores
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="add_variant",
            variant_label="A", variant_name="Minimal",
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=3),
            phase="add_variant",
            variant_label="B", variant_name="Robust",
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=4),
            phase="score_variant",
            variant_label="A",
            dimension="clarity", score=5.0,
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=5),
            phase="score_variant",
            variant_label="B",
            dimension="clarity", score=9.0,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=6),
            phase="recommend",
            recommendation="B is better due to higher clarity",
        )
        assert result.success
        assert result.winner_label == "B"
        assert result.recommendation == (
            "B is better due to higher clarity"
        )

    def test_recommend_explicit_winner(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="add_variant",
            variant_label="A", variant_name="Minimal",
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=3),
            phase="recommend",
            winner_label="A",
            recommendation="A is good enough",
        )
        assert result.success
        assert result.winner_label == "A"

    def test_comparison_matrix(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan", plan_text=SAMPLE_PLAN,
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=2),
            phase="add_variant",
            variant_label="A", variant_name="Minimal",
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=3),
            phase="add_variant",
            variant_label="B", variant_name="Robust",
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=4),
            phase="score_variant",
            variant_label="A",
            dimension="clarity", score=7.0,
        )
        engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=5),
            phase="score_variant",
            variant_label="B",
            dimension="clarity", score=9.0,
        )
        result = engine.process_plan_optimizer(
            "s1", _make_thought(thought_number=6),
            phase="recommend",
        )
        assert result.success
        matrix = result.comparison_matrix
        assert "clarity" in matrix
        assert matrix["clarity"]["A"] == 7.0
        assert matrix["clarity"]["B"] == 9.0
        assert "TOTAL" in matrix


class TestFullPlanOptimizerWorkflow:
    """End-to-end workflow test."""

    def test_full_optimize_flow(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)

        # 1. Submit plan
        r = engine.process_plan_optimizer(
            "s1", _make_thought(thought="Submitting plan"),
            phase="submit_plan",
            plan_text=SAMPLE_PLAN,
            plan_context="Adding authentication",
        )
        assert r.success
        assert isinstance(r.anti_patterns, list)

        # 2. Analyze across all dimensions
        for i, dim in enumerate(PLAN_DIMENSIONS, start=2):
            r = engine.process_plan_optimizer(
                "s1",
                _make_thought(
                    thought=f"Scoring {dim}",
                    thought_number=i,
                ),
                phase="analyze",
                dimension=dim, score=7.5,
            )
            assert r.success

        assert len(r.analysis_scores) == len(PLAN_DIMENSIONS)
        assert r.plan_health_score > 0

        # 3. Add 3 variants
        variants = [
            ("A", "Minimal & Pragmatic", "Quick JWT auth"),
            ("B", "Robust & Scalable", "Full OAuth2 + RBAC"),
            ("C", "Optimal Architecture", "Auth service microservice"),
        ]
        step = 10
        for label, name, summary in variants:
            step += 1
            r = engine.process_plan_optimizer(
                "s1",
                _make_thought(
                    thought=f"Adding variant {label}",
                    thought_number=step,
                ),
                phase="add_variant",
                variant_label=label,
                variant_name=name,
                variant_summary=summary,
                variant_pros=[f"Pro of {label}"],
                variant_cons=[f"Con of {label}"],
            )
            assert r.success

        assert len(r.variants) == 3

        # 4. Score each variant
        variant_scores = {
            "A": {"clarity": 9, "simplicity": 9, "risk": 8,
                   "correctness": 6, "completeness": 5,
                   "testability": 7, "edge_cases": 4,
                   "actionability": 8},
            "B": {"clarity": 7, "simplicity": 5, "risk": 7,
                   "correctness": 9, "completeness": 9,
                   "testability": 8, "edge_cases": 8,
                   "actionability": 7},
            "C": {"clarity": 6, "simplicity": 3, "risk": 5,
                   "correctness": 10, "completeness": 10,
                   "testability": 9, "edge_cases": 9,
                   "actionability": 5},
        }
        for label, scores in variant_scores.items():
            for dim, sc in scores.items():
                step += 1
                r = engine.process_plan_optimizer(
                    "s1",
                    _make_thought(
                        thought=f"Scoring {label}:{dim}",
                        thought_number=step,
                    ),
                    phase="score_variant",
                    variant_label=label,
                    dimension=dim, score=float(sc),
                )
                assert r.success

        # 5. Recommend
        step += 1
        r = engine.process_plan_optimizer(
            "s1",
            _make_thought(
                thought="Final recommendation",
                thought_number=step,
                next_thought_needed=False,
            ),
            phase="recommend",
            recommendation=(
                "Variant B provides the best balance of "
                "correctness, completeness, and testability "
                "while maintaining reasonable simplicity."
            ),
        )
        assert r.success
        # B should win (highest total)
        assert r.winner_label == "B"
        assert r.recommendation
        assert "TOTAL" in r.comparison_matrix
        assert len(r.comparison_matrix["TOTAL"]) == 3

    def test_vague_plan_gets_many_anti_patterns(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        r = engine.process_plan_optimizer(
            "s1", _make_thought(),
            phase="submit_plan",
            plan_text=VAGUE_PLAN,
        )
        assert r.success
        assert r.anti_pattern_count >= 5
        # Health should be low
        # Even without analysis scores, anti-patterns detected
        types = {p.pattern_type for p in r.anti_patterns}
        assert "vague_language" in types
