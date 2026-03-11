"""Tests for ultra effort_mode across all thinking tools."""

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


def _td(
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


class TestUltraEvidenceTracker:
    """Ultra mode auto-boosts strength for code_ref/test_result."""

    def test_auto_boost_code_ref(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        # Create an ultra_thinking session with a hypothesis
        engine.process_ultra_thought("s1", _td(), phase="explore")
        engine.process_ultra_thought(
            "s1", _td(thought_number=2),
            phase="hypothesize", hypothesis="H1",
        )
        # Add evidence with low strength but code_ref type
        result = engine.add_evidence(
            "s1", 0, "Found in source code",
            evidence_type="code_ref",
            strength=0.3,
            effort_mode="ultra",
        )
        assert result.success
        # Strength should be boosted to at least 0.9
        evidence = result.evidence
        assert len(evidence) >= 1
        assert evidence[-1].strength >= 0.9

    def test_auto_boost_test_result(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_ultra_thought("s1", _td(), phase="explore")
        engine.process_ultra_thought(
            "s1", _td(thought_number=2),
            phase="hypothesize", hypothesis="H1",
        )
        result = engine.add_evidence(
            "s1", 0, "Test passes",
            evidence_type="test_result",
            strength=0.5,
            effort_mode="ultra",
        )
        assert result.success
        assert result.evidence[-1].strength >= 0.9

    def test_no_boost_for_data_point(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_ultra_thought("s1", _td(), phase="explore")
        engine.process_ultra_thought(
            "s1", _td(thought_number=2),
            phase="hypothesize", hypothesis="H1",
        )
        result = engine.add_evidence(
            "s1", 0, "Just a data point",
            evidence_type="data_point",
            strength=0.3,
            effort_mode="ultra",
        )
        assert result.success
        assert result.evidence[-1].strength == 0.3


class TestUltraPremortem:
    """Ultra mode auto-ranks + requires all mitigations."""

    def test_auto_rank_at_identify_causes(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_premortem(
            "s1", _td(), phase="describe_plan",
            plan="Build a rocket",
        )
        engine.process_premortem(
            "s1", _td(thought_number=2),
            phase="identify_causes",
            risk_description="Engine failure",
            likelihood=0.9, impact=0.9,
            effort_mode="ultra",
        )
        result = engine.process_premortem(
            "s1", _td(thought_number=3),
            phase="identify_causes",
            risk_description="Fuel leak",
            likelihood=0.3, impact=0.5,
            effort_mode="ultra",
        )
        assert result.success
        # Ultra should auto-include ranked_risks
        assert len(result.ranked_risks) == 2
        # Highest risk score first
        assert result.ranked_risks[0].description == "Engine failure"

    def test_warn_unmitigated_risks(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_premortem(
            "s1", _td(), phase="describe_plan",
            plan="Build a rocket",
        )
        engine.process_premortem(
            "s1", _td(thought_number=2),
            phase="identify_causes",
            risk_description="Engine failure",
            likelihood=0.9, impact=0.9,
        )
        engine.process_premortem(
            "s1", _td(thought_number=3),
            phase="identify_causes",
            risk_description="Fuel leak",
            likelihood=0.3, impact=0.5,
        )
        # Mitigate only one risk
        result = engine.process_premortem(
            "s1", _td(thought_number=4),
            phase="mitigate",
            risk_index=0,
            mitigation="Add redundant engines",
            effort_mode="ultra",
        )
        assert result.success
        # Should warn about unmitigated risks
        assert result.message is not None
        assert "1 risk(s) still lack mitigations" in result.message


class TestUltraInversion:
    """Ultra mode auto-reinverts + auto-populates."""

    def test_auto_reinvert_all_causes(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_inversion(
            "s1", _td(), phase="define_goal", goal="Ship v2",
        )
        engine.process_inversion(
            "s1", _td(thought_number=2), phase="invert",
        )
        engine.process_inversion(
            "s1", _td(thought_number=3),
            phase="list_failure_causes",
            failure_cause="No testing",
        )
        engine.process_inversion(
            "s1", _td(thought_number=4),
            phase="list_failure_causes",
            failure_cause="No code review",
        )
        # Ultra action_plan: should auto-reinvert causes
        result = engine.process_inversion(
            "s1", _td(thought_number=5),
            phase="action_plan",
            effort_mode="ultra",
        )
        assert result.success
        # Both causes should now have inverted_actions
        for cause in result.failure_causes:
            assert cause.inverted_action is not None
            assert len(cause.inverted_action) > 0
        # Action plan should be auto-populated
        assert len(result.action_plan) >= 2


class TestUltraEffortEstimator:
    """Ultra mode adds 99.7% CI + risk buffer."""

    def test_99_ci_and_risk_buffer(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        result = engine.process_estimate(
            "s1", action="add", task="Build feature",
            optimistic=2.0, likely=5.0, pessimistic=12.0,
            effort_mode="ultra",
        )
        assert result.success
        # 99.7% CI should be populated
        assert result.total_confidence_99_low != 0.0
        assert result.total_confidence_99_high != 0.0
        # 99.7% CI should be wider than 95% CI
        assert result.total_confidence_99_low < result.total_confidence_95_low
        assert result.total_confidence_99_high > result.total_confidence_95_high
        # Risk buffer should be pessimistic * 1.5
        assert result.total_risk_buffer == 12.0 * 1.5

    def test_high_does_not_have_99_ci(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        result = engine.process_estimate(
            "s1", action="add", task="Build feature",
            optimistic=2.0, likely=5.0, pessimistic=12.0,
            effort_mode="high",
        )
        assert result.success
        assert result.total_confidence_99_low == 0.0
        assert result.total_confidence_99_high == 0.0
        assert result.total_risk_buffer == 0.0


class TestUltraPlanOptimizer:
    """Ultra mode: auto-score missing dims, require variants."""

    def test_blocks_recommend_without_variants(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _td(),
            phase="submit_plan",
            plan_text="1. Do something\n2. Do more\n",
        )
        result = engine.process_plan_optimizer(
            "s1", _td(thought_number=2),
            phase="recommend",
            effort_mode="ultra",
        )
        assert not result.success
        assert "requires at least one variant" in (
            result.message or ""
        )

    def test_auto_scores_missing_dimensions(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _td(),
            phase="submit_plan",
            plan_text="1. Build it\n2. Test it\n",
        )
        # Only score 2 of 8 dimensions
        engine.process_plan_optimizer(
            "s1", _td(thought_number=2),
            phase="analyze",
            dimension="clarity", score=8.0,
        )
        engine.process_plan_optimizer(
            "s1", _td(thought_number=3),
            phase="analyze",
            dimension="simplicity", score=7.0,
        )
        # Add a variant, score 1 dimension
        engine.process_plan_optimizer(
            "s1", _td(thought_number=4),
            phase="add_variant",
            variant_label="A", variant_name="Quick",
        )
        engine.process_plan_optimizer(
            "s1", _td(thought_number=5),
            phase="score_variant",
            variant_label="A",
            dimension="clarity", score=9.0,
        )
        # Recommend in ultra mode
        result = engine.process_plan_optimizer(
            "s1", _td(thought_number=6),
            phase="recommend",
            effort_mode="ultra",
        )
        assert result.success
        # All 8 dimensions should be present in analysis
        assert len(result.analysis_scores) == len(PLAN_DIMENSIONS)
        for dim in PLAN_DIMENSIONS:
            assert dim in result.analysis_scores
        # Unscored dims should be 0
        assert result.analysis_scores["correctness"] == 0.0
        assert result.analysis_scores["clarity"] == 8.0
        # Variant should also have all dims scored
        assert len(result.variants[0].scores) == len(PLAN_DIMENSIONS)
        assert result.variants[0].scores["clarity"] == 9.0
        assert result.variants[0].scores["completeness"] == 0.0

    def test_medium_does_not_auto_score(
        self, thinking_dir: Path,
    ) -> None:
        engine = ThinkingEngine(thinking_dir)
        engine.process_plan_optimizer(
            "s1", _td(),
            phase="submit_plan",
            plan_text="1. Build\n2. Test\n",
        )
        engine.process_plan_optimizer(
            "s1", _td(thought_number=2),
            phase="analyze",
            dimension="clarity", score=8.0,
        )
        engine.process_plan_optimizer(
            "s1", _td(thought_number=3),
            phase="add_variant",
            variant_label="A", variant_name="Quick",
        )
        result = engine.process_plan_optimizer(
            "s1", _td(thought_number=4),
            phase="recommend",
            effort_mode="medium",
        )
        assert result.success
        # Should only have 1 dimension scored
        assert len(result.analysis_scores) == 1
