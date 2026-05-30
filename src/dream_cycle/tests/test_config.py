"""
Tests for dream_cycle.config — constants and parameters
"""
import pytest
from dream_cycle.config import (
    safe_float,
    DECAY_HALF_LIVES,
    FADEMEM_BETA,
    IMPORTANCE_WEIGHTS,
    RETENTION_FLOOR,
    PROMOTION_MIN_SCORE,
)


class TestSafeFloat:
    """safe_float() — 安全 float 转换"""
    
    def test_normal_number(self):
        assert safe_float("3.14") == 3.14
        assert safe_float("0") == 0.0
        assert safe_float("-1.5") == -1.5
    
    def test_empty_string_returns_default(self):
        assert safe_float("") is None
        assert safe_float("", default=0.0) == 0.0
        assert safe_float("", default=99) == 99
    
    def test_none_returns_default(self):
        assert safe_float(None) is None
        assert safe_float(None, default=42) == 42
    
    def test_invalid_string_returns_default(self):
        assert safe_float("not_a_number") is None
        assert safe_float("not_a_number", default=-1) == -1


class TestDecayParameters:
    """FadeMem 双层衰减参数"""
    
    def test_half_lives_structure(self):
        assert "volatile" in DECAY_HALF_LIVES
        assert "normal" in DECAY_HALF_LIVES
        assert "stable" in DECAY_HALF_LIVES
    
    def test_half_lives_ordering(self):
        """volatile < normal < stable"""
        assert DECAY_HALF_LIVES["volatile"] < DECAY_HALF_LIVES["normal"]
        assert DECAY_HALF_LIVES["normal"] < DECAY_HALF_LIVES["stable"]
    
    def test_beta_structure(self):
        assert "volatile" in FADEMEM_BETA
        assert "normal" in FADEMEM_BETA
        assert "stable" in FADEMEM_BETA
    
    def test_beta_values(self):
        """volatile β>1 (fast), normal β=1, stable β<1 (slow)"""
        assert FADEMEM_BETA["volatile"] > 1.0
        assert FADEMEM_BETA["normal"] == 1.0
        assert FADEMEM_BETA["stable"] < 1.0


class TestImportanceWeights:
    """重要性评分权重"""
    
    def test_weights_sum_to_one(self):
        total = sum(IMPORTANCE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"
    
    def test_all_weights_positive(self):
        for name, weight in IMPORTANCE_WEIGHTS.items():
            assert weight > 0, f"Weight '{name}' is {weight}, expected > 0"
    
    def test_no_single_weight_dominates(self):
        """No single dimension > 50%"""
        for name, weight in IMPORTANCE_WEIGHTS.items():
            assert weight < 0.5, f"Weight '{name}' is {weight}, too dominant"


class TestThresholds:
    """关键阈值"""
    
    def test_retention_floor_reasonable(self):
        assert 0.0 < RETENTION_FLOOR < 0.5
    
    def test_promotion_threshold_reasonable(self):
        assert 0.5 < PROMOTION_MIN_SCORE < 1.0
