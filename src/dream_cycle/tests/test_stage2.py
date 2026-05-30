"""
Tests for dream_cycle.stage2 — REM scoring and FadeMem decay
"""
import pytest
import math
from datetime import datetime, timezone, timedelta
from dream_cycle.stage2 import classify_decay_tier, score_importance


class TestClassifyDecayTier:
    """classify_decay_tier() — FadeMem 衰减层级分类"""
    
    def test_volatile_market_data(self):
        text = "UST 10Y yield 4.25% today, CPI data release tomorrow"
        tier = classify_decay_tier(text)
        assert tier == "volatile"
    
    def test_volatile_chinese(self):
        text = "今天收盘利率3.5%，央行发布GDP数据"
        tier = classify_decay_tier(text)
        assert tier == "volatile"
    
    def test_stable_user_preferences(self):
        text = "I prefer Polars always, from now on use server config setup"
        tier = classify_decay_tier(text)
        assert tier == "stable"
    
    def test_stable_chinese(self):
        text = "我的偏好是Python，服务器配置用Docker"
        tier = classify_decay_tier(text)
        assert tier == "stable"
    
    def test_normal_technical(self):
        text = "Dream Cycle v3.1 clustering algorithm uses vector similarity"
        tier = classify_decay_tier(text)
        assert tier == "normal"
    
    def test_empty_text(self):
        tier = classify_decay_tier("")
        assert tier == "normal"


class TestScoreImportance:
    """score_importance() — 6维重要性评分 + FadeMem 衰减"""
    
    def test_returns_float_between_0_and_1(self):
        memory = {
            "text": "Test memory content",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        score = score_importance(memory, recall_count=0, session_count=0)
        assert 0.0 <= score <= 1.0
    
    def test_recent_memory_scores_higher(self):
        """新记忆 > 旧记忆"""
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        new_date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        
        old_memory = {"text": "Test content", "created_at": old_date}
        new_memory = {"text": "Test content", "created_at": new_date}
        
        old_score = score_importance(old_memory, recall_count=1, session_count=1)
        new_score = score_importance(new_memory, recall_count=1, session_count=1)
        
        assert new_score > old_score
    
    def test_volatile_decays_faster_than_stable(self):
        """FadeMem: volatile 衰减 > stable 衰减"""
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        
        volatile_mem = {
            "text": "UST yield CPI stock price today breaking data release Fed",
            "created_at": old_date,
        }
        stable_mem = {
            "text": "I prefer Polars always server config setup from now on",
            "created_at": old_date,
        }
        
        volatile_score = score_importance(volatile_mem, recall_count=1, session_count=1)
        stable_score = score_importance(stable_mem, recall_count=1, session_count=1)
        
        # At 30 days, stable should score significantly higher than volatile
        assert stable_score > volatile_score
    
    def test_recall_count_boosts_score(self):
        """高 recall_count → 高分"""
        memory = {
            "text": "Test memory",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        low_recall = score_importance(memory, recall_count=0, session_count=0)
        high_recall = score_importance(memory, recall_count=10, session_count=5)
        
        assert high_recall > low_recall
    
    def test_permanent_marker_boosts_confidence(self):
        """PERMANENT 标记 → confidence=1.0"""
        memory = {
            "text": "⚠️ PERMANENT: This is critical information",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        score = score_importance(memory, recall_count=0, session_count=0)
        # Score should be reasonably high due to confidence boost
        assert score > 0.3


class TestFadeMemDecayCurve:
    """FadeMem 衰减曲线验证"""
    
    def test_volatile_hits_floor_by_day_7(self):
        """volatile 7天内触底 RETENTION_FLOOR"""
        from dream_cycle.config import DECAY_HALF_LIVES, FADEMEM_BETA, RETENTION_FLOOR
        
        days = 7
        hl = DECAY_HALF_LIVES["volatile"]
        beta = FADEMEM_BETA["volatile"]
        lam = math.log(2) / hl
        retention = math.exp(-lam * (days ** beta))
        
        assert retention <= RETENTION_FLOOR + 0.01  # Within 1% of floor
    
    def test_stable_retains_at_day_30(self):
        """stable 30天仍有 >0.5 保留"""
        from dream_cycle.config import DECAY_HALF_LIVES, FADEMEM_BETA
        
        days = 30
        hl = DECAY_HALF_LIVES["stable"]
        beta = FADEMEM_BETA["stable"]
        lam = math.log(2) / hl
        retention = math.exp(-lam * (days ** beta))
        
        assert retention > 0.5
    
    def test_decay_ordering_at_day_7(self):
        """7天时: volatile < normal < stable"""
        from dream_cycle.config import DECAY_HALF_LIVES, FADEMEM_BETA
        
        days = 7
        retentions = {}
        for tier in ["volatile", "normal", "stable"]:
            hl = DECAY_HALF_LIVES[tier]
            beta = FADEMEM_BETA[tier]
            lam = math.log(2) / hl
            retentions[tier] = math.exp(-lam * (days ** beta))
        
        assert retentions["volatile"] < retentions["normal"]
        assert retentions["normal"] < retentions["stable"]
