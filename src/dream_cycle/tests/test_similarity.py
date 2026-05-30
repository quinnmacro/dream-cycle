"""
Tests for dream_cycle.similarity — text similarity functions
"""
import pytest
from dream_cycle.config import text_hash
from dream_cycle.similarity import (
    jaccard_similarity,
    ngram_similarity,
    combined_similarity,
)


class TestTextHash:
    """text_hash() — 文本指纹"""
    
    def test_deterministic(self):
        text = "Hello world"
        assert text_hash(text) == text_hash(text)
    
    def test_case_insensitive(self):
        assert text_hash("Hello World") == text_hash("hello world")
    
    def test_whitespace_normalized(self):
        assert text_hash("hello  world") == text_hash("hello world")
    
    def test_different_texts_different_hashes(self):
        h1 = text_hash("hello world")
        h2 = text_hash("goodbye world")
        assert h1 != h2
    
    def test_hash_length(self):
        h = text_hash("test")
        assert len(h) == 16  # MD5 hex digest truncated to 16 chars


class TestJaccardSimilarity:
    """jaccard_similarity() — 集合交并比"""
    
    def test_identical_texts(self):
        sim = jaccard_similarity("hello world", "hello world")
        assert sim == 1.0
    
    def test_completely_different(self):
        sim = jaccard_similarity("hello world", "foo bar baz")
        assert sim == 0.0
    
    def test_partial_overlap(self):
        sim = jaccard_similarity("hello world test", "hello world foo")
        # 交集: {hello, world} = 2
        # 并集: {hello, world, test, foo} = 4
        # Jaccard = 2/4 = 0.5
        assert sim == 0.5
    
    def test_empty_text(self):
        sim = jaccard_similarity("", "hello")
        assert sim == 0.0
    
    def test_symmetric(self):
        s1 = jaccard_similarity("hello world", "world hello foo")
        s2 = jaccard_similarity("world hello foo", "hello world")
        assert s1 == s2


class TestNgramSimilarity:
    """ngram_similarity() — n-gram 重叠"""
    
    def test_identical_texts(self):
        sim = ngram_similarity("hello world", "hello world")
        assert sim == 1.0
    
    def test_completely_different(self):
        sim = ngram_similarity("hello", "world")
        assert sim < 0.3
    
    def test_case_insensitive(self):
        s1 = ngram_similarity("Hello World", "hello world")
        assert s1 == 1.0
    
    def test_short_text(self):
        sim = ngram_similarity("hi", "hello")
        assert 0.0 <= sim <= 1.0


class TestCombinedSimilarity:
    """combined_similarity() — 综合相似度"""
    
    def test_identical_texts(self):
        sim = combined_similarity("hello world test", "hello world test")
        assert sim > 0.9
    
    def test_similar_texts(self):
        t1 = "The quick brown fox jumps over the lazy dog"
        t2 = "The quick brown fox leaps over the lazy dog"
        sim = combined_similarity(t1, t2)
        assert sim > 0.7
    
    def test_different_texts(self):
        t1 = "hello world this is a test"
        t2 = "completely different text here"
        sim = combined_similarity(t1, t2)
        assert sim < 0.3
    
    def test_returns_float_between_0_and_1(self):
        sim = combined_similarity("hello", "world")
        assert 0.0 <= sim <= 1.0
