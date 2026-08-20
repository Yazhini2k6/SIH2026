"""Evaluation subpackage for SIH 2026 Skill-Gap Analyzer."""

from .evaluator import SkillGapEvaluator
from .benchmarks import get_industry_benchmarks

__all__ = ["SkillGapEvaluator", "get_industry_benchmarks"]
