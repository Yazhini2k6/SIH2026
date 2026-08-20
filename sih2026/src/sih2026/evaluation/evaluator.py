"""Unified Skill-Gap Evaluator orchestrating pre-processing, rule-based matching, and LLM inference."""

import json
from typing import Any, Optional

from sih2026.evaluation.benchmarks import get_industry_benchmarks
from sih2026.evaluation.gemini import GeminiClient
from sih2026.evaluation.ollama import OllamaClient
from sih2026.nlp_processing.terminology import lookup_acronyms
from sih2026.processing.boilerplate import remove_boilerplate
from sih2026.processing.chunker import chunk_unit_aware


def _rule_based_skill_gap_analysis(
    skills_list: list[dict],
    target_role: str,
    benchmark_profile: dict
) -> dict:
    """Generates an immediate deterministic Skill-Gap report if LLMs are offline."""
    extracted_skill_texts = {s.get("text", "").lower() for s in skills_list}
    required_skills = benchmark_profile.get("required_skills", [])

    covered = []
    missing = []

    for req in required_skills:
        req_lower = req.lower()
        if any(req_lower in ext or ext in req_lower for ext in extracted_skill_texts):
            covered.append(req)
        else:
            missing.append(req)

    total_req = len(required_skills)
    score = int((len(covered) / total_req * 100)) if total_req > 0 else 50

    return {
        "domain_identified": f"Syllabus evaluated for {target_role}",
        "target_role": target_role,
        "overall_alignment_score": score,
        "syllabus_summary": f"Contains {len(skills_list)} extracted skill candidate primitives.",
        "covered_skills": covered if covered else ["Core Computer Science Concepts"],
        "missing_skills": missing if missing else ["Advanced Production Tools"],
        "outdated_topics": [],
        "recommendations": [
            f"Integrate remaining missing industry skills for '{target_role}': {', '.join(missing[:3])}."
        ] if missing else ["Curriculum aligns well with baseline expectations."]
    }


class SkillGapEvaluator:
    """Orchestrates pre-processing, acronym enrichment, and AI skill-gap evaluation (Ollama or Gemini)."""

    def __init__(
        self,
        provider: str = "auto",
        ollama_host: str = "",
        ollama_model: str = "",
        ollama_timeout: int = 180,
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.5-flash",
    ):
        self.provider = provider.lower().strip()
        self.ollama_client = OllamaClient(
            host=ollama_host, model=ollama_model, timeout=ollama_timeout
        )
        self.gemini_client = GeminiClient(
            api_key=gemini_api_key, model=gemini_model
        )

        if self.provider == "auto":
            if self.gemini_client.is_configured():
                self.provider = "gemini"
            elif self.ollama_client.is_server_online():
                self.provider = "ollama"
            else:
                self.provider = "rule_based"

    def evaluate(
        self,
        stage1_json: dict,
        nlp_result: dict,
        target_role: str = "Full-Stack Software Developer",
        max_words_per_chunk: int = 400
    ) -> dict[str, Any]:
        """Runs boilerplate removal, acronym expansion, unit chunking, and LLM/rule skill gap analysis.

        Args:
            stage1_json: Stage 1 Document dict payload.
            nlp_result: Stage 2 NLP Pipeline output dict payload.
            target_role: Target industry role for gap comparison.
            max_words_per_chunk: Word limit per unit chunk.

        Returns:
            dict: Complete Skill-Gap Analysis Report.
        """
        # Step 1: Remove repeating boilerplate headers/footers
        cleaned_doc = remove_boilerplate(stage1_json, threshold_ratio=0.5)

        # Step 2: Acronym lookup
        full_text = cleaned_doc.get("llm_input_context", "")
        acronym_map = lookup_acronyms(full_text)

        # Step 3: Fetch industry benchmark profile
        benchmark_profile = get_industry_benchmarks(target_role)

        # Step 4: Unit-aware text chunking
        chunks = chunk_unit_aware(cleaned_doc, max_words_per_chunk=max_words_per_chunk)

        # Extracted skill list from Stage 2
        skills_list = nlp_result.get("skills", [])

        system_prompt = (
            "You are a world-class AI academic curriculum auditor and industry skills evaluator. "
            "Your job is to read raw academic course syllabi, extract their actual subjects and topics, "
            "and perform a rigorous, domain-specific skill-gap analysis against the specified target industry role. "
            "Return your analysis strictly as a valid JSON object."
        )

        user_prompt = f"""
Syllabus Document Context:
{cleaned_doc.get('llm_input_context', '')}

Detected Extracted Skills (Stage 2 NLP):
{json.dumps([s.get('text') for s in skills_list[:40]], indent=2)}

Detected Technical Terminology / Acronyms:
{json.dumps(acronym_map, indent=2)}

Target Industry Role: {target_role}

Role Benchmark Context:
{json.dumps(benchmark_profile, indent=2)}

Instructions for AI Analysis:
1. Carefully read the Syllabus Document Context and extracted skills above.
2. Perform an intelligent, domain-aware comparison between syllabus content and modern industry expectations for '{target_role}'.
3. List 'covered_skills' that are taught in the syllabus relevant to '{target_role}'.
4. List 'missing_skills' essential for '{target_role}' but missing from this syllabus.
5. List 'outdated_topics' (obsolete tools/frameworks).
6. Provide actionable 'recommendations' to update and align the curriculum for '{target_role}'.

Output JSON Format:
Return ONLY a valid JSON object:
{{
  "domain_identified": "<Primary field/domain>",
  "target_role": "{target_role}",
  "overall_alignment_score": <Integer from 0 to 100>,
  "syllabus_summary": "<Brief overview>",
  "covered_skills": ["<Skill taught>"],
  "missing_skills": ["<Missing skill>"],
  "outdated_topics": ["<Legacy topic>"],
  "recommendations": ["<Recommendation>"]
}}
"""

        effective_provider = self.provider
        llm_result = None

        if self.provider == "gemini":
            res = self.gemini_client.generate(prompt=user_prompt, system_prompt=system_prompt, format_json=True)
            if isinstance(res, dict) and "overall_alignment_score" in res:
                llm_result = res
            else:
                effective_provider = "rule_based (fallback from gemini)"

        elif self.provider == "ollama":
            try:
                res = self.ollama_client.generate(prompt=user_prompt, system_prompt=system_prompt, format_json=True)
                if isinstance(res, dict) and "overall_alignment_score" in res:
                    llm_result = res
                else:
                    effective_provider = "rule_based (fallback from ollama)"
            except Exception:
                effective_provider = "rule_based (fallback - ollama offline)"

        if llm_result is None:
            llm_result = _rule_based_skill_gap_analysis(skills_list, target_role, benchmark_profile)

        return {
            "filename": stage1_json.get("filename", "document"),
            "target_role": target_role,
            "provider_used": effective_provider,
            "mcp_enrichment": {
                "total_pages": cleaned_doc.get("total_pages", len(cleaned_doc.get("pages", []))),
                "total_words": cleaned_doc.get("total_words", 0),
                "acronyms_detected": len(acronym_map),
                "acronym_map": acronym_map,
                "unit_chunks_count": len(chunks),
                "industry_benchmark": benchmark_profile,
            },
            "llm_analysis": llm_result,
        }
