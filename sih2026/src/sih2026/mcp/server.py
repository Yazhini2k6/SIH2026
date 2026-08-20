"""Model Context Protocol (MCP) server exposing tools for AI assistants."""

import sys
import json
from sih2026.processing.boilerplate import remove_boilerplate
from sih2026.processing.chunker import chunk_unit_aware
from sih2026.nlp_processing.terminology import lookup_acronyms
from sih2026.evaluation.benchmarks import get_industry_benchmarks

try:
    from mcp.server.fastmcp import FastMCP
    _FASTMCP_AVAILABLE = True
except ImportError:
    _FASTMCP_AVAILABLE = False


def create_mcp_server():
    """Initializes FastMCP server instance if FastMCP package is available."""
    if not _FASTMCP_AVAILABLE:
        return None

    mcp = FastMCP("sih2026-mcp", instructions="SIH 2026 Skill-Gap Analyzer FastMCP Server")

    @mcp.tool()
    def remove_boilerplate_tool(doc_json: dict, threshold_ratio: float = 0.5) -> dict:
        """Detects and strips repeating headers, footers, and page numbers across pages."""
        return remove_boilerplate(doc_json, threshold_ratio=threshold_ratio)

    @mcp.tool()
    def chunk_unit_aware_tool(doc_json: dict, max_words_per_chunk: int = 400, overlap_words: int = 50) -> list[dict]:
        """Splits document text into unit-aware chunks (UNIT I, MODULE 1)."""
        return chunk_unit_aware(doc_json, max_words_per_chunk=max_words_per_chunk, overlap_words=overlap_words)

    @mcp.tool()
    def lookup_acronyms_tool(input_data: list[str] | str) -> dict[str, str]:
        """Resolves technical abbreviations into expanded terminology names."""
        return lookup_acronyms(input_data)

    @mcp.tool()
    def get_industry_benchmarks_tool(role_title: str = "Full-Stack Software Developer") -> dict:
        """Returns required skills, concepts, and emerging trends for tech roles."""
        return get_industry_benchmarks(role_title)

    return mcp


def run_mcp_server():
    """Entry point to start FastMCP server stdio transport."""
    server = create_mcp_server()
    if server is None:
        print("FastMCP is not installed. Install with `uv add mcp` or `pip install mcp`.", file=sys.stderr)
        sys.exit(1)

    server.run()


if __name__ == "__main__":
    run_mcp_server()
