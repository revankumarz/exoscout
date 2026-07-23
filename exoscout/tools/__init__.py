"""ExoScout tools: deterministic, structured-return functions.

Each tool returns a plain dict (never raises for expected failures) so it can
be wrapped as an LLM-callable tool later without changing the call sites.
"""
