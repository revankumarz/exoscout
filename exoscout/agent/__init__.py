"""ExoScout agent layer.

Turns the four deterministic tools into an agentic triage loop:

    orchestrator -> (LLM picks a tool) -> execute -> feed result back -> repeat
                 -> final novelty/false-positive verdict + provenance

The LLM is optional. With an OpenAI-compatible endpoint configured (local
Ollama or a hosted API) the orchestrator runs a real ReAct-style loop; without
one it falls back to a deterministic planner so the pipeline is always demoable.
"""
