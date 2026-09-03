"""d10 — which reasoning behaviours does test-awareness steering move?

Modules
-------
env      : load API keys from a dotenv file *outside* the repo
judge    : OpenRouter chat-completions client with caching, concurrency, retries
prompts  : the two upstream judge prompts, verbatim
shipped  : loaders for Abdelnabi & Salem's shipped steered outputs and the 1.5B pilot
parse    : Venhoff-format annotation parser and per-trace morphology metrics
"""
