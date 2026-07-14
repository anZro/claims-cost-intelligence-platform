"""
Shared configuration for the claims-cost intelligence platform's LLM client.

Mirrors the three-mode pattern from cost-anomaly-monitor (mock / local_sim /
live) so both portfolio pieces demonstrate the same LLM cost-control habit.
"""

import os

# mock | local_sim | live
LLM_MODE = os.getenv("LLM_MODE", "mock")

# local_sim: Ollama connection settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")  # no default — set explicitly per machine
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

# live mode + local_sim cost simulation target model
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
COST_SIM_MODEL = os.getenv("COST_SIM_MODEL", "claude-sonnet-5")

# Per-million-token rates (USD). Verify against current Anthropic pricing
# before relying on this for real budgeting — introductory rates expire.
MODEL_PRICING = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}

# semantic layer metadata the guardrail (and the agent's prompt) read from
SEMANTIC_MANIFEST_PATH = os.getenv(
    "SEMANTIC_MANIFEST_PATH",
    "claims_metrics/target/semantic_manifest.json",
)

# dbt project location + profile dir + mf binary, needed by the agent/API
# to actually execute queries.
#
# NOTE: this is deliberately NOT named DBT_PROJECT_DIR — dbt-core itself
# natively recognizes an environment variable with that exact name for its
# own project-directory resolution. Naming ours identically caused a real,
# confusing bug: our subprocess calls inherit the full parent environment
# (os.environ.copy()), so if a shell has DBT_PROJECT_DIR set for any
# reason, it silently interferes with dbt's own internal resolution even
# though we also explicitly set the subprocess's cwd. Namespacing our var
# avoids the collision entirely rather than relying on remembering not to
# set the colliding name.
CLAIMS_DBT_PROJECT_DIR = os.getenv("CLAIMS_DBT_PROJECT_DIR", "claims_metrics")
DBT_PROFILES_DIR = os.getenv("DBT_PROFILES_DIR", os.path.expanduser("~/.dbt"))
MF_BINARY = os.getenv("MF_BINARY", "mf")

# Reference date the agent resolves relative time expressions ("last quarter",
# "this year") against. Defaults to real system time — the production-honest
# choice — but overridable for testing against historical/synthetic datasets,
# where "today" being the real current date would resolve relative windows
# to periods outside the data entirely (a legitimate, correct outcome, but
# not a useful one for demoing against a fixed synthetic dataset).
AGENT_REFERENCE_DATE = os.getenv("AGENT_REFERENCE_DATE")  # None = use real system date

# local_sim cumulative cost log
LOCAL_SIM_COST_LOG_PATH = os.getenv("LOCAL_SIM_COST_LOG_PATH", "local_sim_cost_log.json")
