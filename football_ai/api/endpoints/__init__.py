"""API-level, version-agnostic endpoints (health probes).

Health probes live outside /api/v1 because orchestrators (Kubernetes,
Railway, Fly) hit the pod, not the API — the probe path must stay stable
across API version bumps.
"""
