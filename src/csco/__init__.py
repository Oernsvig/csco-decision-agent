"""CSCO Decision Agent — Milestone 1."""

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # truststore not installed — SSL will use default Python cert store
