# Troubleshooting Guide

## Neo4j Aura Deployment

### Authentication and Database Errors

Neo4j Aura instances sometimes set both `NEO4J_USERNAME` and `NEO4J_DATABASE`
to the instance ID (e.g. `da1cbf58`) rather than the default `neo4j`. 

**Symptom:** Auth errors or "database does not exist" errors when running 
embedding or retrieval operations.

**Solution:** In your `.env` file, explicitly set:

```bash
NEO4J_USERNAME=neo4j
NEO4J_DATABASE=neo4j
```

Also ensure your `NEO4J_PASSWORD` is correct and that the Neo4j instance 
is running and accessible at `NEO4J_URI`.

## Troubleshooting Check List

1. **Neo4j connection fails**: Verify `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` in `.env`
2. **Embedding setup fails**: Ensure `python -m csco.cli.embed --arm vector` completes successfully before running experiments
3. **LLM API errors**: Check that `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set and have quota remaining
4. **Fixture not found**: Verify fixtures are in `fixtures/` directory and use `.yaml` extension
5. **Deterministic oracle disagreement**: Run `pytest tests/test_fairness.py::test_manual_oracles_match_deterministic_engine` to validate oracle consistency
