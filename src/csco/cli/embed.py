"""CLI: embed playbooks (Arm 2) or LexPlaycard summaries (Arm 4) into Neo4j vector indexes.

Usage
-----
  # Arm 2 — chunk and embed all playbook .md files
  python -m csco.cli.embed --arm vector

  # Arm 4 — embed LexPlaycard.summary nodes already in the lexical graph
  #          (requires `python -m csco.cli.load_lexical` to have run first)
  python -m csco.cli.embed --arm lexical
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_PLAYBOOKS_DIR = Path(__file__).parent.parent.parent.parent / "corpus" / "playbooks_embed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed content into Neo4j vector index")
    parser.add_argument(
        "--arm",
        choices=["vector", "lexical"],
        required=True,
        help="vector = embed playbook chunks; lexical = embed LexPlaycard summaries",
    )
    args = parser.parse_args()

    if args.arm == "vector":
        from csco.arms.vector import setup_vectorstore, _NODE_LABEL
        from csco.settings import get_settings
        from neo4j import GraphDatabase

        # Delete all existing Chunk nodes before re-indexing so stale embeddings
        # from previous runs don't accumulate alongside new ones.
        s = get_settings()
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
        with driver.session(database=s.neo4j_database) as session:
            result = session.run(f"MATCH (n:{_NODE_LABEL}) DETACH DELETE n RETURN count(n) AS deleted")
            deleted = result.single()["deleted"]
            logger.info("Deleted %d stale %s nodes from Neo4j", deleted, _NODE_LABEL)
        driver.close()

        playbook_paths = sorted(_PLAYBOOKS_DIR.glob("*.md"))
        if not playbook_paths:
            logger.error("No .md files found in %s", _PLAYBOOKS_DIR)
            sys.exit(1)
        logger.info("Found %d playbook files in %s", len(playbook_paths), _PLAYBOOKS_DIR)
        vs = setup_vectorstore([str(p) for p in playbook_paths])
        logger.info("Vector index ready: %s", vs.index_name)

    elif args.arm == "lexical":
        from csco.arms.lexical import setup_lex_playcard_embeddings
        from csco.settings import get_settings
        from neo4j import GraphDatabase

        # setup_lex_playcard_embeddings() uses Neo4jVector.from_existing_graph,
        # which only embeds nodes whose embedding property is NULL — it silently
        # skips nodes that already carry a (possibly stale) embedding. Clear the
        # property first so re-running always recomputes from the current summary.
        s = get_settings()
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
        with driver.session(database=s.neo4j_database) as session:
            cleared = session.run(
                "MATCH (n:LexPlaycard) SET n.embedding = NULL RETURN count(n) AS c"
            ).single()["c"]
            logger.info("Cleared stale embedding on %d LexPlaycard nodes", cleared)
        driver.close()

        vs = setup_lex_playcard_embeddings()
        logger.info("LexPlaycard embeddings ready: %s", vs.index_name)

    print("Done.")


if __name__ == "__main__":
    main()
