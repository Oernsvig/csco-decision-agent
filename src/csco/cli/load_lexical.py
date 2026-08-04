"""CLI: load corpus_lexical.cypher into Neo4j and report node/rel counts.

Usage:  python -m csco.cli.load_lexical

Must run after:
    python -m csco.graph.build_lexical > corpus/corpus_lexical.cypher
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_CYPHER_PATH = Path(__file__).parent.parent.parent.parent / "corpus" / "corpus_lexical.cypher"


def _split_statements(cypher_text: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    for raw_line in cypher_text.splitlines():
        line    = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buf).rstrip(";").strip()
            if stmt:
                statements.append(stmt)
            buf = []
    if buf:
        stmt = "\n".join(buf).strip()
        if stmt:
            statements.append(stmt)
    return statements


def load_lexical(cypher_path: Path = _CYPHER_PATH) -> dict[str, int]:
    if not cypher_path.exists():
        raise FileNotFoundError(
            f"Lexical Cypher file not found: {cypher_path}\n"
            "Run: python -m csco.graph.build_lexical > corpus/corpus_lexical.cypher"
        )

    from neo4j import GraphDatabase
    from csco.settings import get_settings

    settings = get_settings()
    statements = _split_statements(cypher_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d Cypher statements from %s", len(statements), cypher_path)

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    executed = 0
    try:
        with driver.session(database=settings.neo4j_database) as session:
            for i, stmt in enumerate(statements, 1):
                try:
                    session.run(stmt)
                    executed += 1
                except Exception as exc:
                    logger.error("Statement %d/%d FAILED: %s\nQuery: %.200s", i, len(statements), exc, stmt)
                    raise

            counts: dict[str, int] = {}
            for label in ("LexPlaycard", "LexSection", "LexProvision"):
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                counts[label] = result.single()["cnt"]
            rel_result = session.run(
                "MATCH ()-[r:HAS_SECTION|NEXT|HAS_PROVISION|REFERENCES]->() RETURN count(r) AS cnt"
            )
            counts["lex_relationships"] = rel_result.single()["cnt"]
    finally:
        driver.close()

    logger.info("Lexical load complete: %d statements. Counts: %s", executed, counts)
    return counts


def main() -> None:
    try:
        counts = load_lexical()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Load failed: %s", exc)
        raise

    print("\nLexical graph load complete:")
    for label, count in counts.items():
        print(f"  {label:<25} {count}")


if __name__ == "__main__":
    main()
