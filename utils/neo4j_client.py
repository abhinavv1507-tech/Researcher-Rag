"""
utils/neo4j_client.py

Neo4j AuraDB client with strict schema enforcement and transaction wrapping.

Allowed node types:  Paper, Method, Metric
Allowed edge types:  USES_METHOD, EVALUATES_ON, CITES, PROPOSES
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase, Session  # type: ignore
from utils.logging_config import get_logger

load_dotenv()
log = get_logger(__name__)

ALLOWED_NODE_TYPES = {"Paper", "Method", "Metric"}
ALLOWED_EDGE_TYPES = {"USES_METHOD", "EVALUATES_ON", "CITES", "PROPOSES"}


class Neo4jClient:
    """Thread-safe Neo4j client with strict ontology enforcement."""

    def __init__(self) -> None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ["NEO4J_USER"]
        password = os.environ["NEO4J_PASSWORD"]
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        log.info("neo4j.connected", uri=uri)

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def ensure_constraints(self) -> None:
        """Create uniqueness constraints for all node types."""
        constraints = [
            "CREATE CONSTRAINT paper_arxiv IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (me:Metric) REQUIRE me.name IS UNIQUE",
        ]
        with self._driver.session() as session:
            for cypher in constraints:
                session.run(cypher)
        log.info("neo4j.constraints_ensured")

    # ------------------------------------------------------------------
    # Write helpers (all wrapped in transactions)
    # ------------------------------------------------------------------

    def _validate_node_type(self, node_type: str) -> None:
        if node_type not in ALLOWED_NODE_TYPES:
            raise ValueError(
                f"Rejected node type '{node_type}'. Allowed: {ALLOWED_NODE_TYPES}"
            )

    def _validate_edge_type(self, edge_type: str) -> None:
        if edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(
                f"Rejected edge type '{edge_type}'. Allowed: {ALLOWED_EDGE_TYPES}"
            )

    def upsert_paper(self, arxiv_id: str, title: str, year: int) -> None:
        def _tx(tx):
            tx.run(
                "MERGE (p:Paper {arxiv_id: $arxiv_id}) "
                "SET p.title = $title, p.year = $year",
                arxiv_id=arxiv_id,
                title=title,
                year=year,
            )

        with self._driver.session() as session:
            session.execute_write(_tx)

    def upsert_method(self, name: str, description: str) -> None:
        def _tx(tx):
            tx.run(
                "MERGE (m:Method {name: $name}) SET m.description = $description",
                name=name,
                description=description,
            )

        with self._driver.session() as session:
            session.execute_write(_tx)

    def upsert_metric(self, name: str, unit: str) -> None:
        def _tx(tx):
            tx.run(
                "MERGE (me:Metric {name: $name}) SET me.unit = $unit",
                name=name,
                unit=unit,
            )

        with self._driver.session() as session:
            session.execute_write(_tx)

    def create_relationship(
        self,
        from_type: str,
        from_key: dict[str, Any],
        rel_type: str,
        to_type: str,
        to_key: dict[str, Any],
    ) -> None:
        """
        Generic relationship creator with strict validation.
        from_key / to_key are single-property match dicts e.g. {"arxiv_id": "2401.XXXXX"}.
        """
        self._validate_node_type(from_type)
        self._validate_node_type(to_type)
        self._validate_edge_type(rel_type)

        from_prop, from_val = next(iter(from_key.items()))
        to_prop, to_val = next(iter(to_key.items()))

        cypher = (
            f"MATCH (a:{from_type} {{{from_prop}: $from_val}}) "
            f"MATCH (b:{to_type} {{{to_prop}: $to_val}}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )

        def _tx(tx):
            tx.run(cypher, from_val=from_val, to_val=to_val)

        with self._driver.session() as session:
            session.execute_write(_tx)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def multi_hop_query(self, arxiv_ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve multi-hop context: papers sharing methods or metrics with
        any paper in the provided arxiv_id list.
        """
        cypher = """
        UNWIND $ids AS aid
        MATCH (p:Paper {arxiv_id: aid})
        OPTIONAL MATCH (p)-[:USES_METHOD|PROPOSES]->(m:Method)<-[:USES_METHOD|PROPOSES]-(related:Paper)
        OPTIONAL MATCH (p)-[:EVALUATES_ON]->(me:Metric)<-[:EVALUATES_ON]-(related2:Paper)
        WITH
            collect(DISTINCT {type: 'paper', arxiv_id: related.arxiv_id, title: related.title, relation: 'shared_method', method: m.name}) +
            collect(DISTINCT {type: 'paper', arxiv_id: related2.arxiv_id, title: related2.title, relation: 'shared_metric', metric: me.name}) AS context
        UNWIND context AS item
        RETURN item
        """
        results: list[dict[str, Any]] = []
        with self._driver.session() as session:
            records = session.run(cypher, ids=arxiv_ids)
            for record in records:
                item = dict(record["item"])
                if item.get("arxiv_id"):  # filter null nodes
                    results.append(item)
        log.info("neo4j.multihop", results=len(results))
        return results

    def write_entities_from_extraction(
        self, arxiv_id: str, title: str, year: int, entities: dict[str, Any]
    ) -> None:
        """
        Write a structured entity extraction result to Neo4j.
        `entities` is the validated output from the LLM extraction node.
        """
        # Upsert the paper itself
        self.upsert_paper(arxiv_id, title, year)

        # Methods
        for method in entities.get("methods", []):
            name = method.get("name", "").strip()
            desc = method.get("description", "")
            if not name:
                continue
            self.upsert_method(name, desc)
            rel = method.get("relation", "USES_METHOD")
            if rel not in ALLOWED_EDGE_TYPES:
                rel = "USES_METHOD"
            try:
                self.create_relationship(
                    "Paper", {"arxiv_id": arxiv_id},
                    rel,
                    "Method", {"name": name},
                )
            except Exception as e:
                log.warning("neo4j.relation_skip", error=str(e))

        # Metrics
        for metric in entities.get("metrics", []):
            name = metric.get("name", "").strip()
            unit = metric.get("unit", "")
            if not name:
                continue
            self.upsert_metric(name, unit)
            try:
                self.create_relationship(
                    "Paper", {"arxiv_id": arxiv_id},
                    "EVALUATES_ON",
                    "Metric", {"name": name},
                )
            except Exception as e:
                log.warning("neo4j.relation_skip", error=str(e))

        # Citations
        for cited_id in entities.get("cites", []):
            cited_id = cited_id.strip()
            if not cited_id:
                continue
            # Ensure cited paper node exists (minimal)
            self.upsert_paper(cited_id, cited_id, 0)
            try:
                self.create_relationship(
                    "Paper", {"arxiv_id": arxiv_id},
                    "CITES",
                    "Paper", {"arxiv_id": cited_id},
                )
            except Exception as e:
                log.warning("neo4j.relation_skip", error=str(e))

        log.info("neo4j.entities_written", arxiv_id=arxiv_id)
