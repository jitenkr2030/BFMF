"""
Knowledge Graph Integration for Bharat-FM Phase 3

This module provides comprehensive knowledge graph capabilities including:
- Semantic reasoning and relationship mapping
- Fact verification and consistency checking
- Knowledge extraction and integration
- Graph-based query processing
- Contextual knowledge retrieval
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import pickle
from pathlib import Path
import networkx as nx
import re
from collections import defaultdict, deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    ENTITY = "entity"
    CONCEPT = "concept"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    FACT = "fact"
    SOURCE = "source"
    CATEGORY = "category"


class RelationType(Enum):
    """Types of relations between nodes"""
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    DEPENDS_ON = "depends_on"
    USED_FOR = "used_for"
    LOCATED_IN = "located_in"
    BELONGS_TO = "belongs_to"
    WORKS_FOR = "works_for"
    CREATED_BY = "created_by"
    INFLUENCES = "influences"
    INFLUENCED_BY = "influenced_by"
    INCLUDES = "includes"


class ConfidenceLevel(Enum):
    """Confidence levels for facts and relationships"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


class VerificationStatus(Enum):
    """Verification status of facts"""
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    DEBUNKED = "debunked"
    PARTIALLY_VERIFIED = "partially_verified"


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    node_id: str
    node_type: NodeType
    label: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['node_type'] = self.node_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create from dictionary"""
        data = data.copy()
        data['node_type'] = NodeType(data['node_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class KnowledgeEdge:
    """Edge in the knowledge graph"""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['relation_type'] = self.relation_type.value
        data['verification_status'] = self.verification_status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEdge':
        """Create from dictionary"""
        data = data.copy()
        data['relation_type'] = RelationType(data['relation_type'])
        data['verification_status'] = VerificationStatus(data['verification_status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class Fact:
    """Individual fact with verification status"""
    fact_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    sources: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['verification_status'] = self.verification_status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """Create from dictionary"""
        data = data.copy()
        data['verification_status'] = VerificationStatus(data['verification_status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class KnowledgeGraph:
    """Advanced Knowledge Graph with semantic reasoning and fact verification"""
    
    def __init__(self, storage_path: str = "./knowledge_graph"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        
        # Knowledge storage
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.facts: Dict[str, Fact] = {}
        
        # Storage files
        self.nodes_file = self.storage_path / "nodes.pkl"
        self.edges_file = self.storage_path / "edges.pkl"
        self.facts_file = self.storage_path / "facts.pkl"
        self.graph_file = self.storage_path / "graph.gpickle"
        
        # Load existing data
        self._load_data()
        
        # Initialize with basic knowledge
        self._initialize_basic_knowledge()
        
        logger.info(f"KnowledgeGraph initialized with storage path: {storage_path}")
    
    async def start(self):
        """Start the knowledge graph"""
        logger.info("KnowledgeGraph started")
    
    async def stop(self):
        """Stop the knowledge graph and save data"""
        self._save_data()
        logger.info("KnowledgeGraph stopped")
    
    def _load_data(self):
        """Load data from disk"""
        try:
            if self.nodes_file.exists():
                with open(self.nodes_file, 'rb') as f:
                    self.nodes = pickle.load(f)
            
            if self.edges_file.exists():
                with open(self.edges_file, 'rb') as f:
                    self.edges = pickle.load(f)
            
            if self.facts_file.exists():
                with open(self.facts_file, 'rb') as f:
                    self.facts = pickle.load(f)
            
            if self.graph_file.exists():
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
            
            logger.info(f"Loaded {len(self.nodes)} nodes, {len(self.edges)} edges, {len(self.facts)} facts")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to disk"""
        try:
            with open(self.nodes_file, 'wb') as f:
                pickle.dump(self.nodes, f)
            
            with open(self.edges_file, 'wb') as f:
                pickle.dump(self.edges, f)
            
            with open(self.facts_file, 'wb') as f:
                pickle.dump(self.facts, f)
            
            with open(self.graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
                
            logger.info("Knowledge graph data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = int(time.time())
        random_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_hash}"
    
    def _initialize_basic_knowledge(self):
        """Initialize with basic knowledge about India and AI"""
        if len(self.nodes) == 0:
            # Add basic entities
            india_node = KnowledgeNode(
                node_id="entity_india",
                node_type=NodeType.ENTITY,
                label="India",
                description="Country in South Asia",
                properties={"population": "1.4 billion", "capital": "New Delhi"},
                confidence=1.0,
                sources=["basic_knowledge"]
            )
            self.add_node(india_node)
            
            ai_node = KnowledgeNode(
                node_id="concept_ai",
                node_type=NodeType.CONCEPT,
                label="Artificial Intelligence",
                description="Simulation of human intelligence in machines",
                properties={"field": "Computer Science", "emerged": "1950s"},
                confidence=0.9,
                sources=["basic_knowledge"]
            )
            self.add_node(ai_node)
            
            # Add basic relation
            relation = KnowledgeEdge(
                edge_id="edge_india_ai",
                source_id="entity_india",
                target_id="concept_ai",
                relation_type=RelationType.RELATED_TO,
                weight=0.8,
                confidence=0.7,
                sources=["basic_knowledge"]
            )
            self.add_edge(relation)
            
            logger.info("Initialized basic knowledge graph")
    
    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the knowledge graph"""
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            self.graph.add_node(node.node_id, **node.to_dict())
            logger.info(f"Added node: {node.label} ({node.node_id})")
        return node.node_id
    
    def add_edge(self, edge: KnowledgeEdge) -> str:
        """Add an edge to the knowledge graph"""
        if edge.edge_id not in self.edges:
            self.edges[edge.edge_id] = edge
            self.graph.add_edge(
                edge.source_id, edge.target_id,
                **edge.to_dict()
            )
            logger.info(f"Added edge: {edge.source_id} -> {edge.target_id} ({edge.relation_type.value})")
        return edge.edge_id
    
    def add_fact(self, fact: Fact) -> str:
        """Add a fact to the knowledge base"""
        if fact.fact_id not in self.facts:
            self.facts[fact.fact_id] = fact
            logger.info(f"Added fact: {fact.subject} {fact.predicate} {fact.object}")
        return fact.fact_id
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[KnowledgeEdge]:
        """Get an edge by ID"""
        return self.edges.get(edge_id)
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID"""
        return self.facts.get(fact_id)
    
    def find_nodes_by_label(self, label: str) -> List[KnowledgeNode]:
        """Find nodes by label (fuzzy matching)"""
        matching_nodes = []
        label_lower = label.lower()
        
        for node in self.nodes.values():
            if label_lower in node.label.lower():
                matching_nodes.append(node)
        
        return matching_nodes
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Find nodes by type"""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_related_nodes(self, node_id: str, relation_type: Optional[RelationType] = None) -> List[Tuple[KnowledgeNode, KnowledgeEdge]]:
        """Get nodes related to a given node"""
        related = []
        
        for edge in self.edges.values():
            if edge.source_id == node_id:
                target_node = self.nodes.get(edge.target_id)
                if target_node and (relation_type is None or edge.relation_type == relation_type):
                    related.append((target_node, edge))
            elif edge.target_id == node_id:
                source_node = self.nodes.get(edge.source_id)
                if source_node and (relation_type is None or edge.relation_type == relation_type):
                    related.append((source_node, edge))
        
        return related
    
    def extract_knowledge_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract knowledge entities and relationships from text"""
        extracted = []
        
        # Simple entity extraction (in real implementation, use NLP models)
        entities = self._extract_entities(text)
        relations = self._extract_relations(text, entities)
        
        # Create nodes for entities
        for entity in entities:
            node = KnowledgeNode(
                node_id=self._generate_id("entity"),
                node_type=NodeType.ENTITY,
                label=entity["text"],
                description=f"Entity extracted from text",
                properties=entity.get("properties", {}),
                confidence=entity.get("confidence", 0.5),
                sources=["text_extraction"]
            )
            self.add_node(node)
            extracted.append({"type": "node", "data": node.to_dict()})
        
        # Create edges for relations
        for relation in relations:
            edge = KnowledgeEdge(
                edge_id=self._generate_id("edge"),
                source_id=relation["subject"],
                target_id=relation["object"],
                relation_type=relation["relation_type"],
                weight=relation.get("weight", 1.0),
                confidence=relation.get("confidence", 0.5),
                sources=["text_extraction"]
            )
            self.add_edge(edge)
            extracted.append({"type": "edge", "data": edge.to_dict()})
        
        return extracted
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text (simplified)"""
        entities = []
        
        # Simple pattern-based extraction
        # In real implementation, use proper NLP models
        
        # Extract capitalized words as potential entities
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entities.append({
                    "text": word,
                    "type": "ENTITY",
                    "confidence": 0.6,
                    "properties": {"position": i}
                })
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations between entities (simplified)"""
        relations = []
        
        # Simple pattern-based relation extraction
        text_lower = text.lower()
        
        # Look for common relation patterns
        relation_patterns = {
            "is a": RelationType.IS_A,
            "is an": RelationType.IS_A,
            "has": RelationType.HAS_PROPERTY,
            "part of": RelationType.PART_OF,
            "related to": RelationType.RELATED_TO,
            "causes": RelationType.CAUSES,
            "located in": RelationType.LOCATED_IN,
            "belongs to": RelationType.BELONGS_TO
        }
        
        for pattern, rel_type in relation_patterns.items():
            if pattern in text_lower:
                # Find entities around the pattern
                words = text_lower.split()
                pattern_words = pattern.split()
                
                for i in range(len(words) - len(pattern_words) + 1):
                    if words[i:i+len(pattern_words)] == pattern_words:
                        # Simple heuristic: take previous and next words as entities
                        if i > 0 and i + len(pattern_words) < len(words):
                            subject = words[i-1]
                            object = words[i + len(pattern_words)]
                            
                            relations.append({
                                "subject": subject,
                                "object": object,
                                "relation_type": rel_type,
                                "confidence": 0.5,
                                "weight": 1.0
                            })
        
        return relations
    
    def verify_fact(self, fact: Fact) -> Fact:
        """Verify a fact against the knowledge graph"""
        # Check for supporting evidence
        supporting_facts = []
        contradicting_facts = []
        
        # Look for similar facts
        for existing_fact in self.facts.values():
            if (existing_fact.subject == fact.subject and 
                existing_fact.predicate == fact.predicate):
                
                if existing_fact.object == fact.object:
                    supporting_facts.append(existing_fact)
                else:
                    contradicting_facts.append(existing_fact)
        
        # Check knowledge graph for supporting paths
        subject_nodes = self.find_nodes_by_label(fact.subject)
        object_nodes = self.find_nodes_by_label(fact.object)
        
        if subject_nodes and object_nodes:
            # Check if there's a path between subject and object
            for s_node in subject_nodes:
                for o_node in object_nodes:
                    try:
                        path = nx.shortest_path(self.graph, s_node.node_id, o_node.node_id)
                        if len(path) <= 3:  # Short path indicates strong relation
                            supporting_facts.append(fact)
                    except nx.NetworkXNoPath:
                        pass
        
        # Update fact verification status
        if supporting_facts and not contradicting_facts:
            fact.verification_status = VerificationStatus.VERIFIED
            fact.confidence = min(1.0, fact.confidence + 0.3)
        elif contradicting_facts:
            fact.verification_status = VerificationStatus.DISPUTED
            fact.confidence = max(0.1, fact.confidence - 0.3)
            fact.contradictions = [f.fact_id for f in contradicting_facts]
        elif supporting_facts:
            fact.verification_status = VerificationStatus.PARTIALLY_VERIFIED
            fact.confidence = min(0.8, fact.confidence + 0.2)
        
        fact.evidence = [f.fact_id for f in supporting_facts]
        fact.updated_at = datetime.now()
        
        return fact
    
    def semantic_reasoning(self, query: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic reasoning on the knowledge graph"""
        results = []
        
        # Extract entities from query
        entities = self._extract_entities(query)
        
        for entity in entities:
            entity_nodes = self.find_nodes_by_label(entity["text"])
            
            for node in entity_nodes:
                # Perform graph traversal
                reasoning_paths = self._find_reasoning_paths(node.node_id, max_depth)
                
                for path in reasoning_paths:
                    results.append({
                        "type": "reasoning_path",
                        "start_node": node.label,
                        "path": [self.nodes[n_id].label for n_id in path],
                        "confidence": self._calculate_path_confidence(path),
                        "explanation": self._generate_path_explanation(path)
                    })
        
        return results
    
    def _find_reasoning_paths(self, start_node: str, max_depth: int) -> List[List[str]]:
        """Find reasoning paths from a starting node"""
        paths = []
        visited = set()
        
        def dfs(current_node, path, depth):
            if depth > max_depth or current_node in visited:
                return
            
            visited.add(current_node)
            path.append(current_node)
            
            if len(path) > 1:
                paths.append(path.copy())
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(current_node))
            for neighbor in neighbors:
                dfs(neighbor, path, depth + 1)
            
            path.pop()
            visited.remove(current_node)
        
        dfs(start_node, [], 0)
        return paths
    
    def _calculate_path_confidence(self, path: List[str]) -> float:
        """Calculate confidence score for a reasoning path"""
        if len(path) < 2:
            return 0.0
        
        total_confidence = 0.0
        edge_count = 0
        
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                total_confidence += edge_data.get('confidence', 0.5)
                edge_count += 1
        
        return total_confidence / edge_count if edge_count > 0 else 0.0
    
    def _generate_path_explanation(self, path: List[str]) -> str:
        """Generate human-readable explanation for a reasoning path"""
        if len(path) < 2:
            return "Invalid path"
        
        explanation = f"Because {self.nodes[path[0]].label}"
        
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                relation = edge_data.get('relation_type', 'related_to')
                explanation += f" {relation.replace('_', ' ')} {self.nodes[path[i + 1]].label}"
        
        return explanation
    
    def fact_check_statement(self, statement: str) -> Dict[str, Any]:
        """Fact-check a statement against the knowledge graph"""
        # Extract facts from statement
        extracted_facts = self._extract_facts_from_statement(statement)
        
        results = []
        overall_confidence = 0.0
        
        for fact in extracted_facts:
            verified_fact = self.verify_fact(fact)
            results.append({
                "fact": f"{fact.subject} {fact.predicate} {fact.object}",
                "verification_status": verified_fact.verification_status.value,
                "confidence": verified_fact.confidence,
                "evidence_count": len(verified_fact.evidence),
                "contradiction_count": len(verified_fact.contradictions)
            })
            overall_confidence += verified_fact.confidence
        
        return {
            "statement": statement,
            "overall_verification": "verified" if extracted_facts and overall_confidence / len(extracted_facts) > 0.7 else "disputed",
            "overall_confidence": overall_confidence / len(extracted_facts) if extracted_facts else 0.0,
            "fact_checks": results,
            "knowledge_graph_stats": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "facts": len(self.facts)
            }
        }
    
    def _extract_facts_from_statement(self, statement: str) -> List[Fact]:
        """Extract facts from a statement"""
        facts = []
        
        # Simple fact extraction patterns
        # In real implementation, use more sophisticated NLP
        
        # Pattern: [Subject] is [Object]
        is_pattern = r"(\w+(?:\s+\w+)*)\s+is\s+(\w+(?:\s+\w+)*)"
        matches = re.findall(is_pattern, statement, re.IGNORECASE)
        
        for subject, obj in matches:
            fact = Fact(
                fact_id=self._generate_id("fact"),
                subject=subject.strip(),
                predicate="is",
                object=obj.strip(),
                confidence=0.6,
                sources=["statement_extraction"]
            )
            facts.append(fact)
        
        # Pattern: [Subject] has [Object]
        has_pattern = r"(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)"
        matches = re.findall(has_pattern, statement, re.IGNORECASE)
        
        for subject, obj in matches:
            fact = Fact(
                fact_id=self._generate_id("fact"),
                subject=subject.strip(),
                predicate="has",
                object=obj.strip(),
                confidence=0.6,
                sources=["statement_extraction"]
            )
            facts.append(fact)
        
        return facts
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        stats = {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "facts": len(self.facts),
            "nodes_by_type": {},
            "edges_by_relation": {},
            "facts_by_status": {},
            "graph_metrics": {}
        }
        
        # Nodes by type
        for node in self.nodes.values():
            node_type = node.node_type.value
            stats["nodes_by_type"][node_type] = stats["nodes_by_type"].get(node_type, 0) + 1
        
        # Edges by relation
        for edge in self.edges.values():
            relation = edge.relation_type.value
            stats["edges_by_relation"][relation] = stats["edges_by_relation"].get(relation, 0) + 1
        
        # Facts by status
        for fact in self.facts.values():
            status = fact.verification_status.value
            stats["facts_by_status"][status] = stats["facts_by_status"].get(status, 0) + 1
        
        # Graph metrics
        if len(self.graph.nodes) > 0:
            stats["graph_metrics"] = {
                "density": nx.density(self.graph),
                "is_connected": nx.is_connected(self.graph.to_undirected()),
                "number_of_components": nx.number_connected_components(self.graph.to_undirected()),
                "average_clustering": nx.average_clustering(self.graph.to_undirected())
            }
        
        return stats
    
    def query_knowledge(self, query: str, query_type: str = "semantic") -> List[Dict[str, Any]]:
        """Query the knowledge graph"""
        results = []
        
        if query_type == "semantic":
            # Semantic reasoning query
            results = self.semantic_reasoning(query)
        elif query_type == "fact_check":
            # Fact checking query
            fact_check_result = self.fact_check_statement(query)
            results.append(fact_check_result)
        elif query_type == "entity_search":
            # Entity search
            matching_nodes = self.find_nodes_by_label(query)
            for node in matching_nodes:
                related = self.get_related_nodes(node.node_id)
                results.append({
                    "entity": node.label,
                    "type": node.node_type.value,
                    "description": node.description,
                    "confidence": node.confidence,
                    "related_entities": [(r[0].label, r[1].relation_type.value) for r in related]
                })
        
        return results


# Factory function for creating knowledge graph
async def create_knowledge_graph(config: Dict[str, Any] = None) -> KnowledgeGraph:
    """Create and initialize knowledge graph"""
    config = config or {}
    storage_path = config.get("storage_path", "./knowledge_graph")
    
    kg = KnowledgeGraph(storage_path)
    await kg.start()
    
    return kg