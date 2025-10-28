"""
Creative Thinking and Idea Generation Module for Bharat-FM MLOps Platform

This module implements sophisticated creative thinking capabilities that enable the AI to:
- Generate novel and innovative ideas
- Think outside conventional boundaries
- Combine disparate concepts creatively
- Adapt creative processes to different domains
- Evaluate and refine creative outputs

Author: Advanced AI Team
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json
import time
from datetime import datetime, timedelta
import random
import itertools
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreativityTechnique(Enum):
    """Creative thinking techniques and methods"""
    BRAINSTORMING = "brainstorming"  # Free-flowing idea generation
    MIND_MAPPING = "mind_mapping"  # Visual idea connections
    SCAMPER = "scamper"  # Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse
    SIX_THINKING_HATS = "six_thinking_hats"  # Multiple perspective thinking
    LATERAL_THINKING = "lateral_thinking"  # Indirect creative approach
    ANALOGICAL_THINKING = "analogical_thinking"  # Reasoning by analogy
    DIVERGENT_THINKING = "divergent_thinking"  # Multiple solution paths
    CONVERGENT_THINKING = "convergent_thinking"  # Focusing on best solutions
    DESIGN_THINKING = "design_thinking"  # Human-centered creative process
    TRIZ = "triz"  # Theory of inventive problem solving

class IdeaCategory(Enum):
    """Categories of creative ideas"""
    INNOVATION = "innovation"  # New inventions or improvements
    ARTISTIC = "artistic"  # Creative expressions
    PROBLEM_SOLVING = "problem_solving"  # Solutions to specific problems
    STRATEGIC = "strategic"  # Business or life strategies
    SCIENTIFIC = "scientific"  # Scientific hypotheses or theories
    PHILOSOPHICAL = "philosophical"  # Abstract concepts and theories
    TECHNOLOGICAL = "technological"  # Technology-related ideas
    SOCIAL = "social"  # Social or cultural innovations
    ENVIRONMENTAL = "environmental"  # Environmental solutions
    EDUCATIONAL = "educational"  # Learning and teaching innovations

@dataclass
class CreativeIdea:
    """Represents a creative idea with metadata"""
    idea_id: str
    title: str
    description: str
    category: IdeaCategory
    technique: CreativityTechnique
    novelty_score: float  # 0.0 to 1.0
    feasibility_score: float  # 0.0 to 1.0
    impact_score: float  # 0.0 to 1.0
    creativity_score: float  # 0.0 to 1.0
    components: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    inspiration_sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CreativeProcess:
    """Represents a creative thinking process"""
    process_id: str
    prompt: str
    technique: CreativityTechnique
    ideas_generated: List[CreativeIdea]
    process_steps: List[Dict[str, Any]]
    evaluation_metrics: Dict[str, float]
    duration: timedelta
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConceptConnection:
    """Represents a connection between concepts"""
    concept_a: str
    concept_b: str
    connection_type: str
    strength: float  # 0.0 to 1.0
    creativity_boost: float  # How much this connection boosts creativity
    description: str

class CreativeThinkingEngine:
    """
    Advanced creative thinking and idea generation engine
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Creative parameters
        self.novelty_threshold = self.config.get('novelty_threshold', 0.6)
        self.creativity_boost = self.config.get('creativity_boost', 1.2)
        self.idea_diversity_factor = self.config.get('idea_diversity_factor', 0.8)
        self.max_ideas_per_session = self.config.get('max_ideas_per_session', 50)
        
        # Data structures
        self.idea_database = {}  # idea_id -> CreativeIdea
        self.concept_network = nx.Graph()  # Concept connection network
        self.creative_processes = deque(maxlen=1000)  # Process history
        self.creativity_patterns = defaultdict(list)  # Learned creativity patterns
        
        # Language models for creative generation
        self.tokenizer = None
        self.generative_model = None
        self.embedding_model = None
        
        # Initialize components
        self._initialize_creative_models()
        self._build_concept_network()
        self._initialize_creativity_techniques()
        
        logger.info("Creative Thinking Engine initialized successfully")
    
    def _initialize_creative_models(self):
        """Initialize language models for creative generation"""
        try:
            # Load generative model
            model_name = "gpt2-medium"  # Can be upgraded to larger models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.generative_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.generative_model.eval()
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.generative_model.to(self.device)
            
            logger.info("Creative language models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load creative models: {e}")
            # Fallback to rule-based creativity
            self.tokenizer = None
            self.generative_model = None
    
    def _build_concept_network(self):
        """Build initial concept network for creative connections"""
        # Core concepts across different domains
        core_concepts = [
            # Technology
            "artificial_intelligence", "machine_learning", "blockchain", "quantum_computing",
            "virtual_reality", "augmented_reality", "internet_of_things", "cloud_computing",
            
            # Science
            "biology", "chemistry", "physics", "mathematics", "astronomy", "genetics",
            "neuroscience", "psychology", "ecology", "evolution",
            
            # Arts & Humanities
            "music", "art", "literature", "philosophy", "history", "architecture",
            "design", "theater", "dance", "photography",
            
            # Business & Society
            "economics", "politics", "education", "healthcare", "environment", "energy",
            "transportation", "communication", "agriculture", "manufacturing",
            
            # Abstract Concepts
            "time", "space", "consciousness", "emotion", "logic", "creativity",
            "innovation", "sustainability", "efficiency", "beauty"
        ]
        
        # Add concepts to network
        for concept in core_concepts:
            self.concept_network.add_node(concept, domain=self._get_concept_domain(concept))
        
        # Create initial connections
        self._create_initial_connections()
        
        logger.info(f"Built concept network with {len(self.concept_network.nodes())} concepts")
    
    def _get_concept_domain(self, concept: str) -> str:
        """Determine domain of a concept"""
        domain_mapping = {
            "artificial_intelligence": "technology", "machine_learning": "technology",
            "blockchain": "technology", "quantum_computing": "technology",
            "virtual_reality": "technology", "augmented_reality": "technology",
            "internet_of_things": "technology", "cloud_computing": "technology",
            
            "biology": "science", "chemistry": "science", "physics": "science",
            "mathematics": "science", "astronomy": "science", "genetics": "science",
            "neuroscience": "science", "psychology": "science", "ecology": "science",
            "evolution": "science",
            
            "music": "arts", "art": "arts", "literature": "arts", "philosophy": "arts",
            "history": "arts", "architecture": "arts", "design": "arts",
            "theater": "arts", "dance": "arts", "photography": "arts",
            
            "economics": "society", "politics": "society", "education": "society",
            "healthcare": "society", "environment": "society", "energy": "society",
            "transportation": "society", "communication": "society",
            "agriculture": "society", "manufacturing": "society",
            
            "time": "abstract", "space": "abstract", "consciousness": "abstract",
            "emotion": "abstract", "logic": "abstract", "creativity": "abstract",
            "innovation": "abstract", "sustainability": "abstract",
            "efficiency": "abstract", "beauty": "abstract"
        }
        
        return domain_mapping.get(concept, "general")
    
    def _create_initial_connections(self):
        """Create initial connections between concepts"""
        # Define concept relationships
        connections = [
            ("artificial_intelligence", "machine_learning", "foundation", 0.9),
            ("artificial_intelligence", "neuroscience", "inspiration", 0.7),
            ("blockchain", "economics", "application", 0.6),
            ("quantum_computing", "physics", "foundation", 0.8),
            ("virtual_reality", "psychology", "application", 0.7),
            ("biology", "genetics", "related", 0.9),
            ("psychology", "consciousness", "study", 0.8),
            ("art", "design", "related", 0.8),
            ("technology", "society", "impact", 0.7),
            ("environment", "sustainability", "concern", 0.9),
            ("education", "psychology", "application", 0.6),
            ("healthcare", "biology", "application", 0.8),
            ("creativity", "innovation", "related", 0.8),
            ("logic", "mathematics", "foundation", 0.9),
            ("time", "physics", "concept", 0.8)
        ]
        
        # Add connections to network
        for concept_a, concept_b, conn_type, strength in connections:
            if concept_a in self.concept_network.nodes() and concept_b in self.concept_network.nodes():
                self.concept_network.add_edge(
                    concept_a, concept_b,
                    connection_type=conn_type,
                    strength=strength,
                    creativity_boost=strength * 0.8
                )
    
    def _initialize_creativity_techniques(self):
        """Initialize creativity techniques and their parameters"""
        self.technique_configs = {
            CreativityTechnique.BRAINSTORMING: {
                'idea_generation_rate': 0.9,
                'diversity_weight': 0.8,
                'quantity_over_quality': True,
                'time_limit': 300  # 5 minutes
            },
            CreativityTechnique.MIND_MAPPING: {
                'visual_spacing': 0.7,
                'connection_focus': 0.9,
                'hierarchical_organization': True,
                'branching_factor': 3
            },
            CreativityTechnique.SCAMPER: {
                'substitute_weight': 0.7,
                'combine_weight': 0.8,
                'adapt_weight': 0.6,
                'modify_weight': 0.7,
                'put_to_use_weight': 0.5,
                'eliminate_weight': 0.4,
                'reverse_weight': 0.6
            },
            CreativityTechnique.SIX_THINKING_HATS: {
                'facts_focus': 0.6,
                'emotions_focus': 0.7,
                'criticism_focus': 0.8,
                'optimism_focus': 0.7,
                'creativity_focus': 0.9,
                'process_focus': 0.6
            },
            CreativityTechnique.LATERAL_THINKING: {
                'indirect_approach': 0.9,
                'challenge_assumptions': 0.8,
                'alternative_perspectives': 0.9,
                'random_stimulation': 0.7
            },
            CreativityTechnique.ANALOGICAL_THINKING: {
                'analogy_strength': 0.8,
                'domain_distance': 0.7,
                'mapping_quality': 0.8,
                'transfer_effectiveness': 0.7
            },
            CreativityTechnique.DIVERGENT_THINKING: {
                'multiple_solutions': True,
                'idea_quantity': 0.9,
                'originality_weight': 0.8,
                'flexibility_weight': 0.7
            },
            CreativityTechnique.CONVERGENT_THINKING: {
                'focus_on_best': True,
                'evaluation_criteria': 0.9,
                'solution_quality': 0.8,
                'practicality_weight': 0.7
            },
            CreativityTechnique.DESIGN_THINKING: {
                'empathy_weight': 0.8,
                'ideation_weight': 0.9,
                'prototyping_weight': 0.7,
                'testing_weight': 0.6,
                'iteration_weight': 0.8
            },
            CreativityTechnique.TRIZ: {
                'contradiction_resolution': 0.9,
                'ideality_weight': 0.8,
                'evolution_weight': 0.7,
                'resource_utilization': 0.8
            }
        }
    
    def generate_creative_ideas(self, prompt: str, technique: CreativityTechnique,
                               category: Optional[IdeaCategory] = None,
                               num_ideas: int = 10) -> List[CreativeIdea]:
        """
        Generate creative ideas using specified technique
        
        Args:
            prompt: Creative prompt or problem statement
            technique: Creativity technique to use
            category: Optional idea category
            num_ideas: Number of ideas to generate
            
        Returns:
            List[CreativeIdea]: Generated creative ideas
        """
        start_time = datetime.now()
        
        # Extract concepts from prompt
        prompt_concepts = self._extract_concepts(prompt)
        
        # Generate ideas based on technique
        if technique == CreativityTechnique.BRAINSTORMING:
            ideas = self._brainstorming_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.MIND_MAPPING:
            ideas = self._mind_mapping_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.SCAMPER:
            ideas = self._scamper_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.SIX_THINKING_HATS:
            ideas = self._six_thinking_hats_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.LATERAL_THINKING:
            ideas = self._lateral_thinking_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.ANALOGICAL_THINKING:
            ideas = self._analogical_thinking_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.DIVERGENT_THINKING:
            ideas = self._divergent_thinking_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.CONVERGENT_THINKING:
            ideas = self._convergent_thinking_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.DESIGN_THINKING:
            ideas = self._design_thinking_technique(prompt, prompt_concepts, num_ideas)
        elif technique == CreativityTechnique.TRIZ:
            ideas = self._triz_technique(prompt, prompt_concepts, num_ideas)
        else:
            ideas = self._brainstorming_technique(prompt, prompt_concepts, num_ideas)  # Default
        
        # Apply category if specified
        if category:
            for idea in ideas:
                idea.category = category
        
        # Calculate creativity scores
        for idea in ideas:
            self._calculate_creativity_scores(idea)
        
        # Store ideas
        for idea in ideas:
            self.idea_database[idea.idea_id] = idea
        
        # Record creative process
        duration = datetime.now() - start_time
        process = CreativeProcess(
            process_id=f"process_{int(time.time())}",
            prompt=prompt,
            technique=technique,
            ideas_generated=ideas,
            process_steps=self._generate_process_steps(technique, prompt),
            evaluation_metrics=self._evaluate_idea_set(ideas),
            duration=duration
        )
        self.creative_processes.append(process)
        
        logger.info(f"Generated {len(ideas)} creative ideas using {technique.value}")
        
        return ideas
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        # Check for known concepts
        for concept in self.concept_network.nodes():
            if concept.replace('_', ' ') in text_lower or concept in text_lower:
                concepts.append(concept)
        
        # Extract additional concepts using simple keyword extraction
        words = re.findall(r'\b\w+\b', text_lower)
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] += 1
        
        # Add frequent words as concepts
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
            if word not in concepts:
                concepts.append(word)
        
        return concepts[:10]  # Limit to top 10 concepts
    
    def _brainstorming_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using brainstorming technique"""
        ideas = []
        
        # Generate random concept combinations
        for i in range(num_ideas):
            # Select random concepts
            selected_concepts = random.sample(concepts, min(3, len(concepts)))
            
            # Generate idea title and description
            if self.generative_model and self.tokenizer:
                idea_text = self._generate_with_model(prompt, selected_concepts, "brainstorming")
            else:
                idea_text = self._generate_rule_based_idea(prompt, selected_concepts, "brainstorming")
            
            # Create idea
            idea = CreativeIdea(
                idea_id=f"idea_{int(time.time())}_{i}",
                title=self._generate_idea_title(idea_text),
                description=idea_text,
                category=self._determine_idea_category(idea_text),
                technique=CreativityTechnique.BRAINSTORMING,
                novelty_score=random.uniform(0.5, 0.9),
                feasibility_score=random.uniform(0.4, 0.8),
                impact_score=random.uniform(0.3, 0.9),
                creativity_score=0.0,  # Will be calculated later
                components=selected_concepts,
                connections=self._find_concept_connections(selected_concepts),
                inspiration_sources=["brainstorming", "random_combination"]
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _mind_mapping_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using mind mapping technique"""
        ideas = []
        
        # Build mind map from central concept
        central_concept = concepts[0] if concepts else "central_idea"
        
        # Create branches
        branches = {}
        for concept in concepts[1:]:
            branch_category = self._get_concept_domain(concept)
            if branch_category not in branches:
                branches[branch_category] = []
            branches[branch_category].append(concept)
        
        # Generate ideas from branches
        idea_count = 0
        for branch_category, branch_concepts in branches.items():
            if idea_count >= num_ideas:
                break
            
            for concept in branch_concepts:
                if idea_count >= num_ideas:
                    break
                
                # Generate idea connecting central concept to branch concept
                connection_text = f"Connection between {central_concept} and {concept}"
                
                if self.generative_model and self.tokenizer:
                    idea_text = self._generate_with_model(prompt, [central_concept, concept], "mind_mapping")
                else:
                    idea_text = self._generate_rule_based_idea(prompt, [central_concept, concept], "mind_mapping")
                
                idea = CreativeIdea(
                    idea_id=f"idea_{int(time.time())}_{idea_count}",
                    title=self._generate_idea_title(idea_text),
                    description=idea_text,
                    category=self._determine_idea_category(idea_text),
                    technique=CreativityTechnique.MIND_MAPPING,
                    novelty_score=random.uniform(0.6, 0.9),
                    feasibility_score=random.uniform(0.5, 0.8),
                    impact_score=random.uniform(0.4, 0.8),
                    creativity_score=0.0,
                    components=[central_concept, concept],
                    connections=[connection_text],
                    inspiration_sources=["mind_mapping", branch_category]
                )
                
                ideas.append(idea)
                idea_count += 1
        
        return ideas
    
    def _scamper_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using SCAMPER technique"""
        ideas = []
        
        scamper_actions = [
            ("Substitute", "Replace something with something else"),
            ("Combine", "Merge two or more elements"),
            ("Adapt", "Adjust to fit a new purpose"),
            ("Modify", "Change size, shape, or other attributes"),
            ("Put to another use", "Use for a different purpose"),
            ("Eliminate", "Remove something"),
            ("Reverse", "Change the order or perspective")
        ]
        
        ideas_per_action = max(1, num_ideas // len(scamper_actions))
        
        for action, description in scamper_actions:
            if len(ideas) >= num_ideas:
                break
            
            for i in range(ideas_per_action):
                if len(ideas) >= num_ideas:
                    break
                
                # Apply SCAMPER action to concepts
                modified_concepts = self._apply_scamper_action(concepts, action)
                
                if self.generative_model and self.tokenizer:
                    idea_text = self._generate_with_model(prompt, modified_concepts, f"scamper_{action.lower()}")
                else:
                    idea_text = self._generate_rule_based_idea(prompt, modified_concepts, f"scamper_{action.lower()}")
                
                idea = CreativeIdea(
                    idea_id=f"idea_{int(time.time())}_{len(ideas)}",
                    title=f"{action}: {self._generate_idea_title(idea_text)}",
                    description=idea_text,
                    category=self._determine_idea_category(idea_text),
                    technique=CreativityTechnique.SCAMPER,
                    novelty_score=random.uniform(0.7, 0.95),
                    feasibility_score=random.uniform(0.4, 0.7),
                    impact_score=random.uniform(0.5, 0.9),
                    creativity_score=0.0,
                    components=modified_concepts,
                    connections=[f"SCAMPER {action}"],
                    inspiration_sources=["SCAMPER", action]
                )
                
                ideas.append(idea)
        
        return ideas
    
    def _six_thinking_hats_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using Six Thinking Hats technique"""
        ideas = []
        
        hats = [
            ("White Hat", "Facts and data", "objective"),
            ("Red Hat", "Emotions and feelings", "emotional"),
            ("Black Hat", "Critical judgment", "critical"),
            ("Yellow Hat", "Optimism and benefits", "optimistic"),
            ("Green Hat", "Creativity and new ideas", "creative"),
            ("Blue Hat", "Process and control", "process")
        ]
        
        ideas_per_hat = max(1, num_ideas // len(hats))
        
        for hat_name, hat_description, perspective in hats:
            if len(ideas) >= num_ideas:
                break
            
            for i in range(ideas_per_hat):
                if len(ideas) >= num_ideas:
                    break
                
                # Generate idea from this perspective
                if self.generative_model and self.tokenizer:
                    idea_text = self._generate_with_model(
                        prompt, concepts, f"six_hats_{perspective}", 
                        additional_context=f"From the {hat_name} perspective: {hat_description}"
                    )
                else:
                    idea_text = self._generate_rule_based_idea(
                        prompt, concepts, f"six_hats_{perspective}"
                    )
                
                idea = CreativeIdea(
                    idea_id=f"idea_{int(time.time())}_{len(ideas)}",
                    title=f"{hat_name}: {self._generate_idea_title(idea_text)}",
                    description=idea_text,
                    category=self._determine_idea_category(idea_text),
                    technique=CreativityTechnique.SIX_THINKING_HATS,
                    novelty_score=random.uniform(0.5, 0.8),
                    feasibility_score=random.uniform(0.5, 0.9),
                    impact_score=random.uniform(0.4, 0.8),
                    creativity_score=0.0,
                    components=concepts,
                    connections=[f"Six Thinking Hats: {hat_name}"],
                    inspiration_sources=["Six Thinking Hats", hat_name]
                )
                
                ideas.append(idea)
        
        return ideas
    
    def _lateral_thinking_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using lateral thinking technique"""
        ideas = []
        
        for i in range(num_ideas):
            # Introduce random element or challenge assumptions
            random_concept = random.choice(list(self.concept_network.nodes()))
            challenge_question = f"What if {random_concept} was completely different?"
            
            # Generate lateral thinking idea
            if self.generative_model and self.tokenizer:
                idea_text = self._generate_with_model(
                    prompt, concepts + [random_concept], "lateral_thinking",
                    additional_context=challenge_question
                )
            else:
                idea_text = self._generate_rule_based_idea(
                    prompt, concepts + [random_concept], "lateral_thinking"
                )
            
            idea = CreativeIdea(
                idea_id=f"idea_{int(time.time())}_{len(ideas)}",
                title=self._generate_idea_title(idea_text),
                description=idea_text,
                category=self._determine_idea_category(idea_text),
                technique=CreativityTechnique.LATERAL_THINKING,
                novelty_score=random.uniform(0.8, 1.0),
                feasibility_score=random.uniform(0.2, 0.6),
                impact_score=random.uniform(0.6, 0.9),
                creativity_score=0.0,
                components=concepts + [random_concept],
                connections=[f"Lateral connection: {random_concept}"],
                inspiration_sources=["lateral_thinking", "random_stimulation"]
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _analogical_thinking_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using analogical thinking technique"""
        ideas = []
        
        for i in range(num_ideas):
            # Find analogous domain
            source_domain = self._find_analogous_domain(concepts)
            
            # Generate analogy
            if self.generative_model and self.tokenizer:
                idea_text = self._generate_with_model(
                    prompt, concepts, "analogical_thinking",
                    additional_context=f"Analogous domain: {source_domain}"
                )
            else:
                idea_text = self._generate_rule_based_idea(
                    prompt, concepts, "analogical_thinking"
                )
            
            idea = CreativeIdea(
                idea_id=f"idea_{int(time.time())}_{len(ideas)}",
                title=f"Analogy: {self._generate_idea_title(idea_text)}",
                description=idea_text,
                category=self._determine_idea_category(idea_text),
                technique=CreativityTechnique.ANALOGICAL_THINKING,
                novelty_score=random.uniform(0.7, 0.9),
                feasibility_score=random.uniform(0.4, 0.7),
                impact_score=random.uniform(0.5, 0.8),
                creativity_score=0.0,
                components=concepts,
                connections=[f"Analogy from {source_domain}"],
                inspiration_sources=["analogical_thinking", source_domain]
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _divergent_thinking_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using divergent thinking technique"""
        ideas = []
        
        # Generate multiple diverse solutions
        for i in range(num_ideas):
            # Create diverse concept combinations
            num_concepts = random.randint(2, min(5, len(concepts)))
            selected_concepts = random.sample(concepts, num_concepts)
            
            # Shuffle concepts for different perspectives
            random.shuffle(selected_concepts)
            
            if self.generative_model and self.tokenizer:
                idea_text = self._generate_with_model(prompt, selected_concepts, "divergent_thinking")
            else:
                idea_text = self._generate_rule_based_idea(prompt, selected_concepts, "divergent_thinking")
            
            idea = CreativeIdea(
                idea_id=f"idea_{int(time.time())}_{len(ideas)}",
                title=self._generate_idea_title(idea_text),
                description=idea_text,
                category=self._determine_idea_category(idea_text),
                technique=CreativityTechnique.DIVERGENT_THINKING,
                novelty_score=random.uniform(0.6, 0.9),
                feasibility_score=random.uniform(0.3, 0.8),
                impact_score=random.uniform(0.4, 0.9),
                creativity_score=0.0,
                components=selected_concepts,
                connections=self._find_concept_connections(selected_concepts),
                inspiration_sources=["divergent_thinking", "multiple_solutions"]
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _convergent_thinking_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using convergent thinking technique"""
        # First generate many ideas, then converge on best ones
        all_ideas = self._divergent_thinking_technique(prompt, concepts, num_ideas * 3)
        
        # Evaluate and select best ideas
        evaluated_ideas = []
        for idea in all_ideas:
            self._calculate_creativity_scores(idea)
            evaluated_ideas.append(idea)
        
        # Sort by creativity score and select top ideas
        evaluated_ideas.sort(key=lambda x: x.creativity_score, reverse=True)
        
        # Refine selected ideas
        final_ideas = []
        for idea in evaluated_ideas[:num_ideas]:
            # Refine idea for better feasibility
            refined_idea = self._refine_idea(idea)
            refined_idea.technique = CreativityTechnique.CONVERGENT_THINKING
            final_ideas.append(refined_idea)
        
        return final_ideas
    
    def _design_thinking_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using design thinking technique"""
        ideas = []
        
        # Design thinking phases: Empathize, Define, Ideate, Prototype, Test
        phases = [
            ("Empathize", "Understand user needs and context"),
            ("Define", "Clearly define the problem"),
            ("Ideate", "Generate creative solutions"),
            ("Prototype", "Create tangible representations"),
            ("Test", "Validate with users")
        ]
        
        ideas_per_phase = max(1, num_ideas // len(phases))
        
        for phase_name, phase_description in phases:
            if len(ideas) >= num_ideas:
                break
            
            for i in range(ideas_per_phase):
                if len(ideas) >= num_ideas:
                    break
                
                if self.generative_model and self.tokenizer:
                    idea_text = self._generate_with_model(
                        prompt, concepts, "design_thinking",
                        additional_context=f"Design thinking phase: {phase_name} - {phase_description}"
                    )
                else:
                    idea_text = self._generate_rule_based_idea(
                        prompt, concepts, "design_thinking"
                    )
                
                idea = CreativeIdea(
                    idea_id=f"idea_{int(time.time())}_{len(ideas)}",
                    title=f"Design Thinking ({phase_name}): {self._generate_idea_title(idea_text)}",
                    description=idea_text,
                    category=self._determine_idea_category(idea_text),
                    technique=CreativityTechnique.DESIGN_THINKING,
                    novelty_score=random.uniform(0.6, 0.8),
                    feasibility_score=random.uniform(0.5, 0.8),
                    impact_score=random.uniform(0.6, 0.9),
                    creativity_score=0.0,
                    components=concepts,
                    connections=[f"Design Thinking: {phase_name}"],
                    inspiration_sources=["design_thinking", phase_name]
                )
                
                ideas.append(idea)
        
        return ideas
    
    def _triz_technique(self, prompt: str, concepts: List[str], num_ideas: int) -> List[CreativeIdea]:
        """Generate ideas using TRIZ technique"""
        ideas = []
        
        # TRIZ principles (simplified)
        triz_principles = [
            ("Segmentation", "Divide an object into independent parts"),
            ("Taking out", "Separate the interfering part"),
            ("Local quality", "Make each part function in optimal conditions"),
            ("Asymmetry", "Change from symmetrical to asymmetrical"),
            ("Merging", "Combine similar or identical objects"),
            ("Universality", "Make one object perform multiple functions"),
            ("Nested doll", "Place objects inside each other"),
            ("Anti-weight", "Counteract the weight with other forces")
        ]
        
        ideas_per_principle = max(1, num_ideas // len(triz_principles))
        
        for principle_name, principle_description in triz_principles:
            if len(ideas) >= num_ideas:
                break
            
            for i in range(ideas_per_principle):
                if len(ideas) >= num_ideas:
                    break
                
                if self.generative_model and self.tokenizer:
                    idea_text = self._generate_with_model(
                        prompt, concepts, "triz",
                        additional_context=f"TRIZ principle: {principle_name} - {principle_description}"
                    )
                else:
                    idea_text = self._generate_rule_based_idea(
                        prompt, concepts, "triz"
                    )
                
                idea = CreativeIdea(
                    idea_id=f"idea_{int(time.time())}_{len(ideas)}",
                    title=f"TRIZ ({principle_name}): {self._generate_idea_title(idea_text)}",
                    description=idea_text,
                    category=self._determine_idea_category(idea_text),
                    technique=CreativityTechnique.TRIZ,
                    novelty_score=random.uniform(0.7, 0.95),
                    feasibility_score=random.uniform(0.4, 0.7),
                    impact_score=random.uniform(0.6, 0.9),
                    creativity_score=0.0,
                    components=concepts,
                    connections=[f"TRIZ principle: {principle_name}"],
                    inspiration_sources=["TRIZ", principle_name]
                )
                
                ideas.append(idea)
        
        return ideas
    
    def _generate_with_model(self, prompt: str, concepts: List[str], technique: str,
                           additional_context: str = "") -> str:
        """Generate idea text using language model"""
        try:
            # Create input text
            input_text = f"Prompt: {prompt}\nConcepts: {', '.join(concepts)}\nTechnique: {technique}"
            if additional_context:
                input_text += f"\nContext: {additional_context}"
            input_text += "\nCreative Idea:"
            
            # Tokenize and generate
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.generative_model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean up
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            idea_text = generated_text.replace(input_text, "").strip()
            
            return idea_text if idea_text else "Generated creative idea"
            
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            return self._generate_rule_based_idea(prompt, concepts, technique)
    
    def _generate_rule_based_idea(self, prompt: str, concepts: List[str], technique: str) -> str:
        """Generate idea text using rule-based approach"""
        # Simple rule-based idea generation
        if not concepts:
            return f"Creative solution for: {prompt}"
        
        # Combine concepts creatively
        if len(concepts) == 1:
            return f"Innovative application of {concepts[0]} to address: {prompt}"
        elif len(concepts) == 2:
            return f"Integration of {concepts[0]} and {concepts[1]} to solve: {prompt}"
        else:
            return f"Creative combination of {', '.join(concepts)} for: {prompt}"
    
    def _generate_idea_title(self, idea_text: str) -> str:
        """Generate a concise title for an idea"""
        # Simple title generation - take first few words
        words = idea_text.split()[:5]
        return " ".join(words).title()
    
    def _determine_idea_category(self, idea_text: str) -> IdeaCategory:
        """Determine the category of an idea based on its content"""
        text_lower = idea_text.lower()
        
        # Category keywords
        category_keywords = {
            IdeaCategory.INNOVATION: ['innovate', 'new', 'breakthrough', 'invention', 'disrupt'],
            IdeaCategory.ARTISTIC: ['art', 'creative', 'design', 'aesthetic', 'beauty'],
            IdeaCategory.PROBLEM_SOLVING: ['solve', 'solution', 'problem', 'fix', 'resolve'],
            IdeaCategory.STRATEGIC: ['strategy', 'plan', 'approach', 'method', 'tactic'],
            IdeaCategory.SCIENTIFIC: ['science', 'research', 'study', 'experiment', 'theory'],
            IdeaCategory.PHILOSOPHICAL: ['philosophy', 'meaning', 'existence', 'ethics', 'logic'],
            IdeaCategory.TECHNOLOGICAL: ['tech', 'digital', 'software', 'hardware', 'system'],
            IdeaCategory.SOCIAL: ['social', 'community', 'people', 'society', 'culture'],
            IdeaCategory.ENVIRONMENTAL: ['environment', 'green', 'sustainable', 'eco', 'nature'],
            IdeaCategory.EDUCATIONAL: ['education', 'learning', 'teach', 'student', 'knowledge']
        }
        
        # Count keyword matches
        category_scores = defaultdict(int)
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    category_scores[category] += 1
        
        # Return category with highest score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return IdeaCategory.INNOVATION  # Default
    
    def _apply_scamper_action(self, concepts: List[str], action: str) -> List[str]:
        """Apply SCAMPER action to concepts"""
        modified_concepts = concepts.copy()
        
        if action == "Substitute":
            # Replace one concept with a random related concept
            if modified_concepts:
                old_concept = modified_concepts[0]
                related_concepts = list(self.concept_network.neighbors(old_concept))
                if related_concepts:
                    modified_concepts[0] = random.choice(related_concepts)
        
        elif action == "Combine":
            # Add a random concept
            all_concepts = list(self.concept_network.nodes())
            if all_concepts:
                new_concept = random.choice(all_concepts)
                if new_concept not in modified_concepts:
                    modified_concepts.append(new_concept)
        
        elif action == "Modify":
            # Modify a concept (simplified - just add "modified_" prefix)
            if modified_concepts:
                modified_concepts[0] = f"modified_{modified_concepts[0]}"
        
        elif action == "Eliminate":
            # Remove a concept
            if len(modified_concepts) > 1:
                modified_concepts.pop(random.randint(0, len(modified_concepts) - 1))
        
        elif action == "Reverse":
            # Reverse the order of concepts
            modified_concepts.reverse()
        
        return modified_concepts
    
    def _find_analogous_domain(self, concepts: List[str]) -> str:
        """Find an analogous domain for given concepts"""
        # Get domains of current concepts
        current_domains = set()
        for concept in concepts:
            if concept in self.concept_network.nodes():
                domain = self.concept_network.nodes[concept].get('domain', 'general')
                current_domains.add(domain)
        
        # Find most different domain
        all_domains = set()
        for node in self.concept_network.nodes():
            domain = self.concept_network.nodes[node].get('domain', 'general')
            all_domains.add(domain)
        
        # Select domain not in current domains
        different_domains = all_domains - current_domains
        if different_domains:
            return random.choice(list(different_domains))
        else:
            return random.choice(list(all_domains))
    
    def _find_concept_connections(self, concepts: List[str]) -> List[str]:
        """Find connections between concepts"""
        connections = []
        
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                if (concept_a in self.concept_network.nodes() and 
                    concept_b in self.concept_network.nodes() and
                    self.concept_network.has_edge(concept_a, concept_b)):
                    
                    edge_data = self.concept_network.edges[concept_a, concept_b]
                    connection_type = edge_data.get('connection_type', 'related')
                    connections.append(f"{concept_a} --{connection_type}--> {concept_b}")
        
        return connections
    
    def _calculate_creativity_scores(self, idea: CreativeIdea):
        """Calculate creativity scores for an idea"""
        # Novelty score based on concept uniqueness
        concept_novelty = len(set(idea.components)) / max(len(idea.components), 1)
        idea.novelty_score = min(idea.novelty_score * concept_novelty, 1.0)
        
        # Feasibility score based on concept familiarity
        concept_familiarity = sum(1 for c in idea.components if c in self.concept_network.nodes()) / max(len(idea.components), 1)
        idea.feasibility_score = idea.feasibility_score * (0.5 + 0.5 * concept_familiarity)
        
        # Impact score based on connection strength
        connection_strength = 0
        for connection in idea.connections:
            if '--' in connection:
                # Extract strength from connection (simplified)
                connection_strength += 0.5
        
        idea.impact_score = min(idea.impact_score * (1 + connection_strength * 0.2), 1.0)
        
        # Overall creativity score (weighted average)
        idea.creativity_score = (
            idea.novelty_score * 0.4 +
            idea.feasibility_score * 0.3 +
            idea.impact_score * 0.3
        )
    
    def _refine_idea(self, idea: CreativeIdea) -> CreativeIdea:
        """Refine an idea for better feasibility"""
        # Create a refined copy
        refined_idea = CreativeIdea(
            idea_id=f"refined_{idea.idea_id}",
            title=f"Refined: {idea.title}",
            description=idea.description,
            category=idea.category,
            technique=idea.technique,
            novelty_score=idea.novelty_score * 0.9,  # Slightly reduce novelty
            feasibility_score=min(idea.feasibility_score * 1.2, 1.0),  # Increase feasibility
            impact_score=idea.impact_score,
            creativity_score=0.0,  # Will be recalculated
            components=idea.components,
            connections=idea.connections,
            inspiration_sources=idea.inspiration_sources + ["refinement"]
        )
        
        self._calculate_creativity_scores(refined_idea)
        return refined_idea
    
    def _generate_process_steps(self, technique: CreativityTechnique, prompt: str) -> List[Dict[str, Any]]:
        """Generate process steps for creative process"""
        base_steps = [
            {"step": "analyze_prompt", "description": f"Analyzed prompt: {prompt[:50]}..."},
            {"step": "extract_concepts", "description": "Extracted key concepts"},
            {"step": "apply_technique", "description": f"Applied {technique.value} technique"},
            {"step": "generate_ideas", "description": "Generated creative ideas"},
            {"step": "evaluate_ideas", "description": "Evaluated idea quality"},
            {"step": "refine_output", "description": "Refined final ideas"}
        ]
        
        return base_steps
    
    def _evaluate_idea_set(self, ideas: List[CreativeIdea]) -> Dict[str, float]:
        """Evaluate a set of ideas"""
        if not ideas:
            return {}
        
        metrics = {
            'average_novelty': np.mean([idea.novelty_score for idea in ideas]),
            'average_feasibility': np.mean([idea.feasibility_score for idea in ideas]),
            'average_impact': np.mean([idea.impact_score for idea in ideas]),
            'average_creativity': np.mean([idea.creativity_score for idea in ideas]),
            'idea_diversity': len(set(idea.category for idea in ideas)) / len(IdeaCategory),
            'concept_coverage': len(set(comp for idea in ideas for comp in idea.components)) / len(self.concept_network.nodes())
        }
        
        return metrics
    
    def combine_ideas(self, ideas: List[CreativeIdea], combination_method: str = "merge") -> CreativeIdea:
        """
        Combine multiple ideas into a new creative idea
        
        Args:
            ideas: List of ideas to combine
            combination_method: How to combine ideas ('merge', 'hybrid', 'synthesis')
            
        Returns:
            CreativeIdea: Combined idea
        """
        if not ideas:
            raise ValueError("No ideas to combine")
        
        # Extract unique components
        all_components = list(set(comp for idea in ideas for comp in idea.components))
        
        # Generate combined description
        if combination_method == "merge":
            description = f"Combined idea merging: {' + '.join(idea.title for idea in ideas)}"
        elif combination_method == "hybrid":
            description = f"Hybrid approach combining elements from: {', '.join(idea.title for idea in ideas)}"
        else:  # synthesis
            description = f"Synthesized solution integrating: {' & '.join(idea.title for idea in ideas)}"
        
        # Create combined idea
        combined_idea = CreativeIdea(
            idea_id=f"combined_{int(time.time())}",
            title=f"Combined: {' + '.join(idea.title[:20] for idea in ideas)}",
            description=description,
            category=ideas[0].category,  # Use first idea's category
            technique=ideas[0].technique,  # Use first idea's technique
            novelty_score=np.mean([idea.novelty_score for idea in ideas]) * 1.1,  # Boost novelty
            feasibility_score=np.mean([idea.feasibility_score for idea in ideas]) * 0.9,  # Reduce feasibility
            impact_score=np.mean([idea.impact_score for idea in ideas]) * 1.1,  # Boost impact
            creativity_score=0.0,
            components=all_components,
            connections=[f"Combined from {len(ideas)} ideas"],
            inspiration_sources=[f"combination_{combination_method}"] + [f"source_{i}" for i in range(len(ideas))]
        )
        
        self._calculate_creativity_scores(combined_idea)
        self.idea_database[combined_idea.idea_id] = combined_idea
        
        return combined_idea
    
    def evaluate_creativity(self, idea: CreativeIdea) -> Dict[str, Any]:
        """
        Evaluate the creativity of an idea
        
        Args:
            idea: Idea to evaluate
            
        Returns:
            Dict: Creativity evaluation results
        """
        evaluation = {
            'idea_id': idea.idea_id,
            'novelty_analysis': {
                'score': idea.novelty_score,
                'assessment': self._assess_novelty(idea)
            },
            'feasibility_analysis': {
                'score': idea.feasibility_score,
                'assessment': self._assess_feasibility(idea)
            },
            'impact_analysis': {
                'score': idea.impact_score,
                'assessment': self._assess_impact(idea)
            },
            'overall_creativity': {
                'score': idea.creativity_score,
                'assessment': self._assess_overall_creativity(idea)
            },
            'strengths': self._identify_strengths(idea),
            'weaknesses': self._identify_weaknesses(idea),
            'improvement_suggestions': self._suggest_improvements(idea)
        }
        
        return evaluation
    
    def _assess_novelty(self, idea: CreativeIdea) -> str:
        """Assess novelty of an idea"""
        if idea.novelty_score > 0.8:
            return "Highly novel - breaks new ground"
        elif idea.novelty_score > 0.6:
            return "Moderately novel - fresh approach"
        elif idea.novelty_score > 0.4:
            return "Somewhat novel - minor innovations"
        else:
            return "Low novelty - conventional approach"
    
    def _assess_feasibility(self, idea: CreativeIdea) -> str:
        """Assess feasibility of an idea"""
        if idea.feasibility_score > 0.8:
            return "Highly feasible - easily implementable"
        elif idea.feasibility_score > 0.6:
            return "Moderately feasible - some challenges"
        elif idea.feasibility_score > 0.4:
            return "Somewhat feasible - significant challenges"
        else:
            return "Low feasibility - major obstacles"
    
    def _assess_impact(self, idea: CreativeIdea) -> str:
        """Assess impact of an idea"""
        if idea.impact_score > 0.8:
            return "High impact - transformative potential"
        elif idea.impact_score > 0.6:
            return "Moderate impact - significant improvements"
        elif idea.impact_score > 0.4:
            return "Some impact - incremental improvements"
        else:
            return "Low impact - minimal effects"
    
    def _assess_overall_creativity(self, idea: CreativeIdea) -> str:
        """Assess overall creativity of an idea"""
        if idea.creativity_score > 0.8:
            return "Exceptionally creative - breakthrough idea"
        elif idea.creativity_score > 0.6:
            return "Highly creative - innovative solution"
        elif idea.creativity_score > 0.4:
            return "Moderately creative - good idea"
        else:
            return "Limited creativity - conventional idea"
    
    def _identify_strengths(self, idea: CreativeIdea) -> List[str]:
        """Identify strengths of an idea"""
        strengths = []
        
        if idea.novelty_score > 0.7:
            strengths.append("High novelty and originality")
        if idea.feasibility_score > 0.7:
            strengths.append("Strong feasibility and practicality")
        if idea.impact_score > 0.7:
            strengths.append("Significant potential impact")
        if len(idea.components) > 3:
            strengths.append("Rich conceptual foundation")
        if len(idea.connections) > 2:
            strengths.append("Strong conceptual connections")
        
        return strengths if strengths else ["Basic concept with potential"]
    
    def _identify_weaknesses(self, idea: CreativeIdea) -> List[str]:
        """Identify weaknesses of an idea"""
        weaknesses = []
        
        if idea.novelty_score < 0.4:
            weaknesses.append("Limited novelty and originality")
        if idea.feasibility_score < 0.4:
            weaknesses.append("Significant feasibility challenges")
        if idea.impact_score < 0.4:
            weaknesses.append("Limited potential impact")
        if len(idea.components) < 2:
            weaknesses.append("Narrow conceptual scope")
        if not idea.connections:
            weaknesses.append("Weak conceptual connections")
        
        return weaknesses if weaknesses else ["Well-balanced concept"]
    
    def _suggest_improvements(self, idea: CreativeIdea) -> List[str]:
        """Suggest improvements for an idea"""
        suggestions = []
        
        if idea.novelty_score < 0.6:
            suggestions.append("Consider more unconventional approaches")
        if idea.feasibility_score < 0.6:
            suggestions.append("Break down into smaller, more manageable steps")
        if idea.impact_score < 0.6:
            suggestions.append("Focus on high-impact applications")
        if len(idea.components) < 3:
            suggestions.append("Incorporate additional relevant concepts")
        if len(idea.connections) < 2:
            suggestions.append("Explore more conceptual connections")
        
        return suggestions if suggestions else ["Well-developed idea with good balance"]
    
    def get_creativity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive creativity statistics"""
        return {
            'total_ideas_generated': len(self.idea_database),
            'concepts_in_network': len(self.concept_network.nodes()),
            'connections_in_network': len(self.concept_network.edges()),
            'creative_processes_completed': len(self.creative_processes),
            'techniques_used': list(set(p.technique.value for p in self.creative_processes)),
            'idea_categories': list(set(idea.category.value for idea in self.idea_database.values())),
            'average_creativity_score': np.mean([idea.creativity_score for idea in self.idea_database.values()]) if self.idea_database else 0,
            'top_performing_techniques': self._get_top_techniques(),
            'creivity_trends': self._analyze_creativity_trends()
        }
    
    def _get_top_techniques(self) -> List[Dict[str, Any]]:
        """Get top performing creativity techniques"""
        technique_performance = defaultdict(list)
        
        for process in self.creative_processes:
            if process.ideas_generated:
                avg_creativity = np.mean([idea.creativity_score for idea in process.ideas_generated])
                technique_performance[process.technique].append(avg_creativity)
        
        top_techniques = []
        for technique, scores in technique_performance.items():
            if scores:
                top_techniques.append({
                    'technique': technique.value,
                    'average_score': np.mean(scores),
                    'usage_count': len(scores)
                })
        
        return sorted(top_techniques, key=lambda x: x['average_score'], reverse=True)[:5]
    
    def _analyze_creativity_trends(self) -> Dict[str, Any]:
        """Analyze creativity trends over time"""
        if not self.creative_processes:
            return {}
        
        # Group processes by time periods
        recent_processes = list(self.creative_processes)[-50:]  # Last 50 processes
        
        if len(recent_processes) < 10:
            return {'insufficient_data': True}
        
        # Calculate trends
        creativity_scores = []
        novelty_scores = []
        feasibility_scores = []
        
        for process in recent_processes:
            if process.ideas_generated:
                avg_creativity = np.mean([idea.creativity_score for idea in process.ideas_generated])
                avg_novelty = np.mean([idea.novelty_score for idea in process.ideas_generated])
                avg_feasibility = np.mean([idea.feasibility_score for idea in process.ideas_generated])
                
                creativity_scores.append(avg_creativity)
                novelty_scores.append(avg_novelty)
                feasibility_scores.append(avg_feasibility)
        
        return {
            'creativity_trend': 'increasing' if creativity_scores[-1] > creativity_scores[0] else 'decreasing',
            'novelty_trend': 'increasing' if novelty_scores[-1] > novelty_scores[0] else 'decreasing',
            'feasibility_trend': 'increasing' if feasibility_scores[-1] > feasibility_scores[0] else 'decreasing',
            'average_creativity': np.mean(creativity_scores),
            'creativity_volatility': np.std(creativity_scores)
        }
    
    def save_creative_data(self, filepath: str):
        """Save creative data to file"""
        try:
            data = {
                'idea_database': {
                    idea_id: {
                        'title': idea.title,
                        'description': idea.description,
                        'category': idea.category.value,
                        'technique': idea.technique.value,
                        'novelty_score': idea.novelty_score,
                        'feasibility_score': idea.feasibility_score,
                        'impact_score': idea.impact_score,
                        'creativity_score': idea.creativity_score,
                        'components': idea.components,
                        'connections': idea.connections,
                        'inspiration_sources': idea.inspiration_sources,
                        'timestamp': idea.timestamp.isoformat()
                    }
                    for idea_id, idea in self.idea_database.items()
                },
                'concept_network_data': {
                    'nodes': list(self.concept_network.nodes(data=True)),
                    'edges': list(self.concept_network.edges(data=True))
                },
                'creativity_patterns': dict(self.creativity_patterns)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Creative data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving creative data: {e}")
    
    def load_creative_data(self, filepath: str):
        """Load creative data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load idea database
            self.idea_database = {}
            for idea_id, idea_data in data['idea_database'].items():
                self.idea_database[idea_id] = CreativeIdea(
                    idea_id=idea_id,
                    title=idea_data['title'],
                    description=idea_data['description'],
                    category=IdeaCategory(idea_data['category']),
                    technique=CreativityTechnique(idea_data['technique']),
                    novelty_score=idea_data['novelty_score'],
                    feasibility_score=idea_data['feasibility_score'],
                    impact_score=idea_data['impact_score'],
                    creativity_score=idea_data['creativity_score'],
                    components=idea_data['components'],
                    connections=idea_data['connections'],
                    inspiration_sources=idea_data['inspiration_sources'],
                    timestamp=datetime.fromisoformat(idea_data['timestamp'])
                )
            
            # Load concept network
            self.concept_network = nx.Graph()
            for node_data in data['concept_network_data']['nodes']:
                self.concept_network.add_node(node_data[0], **node_data[1])
            
            for edge_data in data['concept_network_data']['edges']:
                self.concept_network.add_edge(edge_data[0], edge_data[1], **edge_data[2])
            
            # Load creativity patterns
            self.creativity_patterns = defaultdict(list, data['creativity_patterns'])
            
            logger.info(f"Creative data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading creative data: {e}")

# Example usage and demonstration
def demonstrate_creative_thinking():
    """Demonstrate the creative thinking capabilities"""
    print("=== Creative Thinking Engine Demonstration ===")
    
    # Initialize creative thinking engine
    creative_engine = CreativeThinkingEngine()
    
    # Test prompts
    test_prompts = [
        "How can we make cities more sustainable?",
        "What new technologies could help with education?",
        "How might we improve mental health support?",
        "What innovative approaches could reduce waste?",
        "How can we make work more fulfilling?"
    ]
    
    # Test different creativity techniques
    techniques = [
        CreativityTechnique.BRAINSTORMING,
        CreativityTechnique.MIND_MAPPING,
        CreativityTechnique.SCAMPER,
        CreativityTechnique.LATERAL_THINKING,
        CreativityTechnique.ANALOGICAL_THINKING
    ]
    
    print("\nGenerating creative ideas...")
    
    for i, prompt in enumerate(test_prompts[:2]):  # Test with first 2 prompts
        print(f"\nPrompt: {prompt}")
        print("=" * 60)
        
        for technique in techniques[:2]:  # Test with first 2 techniques
            print(f"\nTechnique: {technique.value}")
            print("-" * 40)
            
            # Generate ideas
            ideas = creative_engine.generate_creative_ideas(
                prompt=prompt,
                technique=technique,
                num_ideas=3
            )
            
            # Display ideas
            for j, idea in enumerate(ideas):
                print(f"\nIdea {j+1}:")
                print(f"  Title: {idea.title}")
                print(f"  Description: {idea.description}")
                print(f"  Category: {idea.category.value}")
                print(f"  Novelty: {idea.novelty_score:.3f}")
                print(f"  Feasibility: {idea.feasibility_score:.3f}")
                print(f"  Impact: {idea.impact_score:.3f}")
                print(f"  Creativity: {idea.creativity_score:.3f}")
                print(f"  Components: {', '.join(idea.components)}")
    
    # Test idea combination
    print("\n=== Idea Combination Test ===")
    if len(creative_engine.idea_database) >= 2:
        # Get two random ideas
        idea_ids = list(creative_engine.idea_database.keys())
        ideas_to_combine = [creative_engine.idea_database[idea_ids[0]], creative_engine.idea_database[idea_ids[1]]]
        
        combined_idea = creative_engine.combine_ideas(ideas_to_combine)
        
        print(f"Combined Idea: {combined_idea.title}")
        print(f"Description: {combined_idea.description}")
        print(f"Creativity Score: {combined_idea.creativity_score:.3f}")
    
    # Test creativity evaluation
    print("\n=== Creativity Evaluation Test ===")
    if creative_engine.idea_database:
        # Evaluate a random idea
        sample_idea = next(iter(creative_engine.idea_database.values()))
        evaluation = creative_engine.evaluate_creativity(sample_idea)
        
        print(f"Evaluating idea: {sample_idea.title}")
        print(f"Novelty Assessment: {evaluation['novelty_analysis']['assessment']}")
        print(f"Feasibility Assessment: {evaluation['feasibility_analysis']['assessment']}")
        print(f"Impact Assessment: {evaluation['impact_analysis']['assessment']}")
        print(f"Overall Creativity: {evaluation['overall_creativity']['assessment']}")
        print(f"Strengths: {', '.join(evaluation['strengths'])}")
        print(f"Suggestions: {', '.join(evaluation['improvement_suggestions'])}")
    
    # Show statistics
    print("\n=== Creativity Statistics ===")
    stats = creative_engine.get_creativity_statistics()
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
        elif isinstance(value, dict):
            if 'insufficient_data' in value:
                print(f"{key}: Insufficient data")
            else:
                print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    # Save creative data
    creative_engine.save_creative_data('creative_data.json')
    print("\nCreative data saved successfully!")

if __name__ == "__main__":
    demonstrate_creative_thinking()