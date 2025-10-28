"""
Advanced Natural Language Understanding System for Bharat-FM MLOps Platform

This module implements sophisticated NLU capabilities that enable the AI to:
- Understand complex linguistic structures and semantics
- Process nuanced language including sarcasm, irony, and metaphors
- Handle multiple languages and dialects
- Extract deep meaning from context and subtext
- Perform advanced text analysis and comprehension

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
import re
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx
from pathlib import Path
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinguisticComplexity(Enum):
    """Levels of linguistic complexity"""
    BASIC = "basic"  # Simple sentences, clear meaning
    INTERMEDIATE = "intermediate"  # Compound sentences, some nuance
    ADVANCED = "advanced"  # Complex structures, multiple meanings
    EXPERT = "expert"  # Sophisticated language, deep subtext

class SemanticRole(Enum):
    """Semantic roles in language"""
    AGENT = "agent"  # The doer of the action
    PATIENT = "patient"  # The receiver of the action
    INSTRUMENT = "instrument"  # The means used to perform the action
    BENEFICIARY = "beneficiary"  # The one who benefits
    LOCATION = "location"  # Where the action takes place
    TIME = "time"  # When the action takes place
    MANNER = "manner"  # How the action is performed
    PURPOSE = "purpose"  # Why the action is performed

class PragmaticAspect(Enum):
    """Pragmatic aspects of language"""
    LITERAL = "literal"  # Direct, explicit meaning
    IRONY = "irony"  # Saying the opposite of what is meant
    SARCASM = "sarcasm"  # Mocking or contemptuous irony
    METAPHOR = "metaphor"  # Figurative comparison
    HYPERBOLE = "hyperbole"  # Exaggeration for effect
    UNDERSTATEMENT = "understatement"  # Downplaying for effect
    EUPHEMISM = "euphemism"  # Mild expression for something unpleasant

@dataclass
class LinguisticFeature:
    """Represents a linguistic feature extracted from text"""
    feature_id: str
    feature_type: str
    value: Any
    confidence: float
    span: Tuple[int, int]  # Character span in text
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticFrame:
    """Represents a semantic frame (meaning representation)"""
    frame_id: str
    frame_name: str
    frame_elements: Dict[str, Any]  # role -> filler
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiscourseStructure:
    """Represents the structure of discourse"""
    structure_id: str
    discourse_type: str
    segments: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    coherence_score: float
    topic_flow: List[str]

@dataclass
class PragmaticInterpretation:
    """Represents pragmatic interpretation of text"""
    interpretation_id: str
    literal_meaning: str
    intended_meaning: str
    pragmatic_aspect: PragmaticAspect
    confidence: float
    context_clues: List[str]
    social_context: Dict[str, Any]

@dataclass
class LanguageUnderstanding:
    """Represents comprehensive language understanding"""
    understanding_id: str
    text: str
    linguistic_features: List[LinguisticFeature]
    semantic_frames: List[SemanticFrame]
    discourse_structure: Optional[DiscourseStructure]
    pragmatic_interpretation: Optional[PragmaticInterpretation]
    complexity_level: LinguisticComplexity
    confidence: float
    processing_time: timedelta
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedNLUEngine:
    """
    Advanced Natural Language Understanding engine
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # NLU parameters
        self.max_text_length = self.config.get('max_text_length', 4096)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.complexity_threshold = self.config.get('complexity_threshold', 0.6)
        
        # Data structures
        self.understanding_history = deque(maxlen=1000)
        self.linguistic_patterns = defaultdict(list)
        self.semantic_network = nx.DiGraph()
        self.pragmatic_rules = {}
        
        # Language models
        self.tokenizer = None
        self.semantic_model = None
        self.syntax_model = None
        self.pragmatic_model = None
        
        # NLP components
        self.nlp = None
        self.tfidf_vectorizer = None
        self.pca = None
        
        # Initialize components
        self._initialize_nlu_components()
        self._build_semantic_network()
        self._load_pragmatic_rules()
        
        logger.info("Advanced NLU Engine initialized successfully")
    
    def _initialize_nlu_components(self):
        """Initialize NLU components and models"""
        try:
            # Load spaCy for basic NLP
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, using basic processing")
                self.nlp = spacy.blank("en")
            
            # Load transformer models
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.semantic_model = AutoModel.from_pretrained(model_name)
            
            # Load syntax analysis model
            syntax_model_name = "textattack/bert-base-uncased-CoLA"
            self.syntax_model = AutoModelForSequenceClassification.from_pretrained(syntax_model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.semantic_model.to(self.device)
            self.syntax_model.to(self.device)
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            # Initialize PCA for dimensionality reduction
            self.pca = PCA(n_components=100)
            
            logger.info("NLU models and components loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NLU components: {e}")
            # Initialize with None for fallback processing
            self.tokenizer = None
            self.semantic_model = None
            self.syntax_model = None
            self.nlp = spacy.blank("en")
    
    def _build_semantic_network(self):
        """Build semantic network for word relationships"""
        # Core semantic relationships
        semantic_relations = [
            ("run", "move", "is_a"),
            ("walk", "move", "is_a"),
            ("fly", "move", "is_a"),
            ("eat", "consume", "is_a"),
            ("drink", "consume", "is_a"),
            ("happy", "emotion", "is_a"),
            ("sad", "emotion", "is_a"),
            ("angry", "emotion", "is_a"),
            ("dog", "animal", "is_a"),
            ("cat", "animal", "is_a"),
            ("car", "vehicle", "is_a"),
            ("bike", "vehicle", "is_a"),
            ("house", "building", "is_a"),
            ("office", "building", "is_a"),
            ("book", "read", "used_for"),
            ("knife", "cut", "used_for"),
            ("pen", "write", "used_for"),
            ("key", "open", "used_for"),
            ("hot", "temperature", "has_property"),
            ("cold", "temperature", "has_property"),
            ("big", "size", "has_property"),
            ("small", "size", "has_property")
        ]
        
        # Add nodes and edges to semantic network
        for source, target, relation in semantic_relations:
            self.semantic_network.add_node(source, type="concept")
            self.semantic_network.add_node(target, type="concept")
            self.semantic_network.add_edge(source, target, relation=relation)
        
        logger.info(f"Built semantic network with {len(self.semantic_network.nodes())} concepts")
    
    def _load_pragmatic_rules(self):
        """Load pragmatic rules for language interpretation"""
        self.pragmatic_rules = {
            "irony_detection": {
                "indicators": ["yeah right", "as if", "sure", "obviously"],
                "context_clues": ["contradiction", "exaggeration", "incongruity"],
                "confidence_threshold": 0.8
            },
            "sarcasm_detection": {
                "indicators": ["thanks a lot", "great job", "perfect", "wonderful"],
                "context_clues": ["negative_situation", "mocking_tone", "exaggeration"],
                "confidence_threshold": 0.85
            },
            "metaphor_detection": {
                "patterns": ["is a", "like a", "as a", "metaphor"],
                "context_clues": "figurative_language",
                "confidence_threshold": 0.7
            },
            "hyperbole_detection": {
                "indicators": ["million", "billion", "forever", "never", "always"],
                "context_clues": "exaggeration",
                "confidence_threshold": 0.75
            }
        }
        
        logger.info("Loaded pragmatic rules for language interpretation")
    
    def understand_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> LanguageUnderstanding:
        """
        Perform comprehensive natural language understanding
        
        Args:
            text: Text to understand
            context: Additional context
            
        Returns:
            LanguageUnderstanding: Comprehensive understanding result
        """
        start_time = datetime.now()
        understanding_id = f"understanding_{int(time.time())}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        
        # Build semantic frames
        semantic_frames = self._build_semantic_frames(text, linguistic_features)
        
        # Analyze discourse structure
        discourse_structure = self._analyze_discourse_structure(text)
        
        # Interpret pragmatics
        pragmatic_interpretation = self._interpret_pragmatics(text, context)
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(text, linguistic_features)
        
        # Calculate overall confidence
        confidence = self._calculate_understanding_confidence(
            linguistic_features, semantic_frames, discourse_structure, pragmatic_interpretation
        )
        
        # Calculate processing time
        processing_time = datetime.now() - start_time
        
        # Create understanding object
        understanding = LanguageUnderstanding(
            understanding_id=understanding_id,
            text=text,
            linguistic_features=linguistic_features,
            semantic_frames=semantic_frames,
            discourse_structure=discourse_structure,
            pragmatic_interpretation=pragmatic_interpretation,
            complexity_level=complexity_level,
            confidence=confidence,
            processing_time=processing_time,
            metadata={"context": context or {}}
        )
        
        # Store in history
        self.understanding_history.append(understanding)
        
        # Update linguistic patterns
        self._update_linguistic_patterns(understanding)
        
        logger.info(f"Understood text with complexity {complexity_level.value} and confidence {confidence:.3f}")
        
        return understanding
    
    def _extract_linguistic_features(self, text: str) -> List[LinguisticFeature]:
        """Extract linguistic features from text"""
        features = []
        
        # Basic text statistics
        features.extend(self._extract_text_statistics(text))
        
        # Syntactic features
        features.extend(self._extract_syntactic_features(text))
        
        # Semantic features
        features.extend(self._extract_semantic_features(text))
        
        # Lexical features
        features.extend(self._extract_lexical_features(text))
        
        # Stylistic features
        features.extend(self._extract_stylistic_features(text))
        
        return features
    
    def _extract_text_statistics(self, text: str) -> List[LinguisticFeature]:
        """Extract basic text statistics"""
        features = []
        
        # Character-level statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        features.append(LinguisticFeature(
            feature_id=f"char_count_{int(time.time())}",
            feature_type="text_statistic",
            value=char_count,
            confidence=1.0,
            span=(0, len(text))
        ))
        
        features.append(LinguisticFeature(
            feature_id=f"word_count_{int(time.time())}",
            feature_type="text_statistic",
            value=word_count,
            confidence=1.0,
            span=(0, len(text))
        ))
        
        features.append(LinguisticFeature(
            feature_id=f"sentence_count_{int(time.time())}",
            feature_type="text_statistic",
            value=sentence_count,
            confidence=1.0,
            span=(0, len(text))
        ))
        
        # Average word length
        if word_count > 0:
            avg_word_length = sum(len(word) for word in text.split()) / word_count
            features.append(LinguisticFeature(
                feature_id=f"avg_word_length_{int(time.time())}",
                feature_type="text_statistic",
                value=avg_word_length,
                confidence=1.0,
                span=(0, len(text))
            ))
        
        return features
    
    def _extract_syntactic_features(self, text: str) -> List[LinguisticFeature]:
        """Extract syntactic features"""
        features = []
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Part-of-speech distribution
                pos_counts = defaultdict(int)
                for token in doc:
                    pos_counts[token.pos_] += 1
                
                for pos, count in pos_counts.items():
                    features.append(LinguisticFeature(
                        feature_id=f"pos_{pos}_{int(time.time())}",
                        feature_type="syntactic",
                        value=count,
                        confidence=0.9,
                        span=(0, len(text))
                    ))
                
                # Dependency relations
                dep_counts = defaultdict(int)
                for token in doc:
                    dep_counts[token.dep_] += 1
                
                for dep, count in dep_counts.items():
                    features.append(LinguisticFeature(
                        feature_id=f"dep_{dep}_{int(time.time())}",
                        feature_type="syntactic",
                        value=count,
                        confidence=0.8,
                        span=(0, len(text))
                    ))
                
                # Parse tree depth
                if list(doc.sents):
                    max_depth = max(len(list(sent.root.ancestors)) for sent in doc.sents)
                    features.append(LinguisticFeature(
                        feature_id=f"parse_depth_{int(time.time())}",
                        feature_type="syntactic",
                        value=max_depth,
                        confidence=0.8,
                        span=(0, len(text))
                    ))
                
            except Exception as e:
                logger.error(f"Error extracting syntactic features: {e}")
        
        return features
    
    def _extract_semantic_features(self, text: str) -> List[LinguisticFeature]:
        """Extract semantic features"""
        features = []
        
        # Named entities
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    features.append(LinguisticFeature(
                        feature_id=f"entity_{ent.label_}_{int(time.time())}",
                        feature_type="semantic",
                        value={"text": ent.text, "label": ent.label_},
                        confidence=0.8,
                        span=(ent.start_char, ent.end_char)
                    ))
                
            except Exception as e:
                logger.error(f"Error extracting semantic features: {e}")
        
        # Semantic similarity to concepts
        text_lower = text.lower()
        for concept in self.semantic_network.nodes():
            if concept in text_lower:
                features.append(LinguisticFeature(
                    feature_id=f"concept_{concept}_{int(time.time())}",
                    feature_type="semantic",
                    value=concept,
                    confidence=0.7,
                    span=(text_lower.find(concept), text_lower.find(concept) + len(concept))
                ))
        
        return features
    
    def _extract_lexical_features(self, text: str) -> List[LinguisticFeature]:
        """Extract lexical features"""
        features = []
        
        # Vocabulary richness
        words = text.lower().split()
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0
        
        features.append(LinguisticFeature(
            feature_id=f"vocab_richness_{int(time.time())}",
            feature_type="lexical",
            value=vocabulary_richness,
            confidence=1.0,
            span=(0, len(text))
        ))
        
        # Readability scores (simplified)
        avg_sentence_length = len(words) / max(len(re.split(r'[.!?]+', text)), 1)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        readability_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length
        
        features.append(LinguisticFeature(
            feature_id=f"readability_{int(time.time())}",
            feature_type="lexical",
            value=readability_score,
            confidence=0.9,
            span=(0, len(text))
        ))
        
        return features
    
    def _extract_stylistic_features(self, text: str) -> List[LinguisticFeature]:
        """Extract stylistic features"""
        features = []
        
        # Formality indicators
        formal_indicators = ['therefore', 'however', 'furthermore', 'consequently', 'nevertheless']
        informal_indicators = ['yeah', 'okay', 'like', 'totally', 'awesome']
        
        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())
        
        formality_score = (formal_count - informal_count) / max(formal_count + informal_count, 1)
        
        features.append(LinguisticFeature(
            feature_id=f"formality_{int(time.time())}",
            feature_type="stylistic",
            value=formality_score,
            confidence=0.8,
            span=(0, len(text))
        ))
        
        # Emotion indicators
        emotion_words = {
            'positive': ['happy', 'joy', 'love', 'excellent', 'wonderful', 'amazing'],
            'negative': ['sad', 'angry', 'hate', 'terrible', 'awful', 'horrible']
        }
        
        for emotion, words in emotion_words.items():
            count = sum(1 for word in words if word in text.lower())
            features.append(LinguisticFeature(
                feature_id=f"emotion_{emotion}_{int(time.time())}",
                feature_type="stylistic",
                value=count,
                confidence=0.7,
                span=(0, len(text))
            ))
        
        return features
    
    def _build_semantic_frames(self, text: str, linguistic_features: List[LinguisticFeature]) -> List[SemanticFrame]:
        """Build semantic frames from text"""
        frames = []
        
        # Simple frame construction based on linguistic features
        # In production, this would use more sophisticated frame semantics
        
        # Extract potential actions and their participants
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                for sent in doc.sents:
                    # Find main verb
                    main_verb = None
                    for token in sent:
                        if token.dep_ == "ROOT" and token.pos_ == "VERB":
                            main_verb = token
                            break
                    
                    if main_verb:
                        # Create frame for this verb
                        frame_elements = {}
                        
                        # Find agent (subject)
                        for token in sent:
                            if token.head == main_verb and token.dep_ in ["nsubj", "nsubjpass"]:
                                frame_elements[SemanticRole.AGENT.value] = token.text
                        
                        # Find patient (object)
                        for token in sent:
                            if token.head == main_verb and token.dep_ in ["dobj", "pobj"]:
                                frame_elements[SemanticRole.PATIENT.value] = token.text
                        
                        # Find other elements
                        for token in sent:
                            if token.head == main_verb:
                                if token.dep_ == "prep":
                                    frame_elements[SemanticRole.INSTRUMENT.value] = token.text
                                elif token.dep_ == "attr":
                                    frame_elements[SemanticRole.MANNER.value] = token.text
                        
                        frame = SemanticFrame(
                            frame_id=f"frame_{int(time.time())}_{len(frames)}",
                            frame_name=main_verb.lemma_,
                            frame_elements=frame_elements,
                            confidence=0.7
                        )
                        frames.append(frame)
                
            except Exception as e:
                logger.error(f"Error building semantic frames: {e}")
        
        return frames
    
    def _analyze_discourse_structure(self, text: str) -> Optional[DiscourseStructure]:
        """Analyze discourse structure of text"""
        try:
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return None
            
            # Create segments
            segments = []
            for i, sentence in enumerate(sentences):
                segments.append({
                    "id": f"seg_{i}",
                    "text": sentence,
                    "position": i,
                    "length": len(sentence)
                })
            
            # Analyze relations between segments (simplified)
            relations = []
            for i in range(len(segments) - 1):
                relations.append({
                    "source": segments[i]["id"],
                    "target": segments[i + 1]["id"],
                    "relation": "sequence",
                    "confidence": 0.8
                })
            
            # Calculate coherence score
            coherence_score = self._calculate_coherence_score(sentences)
            
            # Extract topic flow
            topic_flow = self._extract_topic_flow(sentences)
            
            return DiscourseStructure(
                structure_id=f"discourse_{int(time.time())}",
                discourse_type="narrative",
                segments=segments,
                relations=relations,
                coherence_score=coherence_score,
                topic_flow=topic_flow
            )
            
        except Exception as e:
            logger.error(f"Error analyzing discourse structure: {e}")
            return None
    
    def _calculate_coherence_score(self, sentences: List[str]) -> float:
        """Calculate coherence score for text"""
        if len(sentences) < 2:
            return 1.0
        
        # Calculate semantic similarity between consecutive sentences
        similarities = []
        
        for i in range(len(sentences) - 1):
            sim = self._calculate_semantic_similarity(sentences[i], sentences[i + 1])
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.semantic_model and self.tokenizer:
            try:
                # Encode texts
                inputs1 = self.tokenizer(text1, return_tensors="pt", truncation=True, max_length=512)
                inputs2 = self.tokenizer(text2, return_tensors="pt", truncation=True, max_length=512)
                
                inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
                inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs1 = self.semantic_model(**inputs1)
                    outputs2 = self.semantic_model(**inputs2)
                    
                    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
                    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(embeddings1, embeddings2).item()
                return similarity
                
            except Exception as e:
                logger.error(f"Error calculating semantic similarity: {e}")
        
        # Fallback: simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_topic_flow(self, sentences: List[str]) -> List[str]:
        """Extract topic flow through text"""
        topics = []
        
        for sentence in sentences:
            # Simple topic extraction (most frequent content words)
            words = sentence.lower().split()
            content_words = [word for word in words if len(word) > 3 and word.isalpha()]
            
            if content_words:
                # Use most frequent word as topic (simplified)
                topic = max(set(content_words), key=content_words.count)
                topics.append(topic)
            else:
                topics.append("general")
        
        return topics
    
    def _interpret_pragmatics(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[PragmaticInterpretation]:
        """Interpret pragmatic aspects of text"""
        text_lower = text.lower()
        
        # Check for irony
        irony_score = self._detect_irony(text_lower, context)
        if irony_score > self.pragmatic_rules["irony_detection"]["confidence_threshold"]:
            return PragmaticInterpretation(
                interpretation_id=f"pragmatic_{int(time.time())}",
                literal_meaning=text,
                intended_meaning=self._generate_ironic_meaning(text),
                pragmatic_aspect=PragmaticAspect.IRONY,
                confidence=irony_score,
                context_clues=self._find_context_clues(text_lower, "irony"),
                social_context=context or {}
            )
        
        # Check for sarcasm
        sarcasm_score = self._detect_sarcasm(text_lower, context)
        if sarcasm_score > self.pragmatic_rules["sarcasm_detection"]["confidence_threshold"]:
            return PragmaticInterpretation(
                interpretation_id=f"pragmatic_{int(time.time())}",
                literal_meaning=text,
                intended_meaning=self._generate_sarcastic_meaning(text),
                pragmatic_aspect=PragmaticAspect.SARCASM,
                confidence=sarcasm_score,
                context_clues=self._find_context_clues(text_lower, "sarcasm"),
                social_context=context or {}
            )
        
        # Check for metaphor
        metaphor_score = self._detect_metaphor(text_lower, context)
        if metaphor_score > self.pragmatic_rules["metaphor_detection"]["confidence_threshold"]:
            return PragmaticInterpretation(
                interpretation_id=f"pragmatic_{int(time.time())}",
                literal_meaning=text,
                intended_meaning=self._generate_metaphorical_meaning(text),
                pragmatic_aspect=PragmaticAspect.METAPHOR,
                confidence=metaphor_score,
                context_clues=self._find_context_clues(text_lower, "metaphor"),
                social_context=context or {}
            )
        
        # Default to literal interpretation
        return PragmaticInterpretation(
            interpretation_id=f"pragmatic_{int(time.time())}",
            literal_meaning=text,
            intended_meaning=text,
            pragmatic_aspect=PragmaticAspect.LITERAL,
            confidence=0.9,
            context_clues=[],
            social_context=context or {}
        )
    
    def _detect_irony(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Detect irony in text"""
        irony_score = 0.0
        
        # Check for irony indicators
        indicators = self.pragmatic_rules["irony_detection"]["indicators"]
        indicator_count = sum(1 for indicator in indicators if indicator in text)
        
        if indicator_count > 0:
            irony_score += 0.3
        
        # Check for context clues
        context_clues = self.pragmatic_rules["irony_detection"]["context_clues"]
        if "contradiction" in context_clues:
            # Look for contradictory statements
            if self._has_contradiction(text):
                irony_score += 0.4
        
        if "exaggeration" in context_clues:
            # Look for exaggeration
            if self._has_exaggeration(text):
                irony_score += 0.3
        
        return min(irony_score, 1.0)
    
    def _detect_sarcasm(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Detect sarcasm in text"""
        sarcasm_score = 0.0
        
        # Check for sarcasm indicators
        indicators = self.pragmatic_rules["sarcasm_detection"]["indicators"]
        indicator_count = sum(1 for indicator in indicators if indicator in text)
        
        if indicator_count > 0:
            sarcasm_score += 0.4
        
        # Check for mocking tone
        if self._has_mocking_tone(text):
            sarcasm_score += 0.3
        
        # Check for negative situation with positive words
        if self._has_incongruity(text):
            sarcasm_score += 0.3
        
        return min(sarcasm_score, 1.0)
    
    def _detect_metaphor(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Detect metaphor in text"""
        metaphor_score = 0.0
        
        # Check for metaphor patterns
        patterns = self.pragmatic_rules["metaphor_detection"]["patterns"]
        for pattern in patterns:
            if pattern in text:
                metaphor_score += 0.4
        
        # Check for figurative language
        if self._has_figurative_language(text):
            metaphor_score += 0.3
        
        # Check for abstract concepts described concretely
        if self._has_abstract_concrete_mapping(text):
            metaphor_score += 0.3
        
        return min(metaphor_score, 1.0)
    
    def _has_contradiction(self, text: str) -> bool:
        """Check if text contains contradiction"""
        contradiction_words = ["but", "however", "although", "despite", "yet"]
        return any(word in text for word in contradiction_words)
    
    def _has_exaggeration(self, text: str) -> bool:
        """Check if text contains exaggeration"""
        exaggeration_words = ["million", "billion", "forever", "never", "always", "every", "all"]
        return any(word in text for word in exaggeration_words)
    
    def _has_mocking_tone(self, text: str) -> bool:
        """Check if text has mocking tone"""
        mocking_words = ["yeah right", "as if", "sure", "obviously", "great job"]
        return any(phrase in text for phrase in mocking_words)
    
    def _has_incongruity(self, text: str) -> bool:
        """Check if text has incongruity"""
        # Simple check for positive words in negative context or vice versa
        positive_words = ["good", "great", "excellent", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disgusting"]
        
        has_positive = any(word in text for word in positive_words)
        has_negative = any(word in text for word in negative_words)
        
        return has_positive and has_negative
    
    def _has_figurative_language(self, text: str) -> bool:
        """Check if text uses figurative language"""
        figurative_patterns = ["like a", "as a", "is a", "metaphor", "simile"]
        return any(pattern in text for pattern in figurative_patterns)
    
    def _has_abstract_concrete_mapping(self, text: str) -> bool:
        """Check if text maps abstract concepts to concrete ones"""
        abstract_concepts = ["love", "hate", "fear", "joy", "time", "death"]
        concrete_concepts = ["rock", "tree", "river", "mountain", "house", "car"]
        
        has_abstract = any(concept in text for concept in abstract_concepts)
        has_concrete = any(concept in text for concept in concrete_concepts)
        
        return has_abstract and has_concrete
    
    def _find_context_clues(self, text: str, aspect: str) -> List[str]:
        """Find context clues for pragmatic aspect"""
        clues = []
        
        if aspect == "irony":
            if self._has_contradiction(text):
                clues.append("contradiction")
            if self._has_exaggeration(text):
                clues.append("exaggeration")
        
        elif aspect == "sarcasm":
            if self._has_mocking_tone(text):
                clues.append("mocking_tone")
            if self._has_incongruity(text):
                clues.append("incongruity")
        
        elif aspect == "metaphor":
            if self._has_figurative_language(text):
                clues.append("figurative_language")
            if self._has_abstract_concrete_mapping(text):
                clues.append("abstract_concrete_mapping")
        
        return clues
    
    def _generate_ironic_meaning(self, text: str) -> str:
        """Generate ironic meaning from literal text"""
        # Simple ironic meaning generation (opposite of literal)
        if "good" in text.lower():
            return "This is actually bad"
        elif "great" in text.lower():
            return "This is actually terrible"
        else:
            return "The opposite of what was said"
    
    def _generate_sarcastic_meaning(self, text: str) -> str:
        """Generate sarcastic meaning from literal text"""
        # Simple sarcastic meaning generation
        return f"Sarcastic version of: {text}"
    
    def _generate_metaphorical_meaning(self, text: str) -> str:
        """Generate metaphorical meaning from literal text"""
        # Simple metaphorical meaning generation
        return f"Metaphorical interpretation of: {text}"
    
    def _determine_complexity_level(self, text: str, linguistic_features: List[LinguisticFeature]) -> LinguisticComplexity:
        """Determine linguistic complexity level of text"""
        complexity_score = 0.0
        
        # Analyze features for complexity indicators
        for feature in linguistic_features:
            if feature.feature_type == "syntactic":
                if "parse_depth" in feature.feature_id:
                    if feature.value > 3:
                        complexity_score += 0.2
                if "dep_" in feature.feature_id:
                    complexity_score += 0.1
            
            elif feature.feature_type == "lexical":
                if "vocab_richness" in feature.feature_id:
                    if feature.value > 0.7:
                        complexity_score += 0.2
                if "readability" in feature.feature_id:
                    if feature.value < 30:  # Low readability score indicates high complexity
                        complexity_score += 0.2
            
            elif feature.feature_type == "semantic":
                if "entity_" in feature.feature_id:
                    complexity_score += 0.1
        
        # Determine complexity level based on score
        if complexity_score < 0.3:
            return LinguisticComplexity.BASIC
        elif complexity_score < 0.6:
            return LinguisticComplexity.INTERMEDIATE
        elif complexity_score < 0.9:
            return LinguisticComplexity.ADVANCED
        else:
            return LinguisticComplexity.EXPERT
    
    def _calculate_understanding_confidence(self, linguistic_features: List[LinguisticFeature],
                                          semantic_frames: List[SemanticFrame],
                                          discourse_structure: Optional[DiscourseStructure],
                                          pragmatic_interpretation: Optional[PragmaticInterpretation]) -> float:
        """Calculate overall confidence in understanding"""
        confidence_components = []
        
        # Confidence from linguistic features
        if linguistic_features:
            avg_feature_confidence = np.mean([f.confidence for f in linguistic_features])
            confidence_components.append(avg_feature_confidence)
        
        # Confidence from semantic frames
        if semantic_frames:
            avg_frame_confidence = np.mean([f.confidence for f in semantic_frames])
            confidence_components.append(avg_frame_confidence)
        
        # Confidence from discourse structure
        if discourse_structure:
            confidence_components.append(discourse_structure.coherence_score)
        
        # Confidence from pragmatic interpretation
        if pragmatic_interpretation:
            confidence_components.append(pragmatic_interpretation.confidence)
        
        # Calculate overall confidence
        if confidence_components:
            overall_confidence = np.mean(confidence_components)
        else:
            overall_confidence = 0.5
        
        return min(overall_confidence, 1.0)
    
    def _update_linguistic_patterns(self, understanding: LanguageUnderstanding):
        """Update linguistic patterns based on understanding"""
        # Extract patterns from linguistic features
        for feature in understanding.linguistic_features:
            pattern_key = f"{feature.feature_type}_{feature.feature_id.split('_')[0]}"
            self.linguistic_patterns[pattern_key].append({
                'value': feature.value,
                'confidence': feature.confidence,
                'timestamp': datetime.now()
            })
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze text complexity in detail
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict: Complexity analysis results
        """
        understanding = self.understand_text(text)
        
        analysis = {
            'overall_complexity': understanding.complexity_level.value,
            'complexity_score': self._calculate_complexity_score(understanding),
            'syntactic_complexity': self._analyze_syntactic_complexity(understanding),
            'semantic_complexity': self._analyze_semantic_complexity(understanding),
            'lexical_complexity': self._analyze_lexical_complexity(understanding),
            'pragmatic_complexity': self._analyze_pragmatic_complexity(understanding),
            'readability_metrics': self._calculate_readability_metrics(text),
            'comprehension_difficulty': self._estimate_comprehension_difficulty(understanding)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, understanding: LanguageUnderstanding) -> float:
        """Calculate numerical complexity score"""
        level_scores = {
            LinguisticComplexity.BASIC: 0.25,
            LinguisticComplexity.INTERMEDIATE: 0.5,
            LinguisticComplexity.ADVANCED: 0.75,
            LinguisticComplexity.EXPERT: 1.0
        }
        
        base_score = level_scores[understanding.complexity_level]
        
        # Adjust based on confidence
        adjusted_score = base_score * understanding.confidence
        
        return adjusted_score
    
    def _analyze_syntactic_complexity(self, understanding: LanguageUnderstanding) -> Dict[str, Any]:
        """Analyze syntactic complexity"""
        syntactic_features = [f for f in understanding.linguistic_features if f.feature_type == "syntactic"]
        
        analysis = {
            'parse_depth': 0,
            'dependency_relations': 0,
            'syntactic_variety': 0
        }
        
        for feature in syntactic_features:
            if "parse_depth" in feature.feature_id:
                analysis['parse_depth'] = feature.value
            elif "dep_" in feature.feature_id:
                analysis['dependency_relations'] += 1
        
        analysis['syntactic_variety'] = analysis['dependency_relations']
        
        return analysis
    
    def _analyze_semantic_complexity(self, understanding: LanguageUnderstanding) -> Dict[str, Any]:
        """Analyze semantic complexity"""
        semantic_features = [f for f in understanding.linguistic_features if f.feature_type == "semantic"]
        
        analysis = {
            'named_entities': len(semantic_features),
            'semantic_frames': len(understanding.semantic_frames),
            'conceptual_density': 0
        }
        
        if understanding.text:
            analysis['conceptual_density'] = analysis['named_entities'] / len(understanding.text.split())
        
        return analysis
    
    def _analyze_lexical_complexity(self, understanding: LanguageUnderstanding) -> Dict[str, Any]:
        """Analyze lexical complexity"""
        lexical_features = [f for f in understanding.linguistic_features if f.feature_type == "lexical"]
        
        analysis = {
            'vocabulary_richness': 0,
            'readability_score': 0,
            'word_diversity': 0
        }
        
        for feature in lexical_features:
            if "vocab_richness" in feature.feature_id:
                analysis['vocabulary_richness'] = feature.value
            elif "readability" in feature.feature_id:
                analysis['readability_score'] = feature.value
        
        analysis['word_diversity'] = analysis['vocabulary_richness']
        
        return analysis
    
    def _analyze_pragmatic_complexity(self, understanding: LanguageUnderstanding) -> Dict[str, Any]:
        """Analyze pragmatic complexity"""
        analysis = {
            'pragmatic_aspects': 0,
            'figurative_language': 0,
            'contextual_dependencies': 0
        }
        
        if understanding.pragmatic_interpretation:
            analysis['pragmatic_aspects'] = 1
            if understanding.pragmatic_interpretation.pragmatic_aspect != PragmaticAspect.LITERAL:
                analysis['figurative_language'] = 1
        
        analysis['contextual_dependencies'] = len(understanding.metadata.get('context', {}))
        
        return analysis
    
    def _calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences or not words:
            return {'flesch_score': 0, 'flesch_grade': 0}
        
        # Flesch Reading Ease
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length
        flesch_grade = 0.39 * avg_sentence_length + 11.8 * avg_word_length - 15.59
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),
            'flesch_grade': max(0, flesch_grade)
        }
    
    def _estimate_comprehension_difficulty(self, understanding: LanguageUnderstanding) -> str:
        """Estimate comprehension difficulty level"""
        complexity_score = self._calculate_complexity_score(understanding)
        
        if complexity_score < 0.3:
            return "very_easy"
        elif complexity_score < 0.5:
            return "easy"
        elif complexity_score < 0.7:
            return "moderate"
        elif complexity_score < 0.9:
            return "difficult"
        else:
            return "very_difficult"
    
    def extract_key_information(self, text: str, info_types: List[str]) -> Dict[str, Any]:
        """
        Extract key information from text
        
        Args:
            text: Text to analyze
            info_types: Types of information to extract
            
        Returns:
            Dict: Extracted information by type
        """
        understanding = self.understand_text(text)
        
        extracted_info = {}
        
        for info_type in info_types:
            if info_type == "entities":
                extracted_info[info_type] = self._extract_entities(understanding)
            elif info_type == "actions":
                extracted_info[info_type] = self._extract_actions(understanding)
            elif info_type == "sentiment":
                extracted_info[info_type] = self._extract_sentiment(understanding)
            elif info_type == "topics":
                extracted_info[info_type] = self._extract_topics(understanding)
            elif info_type == "relationships":
                extracted_info[info_type] = self._extract_relationships(understanding)
            elif info_type == "temporal":
                extracted_info[info_type] = self._extract_temporal_info(understanding)
        
        return extracted_info
    
    def _extract_entities(self, understanding: LanguageUnderstanding) -> List[Dict[str, Any]]:
        """Extract named entities"""
        entities = []
        
        for feature in understanding.linguistic_features:
            if feature.feature_type == "semantic" and "entity_" in feature.feature_id:
                entity_data = feature.value
                entities.append({
                    'text': entity_data['text'],
                    'type': entity_data['label'],
                    'confidence': feature.confidence,
                    'span': feature.span
                })
        
        return entities
    
    def _extract_actions(self, understanding: LanguageUnderstanding) -> List[Dict[str, Any]]:
        """Extract actions from semantic frames"""
        actions = []
        
        for frame in understanding.semantic_frames:
            action = {
                'verb': frame.frame_name,
                'agent': frame.frame_elements.get(SemanticRole.AGENT.value),
                'patient': frame.frame_elements.get(SemanticRole.PATIENT.value),
                'confidence': frame.confidence
            }
            actions.append(action)
        
        return actions
    
    def _extract_sentiment(self, understanding: LanguageUnderstanding) -> Dict[str, Any]:
        """Extract sentiment information"""
        sentiment_features = [f for f in understanding.linguistic_features if f.feature_type == "stylistic" and "emotion_" in f.feature_id]
        
        sentiment = {
            'positive': 0,
            'negative': 0,
            'neutral': 1,
            'overall': 'neutral'
        }
        
        for feature in sentiment_features:
            if "emotion_positive" in feature.feature_id:
                sentiment['positive'] = feature.value
            elif "emotion_negative" in feature.feature_id:
                sentiment['negative'] = feature.value
        
        # Determine overall sentiment
        if sentiment['positive'] > sentiment['negative']:
            sentiment['overall'] = 'positive'
        elif sentiment['negative'] > sentiment['positive']:
            sentiment['overall'] = 'negative'
        
        return sentiment
    
    def _extract_topics(self, understanding: LanguageUnderstanding) -> List[str]:
        """Extract topics from text"""
        topics = []
        
        if understanding.discourse_structure:
            topics = understanding.discourse_structure.topic_flow
        
        # Extract from semantic concepts
        for feature in understanding.linguistic_features:
            if feature.feature_type == "semantic" and "concept_" in feature.feature_id:
                topics.append(feature.value)
        
        return list(set(topics))  # Remove duplicates
    
    def _extract_relationships(self, understanding: LanguageUnderstanding) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Extract from semantic network
        text_lower = understanding.text.lower()
        for source, target, data in self.semantic_network.edges(data=True):
            if source in text_lower and target in text_lower:
                relationships.append({
                    'source': source,
                    'target': target,
                    'relation': data.get('relation', 'related'),
                    'confidence': 0.7
                })
        
        return relationships
    
    def _extract_temporal_info(self, understanding: LanguageUnderstanding) -> Dict[str, Any]:
        """Extract temporal information"""
        temporal_info = {
            'time_expressions': [],
            'duration': None,
            'sequence': []
        }
        
        # Extract time expressions from text
        time_patterns = [
            r'\b\d{1,2}:\d{2}\b',  # Time like 3:30
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Date like 12/25/2023
            r'\b(yesterday|today|tomorrow|now|soon|later)\b',
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, understanding.text.lower())
            temporal_info['time_expressions'].extend(matches)
        
        # Extract sequence from discourse structure
        if understanding.discourse_structure:
            temporal_info['sequence'] = understanding.discourse_structure.topic_flow
        
        return temporal_info
    
    def get_nlu_statistics(self) -> Dict[str, Any]:
        """Get comprehensive NLU statistics"""
        return {
            'total_understandings': len(self.understanding_history),
            'complexity_distribution': {
                level.value: len([u for u in self.understanding_history if u.complexity_level == level])
                for level in LinguisticComplexity
            },
            'average_confidence': np.mean([u.confidence for u in self.understanding_history]) if self.understanding_history else 0,
            'average_processing_time': np.mean([u.processing_time.total_seconds() for u in self.understanding_history]) if self.understanding_history else 0,
            'linguistic_patterns_count': len(self.linguistic_patterns),
            'semantic_network_size': len(self.semantic_network.nodes()),
            'pragmatic_rules_count': len(self.pragmatic_rules),
            'most_common_complexity': self._get_most_common_complexity(),
            'recent_performance': self._get_recent_performance()
        }
    
    def _get_most_common_complexity(self) -> str:
        """Get most common complexity level in recent understandings"""
        if not self.understanding_history:
            return "none"
        
        recent_understandings = list(self.understanding_history)[-50:]
        complexity_counts = defaultdict(int)
        
        for understanding in recent_understandings:
            complexity_counts[understanding.complexity_level.value] += 1
        
        return max(complexity_counts.items(), key=lambda x: x[1])[0]
    
    def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics"""
        if not self.understanding_history:
            return {}
        
        recent_understandings = list(self.understanding_history)[-20:]
        
        return {
            'average_confidence': np.mean([u.confidence for u in recent_understandings]),
            'average_processing_time': np.mean([u.processing_time.total_seconds() for u in recent_understandings]),
            'high_confidence_rate': len([u for u in recent_understandings if u.confidence > 0.8]) / len(recent_understandings)
        }
    
    def save_nlu_data(self, filepath: str):
        """Save NLU data to file"""
        try:
            data = {
                'linguistic_patterns': dict(self.linguistic_patterns),
                'pragmatic_rules': self.pragmatic_rules,
                'semantic_network_data': {
                    'nodes': list(self.semantic_network.nodes(data=True)),
                    'edges': list(self.semantic_network.edges(data=True))
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"NLU data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving NLU data: {e}")
    
    def load_nlu_data(self, filepath: str):
        """Load NLU data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load linguistic patterns
            self.linguistic_patterns = defaultdict(list, data['linguistic_patterns'])
            
            # Load pragmatic rules
            self.pragmatic_rules = data['pragmatic_rules']
            
            # Load semantic network
            self.semantic_network = nx.DiGraph()
            for node_data in data['semantic_network_data']['nodes']:
                self.semantic_network.add_node(node_data[0], **node_data[1])
            
            for edge_data in data['semantic_network_data']['edges']:
                self.semantic_network.add_edge(edge_data[0], edge_data[1], **edge_data[2])
            
            logger.info(f"NLU data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading NLU data: {e}")

# Example usage and demonstration
def demonstrate_advanced_nlu():
    """Demonstrate advanced NLU capabilities"""
    print("=== Advanced NLU Engine Demonstration ===")
    
    # Initialize NLU engine
    nlu_engine = AdvancedNLUEngine()
    
    # Test texts with different complexity levels
    test_texts = [
        "The cat sat on the mat.",  # Simple
        "Although it was raining, John decided to go for a walk because he needed some exercise.",  # Intermediate
        "The juxtaposition of existential nihilism and pragmatic optimism creates a philosophical tension that permeates contemporary discourse.",  # Advanced
        "Yeah right, this is just perfect!",  # Sarcastic
        "Her voice was music to his ears.",  # Metaphorical
        "The quantum entanglement of particles across vast distances challenges our classical understanding of spatial separation and locality."  # Expert
    ]
    
    print("\nProcessing texts with different complexity levels...")
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: {text}")
        print("-" * 60)
        
        # Understand text
        understanding = nlu_engine.understand_text(text)
        
        print(f"Complexity Level: {understanding.complexity_level.value}")
        print(f"Confidence: {understanding.confidence:.3f}")
        print(f"Processing Time: {understanding.processing_time.total_seconds():.3f}s")
        print(f"Linguistic Features: {len(understanding.linguistic_features)}")
        print(f"Semantic Frames: {len(understanding.semantic_frames)}")
        
        if understanding.pragmatic_interpretation:
            print(f"Pragmatic Aspect: {understanding.pragmatic_interpretation.pragmatic_aspect.value}")
            print(f"Intended Meaning: {understanding.pragmatic_interpretation.intended_meaning}")
        
        if understanding.discourse_structure:
            print(f"Coherence Score: {understanding.discourse_structure.coherence_score:.3f}")
    
    # Test complexity analysis
    print("\n=== Complexity Analysis ===")
    complex_text = "The intricate interplay between socioeconomic factors and technological advancement necessitates a comprehensive reevaluation of our established paradigms."
    
    complexity_analysis = nlu_engine.analyze_text_complexity(complex_text)
    
    print(f"Text: {complex_text}")
    print(f"Overall Complexity: {complexity_analysis['overall_complexity']}")
    print(f"Complexity Score: {complexity_analysis['complexity_score']:.3f}")
    print(f"Syntactic Complexity: {complexity_analysis['syntactic_complexity']}")
    print(f"Semantic Complexity: {complexity_analysis['semantic_complexity']}")
    print(f"Lexical Complexity: {complexity_analysis['lexical_complexity']}")
    print(f"Comprehension Difficulty: {complexity_analysis['comprehension_difficulty']}")
    
    # Test information extraction
    print("\n=== Information Extraction ===")
    info_text = "Yesterday, John visited Microsoft in Seattle to discuss the new AI project with Sarah. The meeting lasted for 3 hours."
    
    info_types = ["entities", "actions", "sentiment", "topics", "relationships", "temporal"]
    extracted_info = nlu_engine.extract_key_information(info_text, info_types)
    
    print(f"Text: {info_text}")
    for info_type, info_data in extracted_info.items():
        print(f"{info_type.title()}: {info_data}")
    
    # Show statistics
    print("\n=== NLU Statistics ===")
    stats = nlu_engine.get_nlu_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}: {dict(value)}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Save NLU data
    nlu_engine.save_nlu_data('nlu_data.json')
    print("\nNLU data saved successfully!")

if __name__ == "__main__":
    demonstrate_advanced_nlu()