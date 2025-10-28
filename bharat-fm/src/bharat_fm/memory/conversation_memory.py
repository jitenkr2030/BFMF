"""
Conversation Memory & Context Management for Bharat-FM
Implements intelligent conversation memory with semantic search and personalization
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging
from collections import defaultdict, deque
import hashlib
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class ConversationExchange:
    """Single conversation exchange"""
    timestamp: datetime
    user_message: str
    assistant_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    sentiment_score: float = 0.0
    topics: List[str] = field(default_factory=list)
    importance_score: float = 0.0

@dataclass
class PersonalizationProfile:
    """User personalization profile"""
    user_id: str
    language_preferences: List[str] = field(default_factory=list)
    topic_interests: Dict[str, float] = field(default_factory=dict)
    communication_style: str = "neutral"  # formal, casual, technical
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    cultural_context: str = "general"
    interaction_history: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EmotionalState:
    """Emotional state tracking"""
    user_id: str
    current_emotion: str = "neutral"
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    sentiment_trend: List[float] = field(default_factory=list)
    engagement_level: float = 0.5  # 0-1 scale
    frustration_level: float = 0.0  # 0-1 scale
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConversationContext:
    """Complete conversation context"""
    user_id: str
    session_id: str
    personalization: PersonalizationProfile
    emotional_state: EmotionalState
    history: deque[ConversationExchange] = field(default_factory=lambda: deque(maxlen=1000))
    key_topics: List[str] = field(default_factory=list)
    session_start_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    context_summary: str = ""
    relevant_facts: Dict[str, Any] = field(default_factory=dict)

class ConversationMemoryManager:
    """Advanced conversation memory management system"""
    
    def __init__(self, 
                 memory_dir: str = "./conversation_memory",
                 max_history_length: int = 1000,
                 max_sessions_per_user: int = 10,
                 context_retention_days: int = 30,
                 config: Optional[Dict[str, Any]] = None):
        
        # Handle both old-style parameters and new config dict
        if isinstance(memory_dir, dict):
            # Called with config dict as first parameter
            config = memory_dir
            memory_dir = config.get("memory_dir", "./conversation_memory")
            max_history_length = config.get("max_history_length", 1000)
            max_sessions_per_user = config.get("max_sessions_per_user", 10)
            context_retention_days = config.get("context_retention_days", 30)
        elif config:
            # Called with both parameters and config
            memory_dir = config.get("memory_dir", memory_dir)
            max_history_length = config.get("max_history_length", max_history_length)
            max_sessions_per_user = config.get("max_sessions_per_user", max_sessions_per_user)
            context_retention_days = config.get("context_retention_days", context_retention_days)
        
        self.memory_dir = memory_dir
        self.max_history_length = max_history_length
        self.max_sessions_per_user = max_sessions_per_user
        self.context_retention_days = context_retention_days
        
        # Active conversations
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # Persistent storage
        self.user_profiles: Dict[str, PersonalizationProfile] = {}
        self.conversation_archives: Dict[str, List[ConversationExchange]] = defaultdict(list)
        
        # Semantic search
        self.embedding_model = None
        self.conversation_embeddings: Dict[str, List[Tuple[np.ndarray, str]]] = defaultdict(list)
        
        # Topic extraction
        self.topic_extractor = None
        self.topic_keywords: Dict[str, List[str]] = {}
        
        # Memory management
        self.memory_cleanup_task = None
        self.running = False
        
        # Create memory directory
        os.makedirs(memory_dir, exist_ok=True)
        
        # Load existing data
        self._load_persistent_data()
    
    async def start(self):
        """Start conversation memory manager"""
        if self.running:
            return
        
        self.running = True
        self.memory_cleanup_task = asyncio.create_task(self._cleanup_old_memories())
        
        logger.info("Conversation memory manager started")
    
    async def stop(self):
        """Stop conversation memory manager"""
        self.running = False
        
        if self.memory_cleanup_task:
            self.memory_cleanup_task.cancel()
            try:
                await self.memory_cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save data before stopping
        self._save_persistent_data()
        
        logger.info("Conversation memory manager stopped")
    
    async def store_exchange(self, user_id: str, session_id: str, 
                           user_message: str, assistant_response: str,
                           metadata: Optional[Dict[str, Any]] = None):
        """Store conversation exchange with semantic indexing"""
        
        # Get or create conversation context
        context = await self._get_or_create_context(user_id, session_id)
        
        # Create conversation exchange
        exchange = ConversationExchange(
            timestamp=datetime.utcnow(),
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )
        
        # Generate embedding for semantic search
        exchange.embedding = await self._generate_conversation_embedding(
            user_message, assistant_response
        )
        
        # Extract topics
        exchange.topics = await self._extract_topics(user_message, assistant_response)
        
        # Analyze sentiment
        exchange.sentiment_score = await self._analyze_sentiment(user_message)
        
        # Calculate importance
        exchange.importance_score = await self._calculate_importance(exchange)
        
        # Add to conversation history
        context.history.append(exchange)
        
        # Update key topics
        context.key_topics.extend(exchange.topics)
        context.key_topics = list(set(context.key_topics))[-20:]  # Keep last 20 unique topics
        
        # Update personalization profile
        await self._update_personalization_profile(context, exchange)
        
        # Update emotional state
        await self._update_emotional_state(context, exchange)
        
        # Update context summary
        await self._update_context_summary(context)
        
        # Store in semantic index
        await self._store_in_semantic_index(user_id, exchange)
        
        # Update last activity
        context.last_activity = datetime.utcnow()
        
        logger.debug(f"Stored exchange for user {user_id}, session {session_id}")
    
    async def retrieve_relevant_context(self, user_id: str, current_message: str,
                                      limit: int = 5) -> Dict[str, Any]:
        """Retrieve relevant conversation context"""
        
        # Get current context
        context = self.active_conversations.get(f"{user_id}_current")
        
        # Semantic search for relevant exchanges
        relevant_exchanges = await self._semantic_search(user_id, current_message, limit)
        
        # Get recent history
        recent_history = []
        if context:
            recent_history = list(context.history)[-10:]  # Last 10 exchanges
        
        # Get personalization profile
        personalization = self.user_profiles.get(user_id)
        if not personalization:
            personalization = await self._create_default_profile(user_id)
            self.user_profiles[user_id] = personalization
        
        # Get emotional state
        emotional_state = None
        if context:
            emotional_state = context.emotional_state
        
        # Get key topics
        key_topics = []
        if context:
            key_topics = context.key_topics[-10:]  # Last 10 topics
        
        # Generate context summary
        context_summary = ""
        if context:
            context_summary = context.context_summary
        
        return {
            "recent_history": [
                {
                    "timestamp": exchange.timestamp.isoformat(),
                    "user_message": exchange.user_message,
                    "assistant_response": exchange.assistant_response,
                    "sentiment_score": exchange.sentiment_score,
                    "topics": exchange.topics,
                    "importance_score": exchange.importance_score
                }
                for exchange in recent_history
            ],
            "relevant_exchanges": [
                {
                    "timestamp": exchange.timestamp.isoformat(),
                    "user_message": exchange.user_message,
                    "assistant_response": exchange.assistant_response,
                    "relevance_score": score,
                    "topics": exchange.topics
                }
                for exchange, score in relevant_exchanges
            ],
            "personalization": {
                "language_preferences": personalization.language_preferences,
                "topic_interests": personalization.topic_interests,
                "communication_style": personalization.communication_style,
                "expertise_level": personalization.expertise_level,
                "cultural_context": personalization.cultural_context
            },
            "emotional_state": {
                "current_emotion": emotional_state.current_emotion if emotional_state else "neutral",
                "emotion_scores": emotional_state.emotion_scores if emotional_state else {},
                "engagement_level": emotional_state.engagement_level if emotional_state else 0.5,
                "frustration_level": emotional_state.frustration_level if emotional_state else 0.0
            },
            "key_topics": key_topics,
            "context_summary": context_summary,
            "session_info": {
                "session_id": context.session_id if context else None,
                "session_start": context.session_start_time.isoformat() if context else None,
                "last_activity": context.last_activity.isoformat() if context else None,
                "exchange_count": len(context.history) if context else 0
            }
        }
    
    async def _get_or_create_context(self, user_id: str, session_id: str) -> ConversationContext:
        """Get or create conversation context"""
        context_key = f"{user_id}_{session_id}"
        
        if context_key not in self.active_conversations:
            # Load user profile
            personalization = self.user_profiles.get(user_id)
            if not personalization:
                personalization = await self._create_default_profile(user_id)
                self.user_profiles[user_id] = personalization
            
            # Create emotional state
            emotional_state = EmotionalState(user_id=user_id)
            
            # Create new context
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                personalization=personalization,
                emotional_state=emotional_state
            )
            
            self.active_conversations[context_key] = context
            
            # Load conversation history if available
            await self._load_conversation_history(user_id, context)
        
        return self.active_conversations[context_key]
    
    async def _create_default_profile(self, user_id: str) -> PersonalizationProfile:
        """Create default personalization profile"""
        return PersonalizationProfile(
            user_id=user_id,
            language_preferences=["en"],  # Default to English
            communication_style="neutral",
            expertise_level="intermediate",
            cultural_context="general"
        )
    
    async def _generate_conversation_embedding(self, user_message: str, assistant_response: str) -> np.ndarray:
        """Generate embedding for conversation exchange"""
        # Combine messages
        combined_text = f"{user_message} {assistant_response}"
        
        # Simple hash-based embedding (placeholder)
        # In production, use proper sentence transformers
        text_hash = hashlib.md5(combined_text.encode()).hexdigest()
        
        # Convert to numerical embedding
        try:
            embedding = np.array([
                int(text_hash[i:i+8], 16) % 1000 / 1000.0
                for i in range(0, 64, 8)
            ])
        except (ValueError, IndexError):
            # Fallback to random embedding if hash fails
            embedding = np.random.random(8)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    async def _extract_topics(self, user_message: str, assistant_response: str) -> List[str]:
        """Extract topics from conversation exchange"""
        # Simple keyword-based topic extraction (placeholder)
        # In production, use proper topic modeling or NLP techniques
        
        combined_text = f"{user_message} {assistant_response}".lower()
        
        # Define topic keywords
        topic_keywords = {
            "technology": ["tech", "software", "computer", "ai", "ml", "programming"],
            "business": ["business", "company", "market", "finance", "economy"],
            "education": ["education", "learning", "school", "university", "study"],
            "health": ["health", "medical", "doctor", "hospital", "medicine"],
            "entertainment": ["movie", "music", "game", "book", "sport"],
            "politics": ["politics", "government", "election", "policy", "law"],
            "science": ["science", "research", "experiment", "discovery", "theory"],
            "travel": ["travel", "trip", "vacation", "hotel", "flight"]
        }
        
        extracted_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                extracted_topics.append(topic)
        
        return extracted_topics
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        # Simple sentiment analysis (placeholder)
        # In production, use proper sentiment analysis models
        
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "hate", "dislike"]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment_score))
    
    async def _calculate_importance(self, exchange: ConversationExchange) -> float:
        """Calculate importance score for conversation exchange"""
        importance = 0.0
        
        # Length importance
        text_length = len(exchange.user_message) + len(exchange.assistant_response)
        importance += min(text_length / 1000, 0.3)  # Max 0.3 for length
        
        # Sentiment importance (extreme sentiments are more important)
        importance += abs(exchange.sentiment_score) * 0.2
        
        # Topic importance (certain topics are more important)
        important_topics = ["technology", "business", "health", "politics"]
        if any(topic in exchange.topics for topic in important_topics):
            importance += 0.2
        
        # Recency importance (more recent exchanges are slightly more important)
        age_hours = (datetime.utcnow() - exchange.timestamp).total_seconds() / 3600
        recency_weight = max(0, 1 - age_hours / 24)  # Decay over 24 hours
        importance += recency_weight * 0.1
        
        # Question importance (questions are important)
        if "?" in exchange.user_message:
            importance += 0.2
        
        return min(1.0, importance)
    
    async def _update_personalization_profile(self, context: ConversationContext, exchange: ConversationExchange):
        """Update user personalization profile"""
        profile = context.personalization
        
        # Update language preferences based on detected languages
        # This is simplified - in production, use proper language detection
        if any(lang in exchange.user_message.lower() for lang in ["hindi", "हिंदी", "namaste"]):
            if "hi" not in profile.language_preferences:
                profile.language_preferences.append("hi")
        
        # Update topic interests
        for topic in exchange.topics:
            profile.topic_interests[topic] = profile.topic_interests.get(topic, 0) + 1
        
        # Update communication style based on message patterns
        if exchange.user_message.isupper():
            profile.communication_style = "enthusiastic"
        elif any(formal_word in exchange.user_message.lower() for formal_word in ["please", "thank you", "sir", "madam"]):
            profile.communication_style = "formal"
        elif any(casual_word in exchange.user_message.lower() for casual_word in ["hey", "yo", "dude", "man"]):
            profile.communication_style = "casual"
        
        # Update expertise level based on vocabulary complexity
        avg_word_length = len(exchange.user_message.split()) / max(1, len(exchange.user_message.split()))
        if avg_word_length > 6:
            profile.expertise_level = "expert"
        elif avg_word_length > 4:
            profile.expertise_level = "intermediate"
        else:
            profile.expertise_level = "beginner"
        
        profile.last_updated = datetime.utcnow()
    
    async def _update_emotional_state(self, context: ConversationContext, exchange: ConversationExchange):
        """Update emotional state tracking"""
        emotional_state = context.emotional_state
        
        # Update emotion scores
        emotions = {
            "joy": max(0, exchange.sentiment_score),
            "sadness": max(0, -exchange.sentiment_score),
            "anger": max(0, -exchange.sentiment_score) if "!" in exchange.user_message else 0,
            "surprise": 0.1 if "?" in exchange.user_message else 0,
            "disgust": max(0, -exchange.sentiment_score * 0.5)
        }
        
        emotional_state.emotion_scores = emotions
        emotional_state.current_emotion = max(emotions, key=emotions.get)
        
        # Update sentiment trend
        emotional_state.sentiment_trend.append(exchange.sentiment_score)
        if len(emotional_state.sentiment_trend) > 20:
            emotional_state.sentiment_trend = emotional_state.sentiment_trend[-20:]
        
        # Update engagement level
        message_length = len(exchange.user_message)
        if message_length > 100:
            emotional_state.engagement_level = min(1.0, emotional_state.engagement_level + 0.1)
        elif message_length < 10:
            emotional_state.engagement_level = max(0.0, emotional_state.engagement_level - 0.05)
        
        # Update frustration level
        if exchange.sentiment_score < -0.5:
            emotional_state.frustration_level = min(1.0, emotional_state.frustration_level + 0.2)
        elif exchange.sentiment_score > 0.5:
            emotional_state.frustration_level = max(0.0, emotional_state.frustration_level - 0.1)
        
        emotional_state.last_updated = datetime.utcnow()
    
    async def _update_context_summary(self, context: ConversationContext):
        """Update conversation context summary"""
        if len(context.history) < 3:
            return
        
        # Get recent exchanges
        recent_exchanges = list(context.history)[-5:]
        
        # Generate simple summary (placeholder)
        # In production, use proper text summarization
        topics_mentioned = []
        for exchange in recent_exchanges:
            topics_mentioned.extend(exchange.topics)
        
        unique_topics = list(set(topics_mentioned))
        
        context.context_summary = f"Conversation about {', '.join(unique_topics[:3])}. " \
                                 f"User seems {context.emotional_state.current_emotion}. " \
                                 f"Communication style is {context.personalization.communication_style}."
    
    async def _store_in_semantic_index(self, user_id: str, exchange: ConversationExchange):
        """Store conversation exchange in semantic index"""
        if exchange.embedding is not None:
            self.conversation_embeddings[user_id].append(
                (exchange.embedding, exchange.timestamp.isoformat())
            )
            
            # Keep only recent embeddings
            if len(self.conversation_embeddings[user_id]) > 1000:
                self.conversation_embeddings[user_id] = self.conversation_embeddings[user_id][-1000:]
    
    async def _semantic_search(self, user_id: str, query: str, limit: int = 5) -> List[Tuple[ConversationExchange, float]]:
        """Perform semantic search for relevant conversations"""
        # Generate query embedding
        query_embedding = await self._generate_conversation_embedding(query, "")
        
        # Get user's conversation embeddings
        user_embeddings = self.conversation_embeddings.get(user_id, [])
        
        if not user_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for embedding, timestamp in user_embeddings:
            similarity = np.dot(query_embedding, embedding)
            similarities.append((similarity, timestamp))
        
        # Sort by similarity and get top results
        similarities.sort(reverse=True)
        
        # Get corresponding exchanges (simplified - in production, you'd store the actual exchanges)
        relevant_exchanges = []
        for similarity, timestamp in similarities[:limit]:
            if similarity > 0.5:  # Threshold for relevance
                # Create a mock exchange for demonstration
                exchange = ConversationExchange(
                    timestamp=datetime.fromisoformat(timestamp),
                    user_message="Relevant conversation",
                    assistant_response="Relevant response",
                    sentiment_score=0.0,
                    importance_score=similarity
                )
                relevant_exchanges.append((exchange, similarity))
        
        return relevant_exchanges
    
    async def _load_conversation_history(self, user_id: str, context: ConversationContext):
        """Load conversation history for user"""
        # This is a placeholder - in production, load from persistent storage
        pass
    
    async def _cleanup_old_memories(self):
        """Background task to clean up old memories"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Clean up old active conversations
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                expired_contexts = []
                for context_key, context in self.active_conversations.items():
                    if context.last_activity < cutoff_time:
                        expired_contexts.append(context_key)
                
                for context_key in expired_contexts:
                    # Archive conversation before removing
                    context = self.active_conversations[context_key]
                    await self._archive_conversation(context)
                    del self.active_conversations[context_key]
                
                if expired_contexts:
                    logger.info(f"Cleaned up {len(expired_contexts)} expired conversations")
                
                # Save persistent data
                self._save_persistent_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup: {e}")
    
    async def _archive_conversation(self, context: ConversationContext):
        """Archive conversation to persistent storage"""
        user_id = context.user_id
        
        # Add to archives
        for exchange in context.history:
            self.conversation_archives[user_id].append(exchange)
        
        # Keep only recent archives
        if len(self.conversation_archives[user_id]) > self.max_history_length:
            self.conversation_archives[user_id] = self.conversation_archives[user_id][-self.max_history_length:]
    
    def _load_persistent_data(self):
        """Load persistent data from disk"""
        # Load user profiles
        profiles_file = os.path.join(self.memory_dir, "user_profiles.pkl")
        if os.path.exists(profiles_file):
            try:
                with open(profiles_file, 'rb') as f:
                    self.user_profiles = pickle.load(f)
            except:
                self.user_profiles = {}
        
        # Load conversation archives
        archives_file = os.path.join(self.memory_dir, "conversation_archives.pkl")
        if os.path.exists(archives_file):
            try:
                with open(archives_file, 'rb') as f:
                    self.conversation_archives = pickle.load(f)
            except:
                self.conversation_archives = defaultdict(list)
    
    def _save_persistent_data(self):
        """Save persistent data to disk"""
        # Save user profiles
        profiles_file = os.path.join(self.memory_dir, "user_profiles.pkl")
        try:
            with open(profiles_file, 'wb') as f:
                pickle.dump(self.user_profiles, f)
        except:
            pass
        
        # Save conversation archives
        archives_file = os.path.join(self.memory_dir, "conversation_archives.pkl")
        try:
            with open(archives_file, 'wb') as f:
                pickle.dump(dict(self.conversation_archives), f)
        except:
            pass
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory management statistics"""
        return {
            "active_conversations": len(self.active_conversations),
            "user_profiles": len(self.user_profiles),
            "total_archived_exchanges": sum(len(exchanges) for exchanges in self.conversation_archives.values()),
            "semantic_index_size": sum(len(embeddings) for embeddings in self.conversation_embeddings.values()),
            "memory_usage_mb": self._calculate_memory_usage(),
            "cleanup_status": "running" if self.running else "stopped"
        }
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage in MB"""
        # This is a simplified calculation
        total_size = 0
        
        # User profiles
        for profile in self.user_profiles.values():
            total_size += len(str(profile))
        
        # Conversation archives
        for exchanges in self.conversation_archives.values():
            for exchange in exchanges:
                total_size += len(str(exchange))
        
        # Semantic embeddings
        for embeddings in self.conversation_embeddings.values():
            for embedding, _ in embeddings:
                total_size += embedding.nbytes
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def end_session(self, user_id: str, session_id: str):
        """End conversation session and archive it"""
        context_key = f"{user_id}_{session_id}"
        
        if context_key in self.active_conversations:
            context = self.active_conversations[context_key]
            await self._archive_conversation(context)
            del self.active_conversations[context_key]
            
            logger.info(f"Ended session {session_id} for user {user_id}")
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user personalization profile"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return None
        
        return {
            "user_id": profile.user_id,
            "language_preferences": profile.language_preferences,
            "topic_interests": profile.topic_interests,
            "communication_style": profile.communication_style,
            "expertise_level": profile.expertise_level,
            "cultural_context": profile.cultural_context,
            "last_updated": profile.last_updated.isoformat()
        }
    
    async def update_user_profile(self, user_id: str, profile_updates: Dict[str, Any]):
        """Update user personalization profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = await self._create_default_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update profile fields
        if "language_preferences" in profile_updates:
            profile.language_preferences = profile_updates["language_preferences"]
        
        if "communication_style" in profile_updates:
            profile.communication_style = profile_updates["communication_style"]
        
        if "expertise_level" in profile_updates:
            profile.expertise_level = profile_updates["expertise_level"]
        
        if "cultural_context" in profile_updates:
            profile.cultural_context = profile_updates["cultural_context"]
        
        profile.last_updated = datetime.utcnow()
        
        logger.info(f"Updated profile for user {user_id}")