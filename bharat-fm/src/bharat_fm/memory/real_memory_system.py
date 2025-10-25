"""
Real Memory Management System for Bharat-FM
Actual implementation with proper context management and conversation memory
"""

import asyncio
import time
import json
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    id: str
    content: str
    role: str  # 'user', 'assistant', 'system'
    timestamp: datetime
    embedding: Optional[np.ndarray] = None
    importance_score: float = 0.0
    tags: List[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ConversationSession:
    """Conversation session with multiple memory entries"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    memory_entries: List[str] = None  # List of memory entry IDs
    context_summary: str = ""
    title: Optional[str] = None
    
    def __post_init__(self):
        if self.memory_entries is None:
            self.memory_entries = []

class RealMemorySystem:
    """Real memory system with proper context management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Memory storage
        self.memory_entries: Dict[str, MemoryEntry] = {}
        self.conversation_sessions: Dict[str, ConversationSession] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_memory_entries = self.config.get("max_memory_entries", 10000)
        self.max_context_length = self.config.get("max_context_length", 4096)
        self.memory_retention_days = self.config.get("memory_retention_days", 30)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        
        # Text processing for similarity
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Cache for frequently accessed memories
        self.recent_memories = deque(maxlen=100)
        
        # Background tasks
        self.cleanup_task = None
        self.is_running = False
        
        # Storage paths
        self.storage_dir = self.config.get("storage_dir", "./memory_storage")
        self._ensure_storage_dir()
        
        # Load existing data
        self._load_memory_data()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def _load_memory_data(self):
        """Load existing memory data from disk"""
        try:
            # Load memory entries
            entries_file = os.path.join(self.storage_dir, "memory_entries.pkl")
            if os.path.exists(entries_file):
                with open(entries_file, 'rb') as f:
                    self.memory_entries = pickle.load(f)
                logger.info(f"Loaded {len(self.memory_entries)} memory entries")
            
            # Load conversation sessions
            sessions_file = os.path.join(self.storage_dir, "conversation_sessions.pkl")
            if os.path.exists(sessions_file):
                with open(sessions_file, 'rb') as f:
                    self.conversation_sessions = pickle.load(f)
                logger.info(f"Loaded {len(self.conversation_sessions)} conversation sessions")
            
            # Load user profiles
            profiles_file = os.path.join(self.storage_dir, "user_profiles.pkl")
            if os.path.exists(profiles_file):
                with open(profiles_file, 'rb') as f:
                    self.user_profiles = pickle.load(f)
                logger.info(f"Loaded {len(self.user_profiles)} user profiles")
                
        except Exception as e:
            logger.error(f"Error loading memory data: {e}")
    
    def _save_memory_data(self):
        """Save memory data to disk"""
        try:
            # Save memory entries
            entries_file = os.path.join(self.storage_dir, "memory_entries.pkl")
            with open(entries_file, 'wb') as f:
                pickle.dump(self.memory_entries, f)
            
            # Save conversation sessions
            sessions_file = os.path.join(self.storage_dir, "conversation_sessions.pkl")
            with open(sessions_file, 'wb') as f:
                pickle.dump(self.conversation_sessions, f)
            
            # Save user profiles
            profiles_file = os.path.join(self.storage_dir, "user_profiles.pkl")
            with open(profiles_file, 'wb') as f:
                pickle.dump(self.user_profiles, f)
                
        except Exception as e:
            logger.error(f"Error saving memory data: {e}")
    
    async def start(self):
        """Start the memory system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_old_memories())
        
        logger.info("Real memory system started")
    
    async def stop(self):
        """Stop the memory system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save data before stopping
        self._save_memory_data()
        
        logger.info("Real memory system stopped")
    
    async def add_memory(self, 
                        content: str, 
                        role: str, 
                        session_id: str,
                        user_id: str,
                        tags: Optional[List[str]] = None,
                        importance_score: float = 0.0) -> str:
        """Add a new memory entry"""
        memory_id = f"mem_{int(time.time() * 1000000)}_{len(self.memory_entries)}"
        
        # Create memory entry
        memory_entry = MemoryEntry(
            id=memory_id,
            content=content,
            role=role,
            timestamp=datetime.utcnow(),
            importance_score=importance_score,
            tags=tags or [],
            session_id=session_id,
            user_id=user_id
        )
        
        # Generate embedding for similarity search
        try:
            memory_entry.embedding = self._generate_embedding(content)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for memory {memory_id}: {e}")
        
        # Store memory
        self.memory_entries[memory_id] = memory_entry
        self.recent_memories.append(memory_id)
        
        # Update conversation session
        if session_id not in self.conversation_sessions:
            self.conversation_sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.utcnow()
            )
        
        self.conversation_sessions[session_id].memory_entries.append(memory_id)
        
        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "total_memories": 0,
                "total_sessions": 0,
                "preferences": {},
                "interaction_history": []
            }
        
        self.user_profiles[user_id]["total_memories"] += 1
        self.user_profiles[user_id]["interaction_history"].append({
            "memory_id": memory_id,
            "timestamp": datetime.utcnow().isoformat(),
            "role": role
        })
        
        # Auto-save after significant changes
        if len(self.memory_entries) % 100 == 0:
            self._save_memory_data()
        
        logger.info(f"Added memory entry: {memory_id}")
        return memory_id
    
    async def get_context(self, 
                         session_id: str, 
                         max_tokens: int = 2048,
                         include_system: bool = True) -> List[Dict[str, Any]]:
        """Get conversation context with proper token management"""
        if session_id not in self.conversation_sessions:
            return []
        
        session = self.conversation_sessions[session_id]
        context = []
        
        # Get memory entries in chronological order
        memory_entries = []
        for memory_id in session.memory_entries:
            if memory_id in self.memory_entries:
                memory_entries.append(self.memory_entries[memory_id])
        
        # Sort by timestamp
        memory_entries.sort(key=lambda x: x.timestamp)
        
        # Build context with token limit
        current_tokens = 0
        for entry in memory_entries:
            if entry.role == 'system' and not include_system:
                continue
            
            # Estimate tokens (rough approximation)
            entry_tokens = len(entry.content.split())
            
            if current_tokens + entry_tokens > max_tokens:
                break
            
            context.append({
                "role": entry.role,
                "content": entry.content,
                "timestamp": entry.timestamp.isoformat(),
                "memory_id": entry.id
            })
            
            current_tokens += entry_tokens
        
        return context
    
    async def search_memories(self, 
                            query: str, 
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Filter memories based on criteria
            candidate_memories = []
            for memory_id, entry in self.memory_entries.items():
                if user_id and entry.user_id != user_id:
                    continue
                if session_id and entry.session_id != session_id:
                    continue
                
                candidate_memories.append((memory_id, entry))
            
            # Calculate similarities
            similarities = []
            for memory_id, entry in candidate_memories:
                if entry.embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        entry.embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((memory_id, similarity, entry))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for memory_id, similarity, entry in similarities[:limit]:
                if similarity >= self.similarity_threshold:
                    results.append({
                        "memory_id": memory_id,
                        "content": entry.content,
                        "role": entry.role,
                        "similarity": float(similarity),
                        "timestamp": entry.timestamp.isoformat(),
                        "tags": entry.tags,
                        "importance_score": entry.importance_score
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile with memory statistics"""
        if user_id not in self.user_profiles:
            return {"error": "User not found"}
        
        profile = self.user_profiles[user_id].copy()
        
        # Add additional statistics
        user_memories = [entry for entry in self.memory_entries.values() 
                        if entry.user_id == user_id]
        
        profile["total_memory_entries"] = len(user_memories)
        profile["total_sessions"] = len([s for s in self.conversation_sessions.values() 
                                       if s.user_id == user_id])
        
        # Calculate interaction patterns
        if user_memories:
            profile["average_memories_per_session"] = len(user_memories) / max(profile["total_sessions"], 1)
            profile["most_common_tags"] = self._get_most_common_tags(user_memories)
        
        return profile
    
    async def update_memory_importance(self, memory_id: str, importance_score: float):
        """Update importance score of a memory entry"""
        if memory_id not in self.memory_entries:
            return False
        
        self.memory_entries[memory_id].importance_score = importance_score
        logger.info(f"Updated importance score for memory {memory_id}: {importance_score}")
        return True
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        if memory_id not in self.memory_entries:
            return False
        
        entry = self.memory_entries[memory_id]
        
        # Remove from conversation session
        if entry.session_id in self.conversation_sessions:
            session = self.conversation_sessions[entry.session_id]
            if memory_id in session.memory_entries:
                session.memory_entries.remove(memory_id)
        
        # Remove from memory entries
        del self.memory_entries[memory_id]
        
        # Remove from recent memories
        if memory_id in self.recent_memories:
            self.recent_memories.remove(memory_id)
        
        logger.info(f"Deleted memory entry: {memory_id}")
        return True
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_memories = len(self.memory_entries)
        total_sessions = len(self.conversation_sessions)
        total_users = len(self.user_profiles)
        
        # Calculate age distribution
        now = datetime.utcnow()
        age_distribution = {"1_day": 0, "7_days": 0, "30_days": 0, "older": 0}
        
        for entry in self.memory_entries.values():
            age = (now - entry.timestamp).days
            if age <= 1:
                age_distribution["1_day"] += 1
            elif age <= 7:
                age_distribution["7_days"] += 1
            elif age <= 30:
                age_distribution["30_days"] += 1
            else:
                age_distribution["older"] += 1
        
        return {
            "total_memory_entries": total_memories,
            "total_conversation_sessions": total_sessions,
            "total_users": total_users,
            "age_distribution": age_distribution,
            "average_memories_per_session": total_memories / max(total_sessions, 1),
            "system_running": self.is_running
        }
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using TF-IDF"""
        try:
            # Fit vectorizer if not already fitted
            if not hasattr(self.vectorizer, 'vocabulary_'):
                # Use existing memories to fit vectorizer
                existing_texts = [entry.content for entry in self.memory_entries.values()]
                if existing_texts:
                    self.vectorizer.fit(existing_texts + [text])
                else:
                    self.vectorizer.fit([text])
            
            # Transform text to embedding
            embedding = self.vectorizer.transform([text]).toarray()
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(1000)
    
    def _get_most_common_tags(self, memories: List[MemoryEntry], limit: int = 10) -> List[str]:
        """Get most common tags from memories"""
        tag_counts = defaultdict(int)
        for memory in memories:
            for tag in memory.tags:
                tag_counts[tag] += 1
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_tags[:limit]]
    
    async def _cleanup_old_memories(self):
        """Background task to clean up old memories"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.utcnow() - timedelta(days=self.memory_retention_days)
                
                # Find old memories
                old_memories = []
                for memory_id, entry in self.memory_entries.items():
                    if entry.timestamp < cutoff_time:
                        old_memories.append(memory_id)
                
                # Delete old memories (except high importance ones)
                deleted_count = 0
                for memory_id in old_memories:
                    entry = self.memory_entries[memory_id]
                    if entry.importance_score < 0.8:  # Keep high importance memories
                        await self.delete_memory(memory_id)
                        deleted_count += 1
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old memories")
                    self._save_memory_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup: {e}")


# Convenience function for easy usage
async def create_real_memory_system(config: Optional[Dict[str, Any]] = None) -> RealMemorySystem:
    """Create and start real memory system"""
    memory_system = RealMemorySystem(config)
    await memory_system.start()
    return memory_system