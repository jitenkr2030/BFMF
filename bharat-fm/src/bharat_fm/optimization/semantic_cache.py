"""
Semantic Cache for Bharat-FM Inference Optimization
Implements intelligent caching based on semantic similarity rather than exact matches
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

@dataclass
class CacheEntry:
    """Represents a cached inference result"""
    request_hash: str
    request_embedding: np.ndarray
    response: Any
    model_used: str
    timestamp: datetime
    access_count: int
    ttl: int  # Time to live in seconds
    cost: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl)
    
    def increment_access(self):
        """Increment access count and update timestamp"""
        self.access_count += 1
        self.timestamp = datetime.utcnow()

class SemanticCache:
    """Semantic cache for inference requests"""
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 max_entries: int = 10000,
                 default_ttl: int = 3600,
                 similarity_threshold: float = 0.95):
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, CacheEntry] = {}
        self.embedding_model = None
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
    
    async def get(self, request: Dict[str, Any]) -> Optional[CacheEntry]:
        """Get cached result for request"""
        # Generate request hash and embedding
        request_hash = self._generate_request_hash(request)
        request_embedding = await self._generate_request_embedding(request)
        
        # Check exact match first
        if request_hash in self.cache:
            entry = self.cache[request_hash]
            if not entry.is_expired():
                entry.increment_access()
                self._save_cache()
                return entry
            else:
                # Remove expired entry
                del self.cache[request_hash]
        
        # Check semantic similarity
        similar_entry = await self._find_similar_request(request_embedding)
        if similar_entry:
            similar_entry.increment_access()
            self._save_cache()
            return similar_entry
        
        return None
    
    async def store(self, request: Dict[str, Any], response: Any, model_used: str, cost: float = 0.0, ttl: int = None):
        """Store inference result in cache"""
        request_hash = self._generate_request_hash(request)
        request_embedding = await self._generate_request_embedding(request)
        
        # Check if cache is full
        if len(self.cache) >= self.max_entries:
            await self._evict_entries()
        
        # Create cache entry
        entry = CacheEntry(
            request_hash=request_hash,
            request_embedding=request_embedding,
            response=response,
            model_used=model_used,
            timestamp=datetime.utcnow(),
            access_count=1,
            ttl=ttl or self.default_ttl,
            cost=cost
        )
        
        # Store in cache
        self.cache[request_hash] = entry
        self._save_cache()
    
    async def _find_similar_request(self, request_embedding: np.ndarray) -> Optional[CacheEntry]:
        """Find semantically similar request in cache"""
        if not self.cache:
            return None
        
        similarities = []
        for entry in self.cache.values():
            if entry.is_expired():
                continue
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                request_embedding.reshape(1, -1),
                entry.request_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= self.similarity_threshold:
                similarities.append((similarity, entry))
        
        # Return most similar entry
        if similarities:
            return max(similarities, key=lambda x: x[0])[1]
        
        return None
    
    async def _evict_entries(self):
        """Evict least recently used entries"""
        # Sort by access count and timestamp
        entries = sorted(
            self.cache.values(),
            key=lambda x: (x.access_count, x.timestamp)
        )
        
        # Remove 20% of entries
        entries_to_remove = entries[:len(self.cache) // 5]
        for entry in entries_to_remove:
            del self.cache[entry.request_hash]
    
    def _generate_request_hash(self, request: Dict[str, Any]) -> str:
        """Generate hash for request"""
        request_str = json.dumps(request, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    async def _generate_request_embedding(self, request: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for request"""
        # Simple text-based embedding (in production, use proper embedding model)
        text = request.get("input", "")
        if isinstance(text, dict):
            text = str(text)
        
        # Convert to simple numerical embedding
        # This is a placeholder - in production, use proper sentence transformers
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding = np.array([int(text_hash[i:i+8], 16) % 1000 / 1000.0 
                             for i in range(0, 32, 8)])
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _load_cache(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, "semantic_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except:
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        cache_file = os.path.join(self.cache_dir, "semantic_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except:
            pass  # Ignore save errors
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "cache_size_mb": os.path.getsize(os.path.join(self.cache_dir, "semantic_cache.pkl")) / (1024 * 1024) if os.path.exists(os.path.join(self.cache_dir, "semantic_cache.pkl")) else 0,
            "hit_rate": self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder)"""
        # In production, track actual hits and misses
        return 0.75  # Placeholder value
    
    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self._save_cache()