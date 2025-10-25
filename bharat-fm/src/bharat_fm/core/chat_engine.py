"""
Enhanced Chat Engine for Bharat-FM
Integrates conversation memory with optimized inference
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime
import logging

from .inference_engine import InferenceEngine
from ..memory.conversation_memory import ConversationMemoryManager

logger = logging.getLogger(__name__)

class ChatEngine:
    """Enhanced chat engine with conversation memory and optimized inference"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.inference_engine = InferenceEngine(config.get("inference", {}))
        self.memory_manager = ConversationMemoryManager(config.get("memory", {}))
        
        # Engine state
        self.engine_id = str(uuid.uuid4())
        self.start_time = datetime.utcnow()
        self.is_running = False
        
        # Chat statistics
        self.chat_stats = {
            "total_conversations": 0,
            "total_exchanges": 0,
            "active_sessions": 0,
            "avg_response_time": 0.0,
            "memory_hits": 0
        }
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def start(self):
        """Start the chat engine"""
        if self.is_running:
            logger.warning("Chat engine is already running")
            return
        
        await self.inference_engine.start()
        await self.memory_manager.start()
        self.is_running = True
        
        logger.info(f"Chat engine {self.engine_id} started")
    
    async def stop(self):
        """Stop the chat engine"""
        if not self.is_running:
            logger.warning("Chat engine is not running")
            return
        
        # End all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_session(session_id)
        
        await self.inference_engine.stop()
        await self.memory_manager.stop()
        self.is_running = False
        
        logger.info(f"Chat engine {self.engine_id} stopped")
    
    async def generate_response(self, user_id: str, session_id: str, 
                              message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate chat response with conversation memory and optimization"""
        
        if not self.is_running:
            raise RuntimeError("Chat engine is not running")
        
        start_time = time.time()
        
        try:
            # Get or create session
            session = await self._get_or_create_session(user_id, session_id)
            
            # Retrieve conversation context
            conversation_context = await self.memory_manager.retrieve_relevant_context(
                user_id, message
            )
            
            # Store user message
            await self.memory_manager.store_exchange(
                user_id, session_id, message, "", {"type": "user_message"}
            )
            
            # Prepare inference request with context
            inference_request = await self._prepare_inference_request(
                message, conversation_context, context
            )
            
            # Execute optimized inference
            inference_response = await self.inference_engine.predict(inference_request)
            
            # Store assistant response
            await self.memory_manager.store_exchange(
                user_id, session_id, message, inference_response.get("output", ""),
                {
                    "type": "assistant_response",
                    "inference_id": inference_response.get("request_id"),
                    "latency": inference_response.get("latency", 0),
                    "cost": inference_response.get("cost", 0),
                    "cache_hit": inference_response.get("cache_hit", False)
                }
            )
            
            # Update session
            session["last_activity"] = datetime.utcnow()
            session["exchange_count"] += 1
            
            # Update statistics
            self._update_chat_stats(inference_response)
            
            # Prepare response
            response_time = time.time() - start_time
            
            response = {
                "session_id": session_id,
                "user_id": user_id,
                "message": message,
                "response": inference_response.get("output", {}),
                "context_used": {
                    "recent_exchanges": len(conversation_context.get("recent_history", [])),
                    "relevant_exchanges": len(conversation_context.get("relevant_exchanges", [])),
                    "personalization_applied": bool(conversation_context.get("personalization")),
                    "emotional_state_considered": bool(conversation_context.get("emotional_state"))
                },
                "performance": {
                    "response_time": response_time,
                    "inference_latency": inference_response.get("latency", 0),
                    "cache_hit": inference_response.get("cache_hit", False),
                    "batched": inference_response.get("batched", False),
                    "cost": inference_response.get("cost", 0)
                },
                "session_info": {
                    "exchange_count": session["exchange_count"],
                    "session_duration": (datetime.utcnow() - session["start_time"]).total_seconds(),
                    "personalization": conversation_context.get("personalization", {}),
                    "emotional_state": conversation_context.get("emotional_state", {})
                },
                "timestamp": datetime.utcnow().isoformat(),
                "engine_id": self.engine_id
            }
            
            logger.info(f"Chat response generated for user {user_id}, session {session_id} in {response_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating chat response for user {user_id}, session {session_id}: {e}")
            raise
    
    async def generate_streaming_response(self, user_id: str, session_id: str, 
                                       message: str, context: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """Generate streaming chat response"""
        
        if not self.is_running:
            raise RuntimeError("Chat engine is not running")
        
        # Get or create session
        session = await self._get_or_create_session(user_id, session_id)
        
        # Send initial response
        yield {
            "type": "start",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Get conversation context
            conversation_context = await self.memory_manager.retrieve_relevant_context(
                user_id, message
            )
            
            # Prepare inference request
            inference_request = await self._prepare_inference_request(
                message, conversation_context, context
            )
            
            # Execute streaming inference
            async for chunk in self.inference_engine.predict_streaming(inference_request):
                if chunk["type"] == "chunk":
                    yield {
                        "type": "chunk",
                        "session_id": session_id,
                        "chunk": chunk["chunk"],
                        "chunk_index": chunk["chunk_index"],
                        "timestamp": chunk["timestamp"]
                    }
                elif chunk["type"] == "complete":
                    # Store complete response
                    complete_response = chunk["response"]
                    
                    await self.memory_manager.store_exchange(
                        user_id, session_id, message, complete_response.get("output", ""),
                        {
                            "type": "assistant_response",
                            "inference_id": complete_response.get("request_id"),
                            "latency": complete_response.get("latency", 0),
                            "cost": complete_response.get("cost", 0),
                            "cache_hit": complete_response.get("cache_hit", False)
                        }
                    )
                    
                    # Update session
                    session["last_activity"] = datetime.utcnow()
                    session["exchange_count"] += 1
                    
                    # Update statistics
                    self._update_chat_stats(complete_response)
                    
                    yield {
                        "type": "complete",
                        "session_id": session_id,
                        "response": complete_response,
                        "context_summary": {
                            "recent_exchanges": len(conversation_context.get("recent_history", [])),
                            "personalization_applied": bool(conversation_context.get("personalization"))
                        },
                        "timestamp": chunk["timestamp"]
                    }
                    
        except Exception as e:
            yield {
                "type": "error",
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_or_create_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get or create chat session"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "start_time": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "exchange_count": 0,
                "context": {}
            }
            self.chat_stats["active_sessions"] += 1
        
        return self.active_sessions[session_id]
    
    async def _prepare_inference_request(self, message: str, 
                                       conversation_context: Dict[str, Any],
                                       additional_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare inference request with conversation context"""
        
        # Build context-aware prompt
        prompt_parts = []
        
        # Add personalization context
        personalization = conversation_context.get("personalization", {})
        if personalization:
            if personalization.get("communication_style"):
                prompt_parts.append(f"Communication style: {personalization['communication_style']}")
            if personalization.get("expertise_level"):
                prompt_parts.append(f"Expertise level: {personalization['expertise_level']}")
            if personalization.get("language_preferences"):
                prompt_parts.append(f"Preferred languages: {', '.join(personalization['language_preferences'])}")
        
        # Add emotional context
        emotional_state = conversation_context.get("emotional_state", {})
        if emotional_state.get("current_emotion") and emotional_state["current_emotion"] != "neutral":
            prompt_parts.append(f"User seems {emotional_state['current_emotion']}")
        
        # Add conversation summary
        context_summary = conversation_context.get("context_summary", "")
        if context_summary:
            prompt_parts.append(f"Context: {context_summary}")
        
        # Add recent conversation history
        recent_history = conversation_context.get("recent_history", [])
        if recent_history:
            prompt_parts.append("Recent conversation:")
            for exchange in recent_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"User: {exchange['user_message']}")
                prompt_parts.append(f"Assistant: {exchange['assistant_response']}")
        
        # Add current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        # Combine prompt
        full_prompt = "\n".join(prompt_parts)
        
        # Prepare inference request
        inference_request = {
            "id": f"chat_{int(time.time() * 1000000)}",
            "input": {
                "prompt": full_prompt,
                "conversation_context": conversation_context,
                "additional_context": additional_context
            },
            "model_id": additional_context.get("model_id", "default") if additional_context else "default",
            "requirements": {
                "max_latency": additional_context.get("max_latency", 2.0) if additional_context else 2.0,
                "temperature": additional_context.get("temperature", 0.7) if additional_context else 0.7,
                "max_tokens": additional_context.get("max_tokens", 500) if additional_context else 500
            }
        }
        
        return inference_request
    
    def _update_chat_stats(self, inference_response: Dict[str, Any]):
        """Update chat statistics"""
        self.chat_stats["total_exchanges"] += 1
        
        # Update average response time
        latency = inference_response.get("latency", 0)
        current_avg = self.chat_stats["avg_response_time"]
        total_exchanges = self.chat_stats["total_exchanges"]
        
        self.chat_stats["avg_response_time"] = (
            (current_avg * (total_exchanges - 1) + latency) / total_exchanges
        )
        
        # Track cache hits
        if inference_response.get("cache_hit", False):
            self.chat_stats["memory_hits"] += 1
    
    async def start_session(self, user_id: str, session_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a new chat session"""
        session_id = str(uuid.uuid4())
        
        session = {
            "user_id": user_id,
            "session_id": session_id,
            "start_time": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "exchange_count": 0,
            "config": session_config or {},
            "context": {}
        }
        
        self.active_sessions[session_id] = session
        self.chat_stats["total_conversations"] += 1
        self.chat_stats["active_sessions"] += 1
        
        logger.info(f"Started new session {session_id} for user {user_id}")
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": session["start_time"].isoformat(),
            "config": session["config"]
        }
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a chat session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Archive conversation memory
        await self.memory_manager.end_session(session["user_id"], session_id)
        
        # Calculate session statistics
        session_duration = (datetime.utcnow() - session["start_time"]).total_seconds()
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        self.chat_stats["active_sessions"] -= 1
        
        session_info = {
            "session_id": session_id,
            "user_id": session["user_id"],
            "start_time": session["start_time"].isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": session_duration,
            "exchange_count": session["exchange_count"],
            "config": session["config"]
        }
        
        logger.info(f"Ended session {session_id} for user {session['user_id']}")
        
        return session_info
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "start_time": session["start_time"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "duration_seconds": (datetime.utcnow() - session["start_time"]).total_seconds(),
            "exchange_count": session["exchange_count"],
            "config": session["config"]
        }
    
    async def list_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active sessions"""
        sessions = []
        
        for session_id, session in self.active_sessions.items():
            if user_id is None or session["user_id"] == user_id:
                sessions.append({
                    "session_id": session_id,
                    "user_id": session["user_id"],
                    "start_time": session["start_time"].isoformat(),
                    "last_activity": session["last_activity"].isoformat(),
                    "exchange_count": session["exchange_count"]
                })
        
        return sessions
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user conversation profile"""
        return await self.memory_manager.get_user_profile(user_id)
    
    async def update_user_profile(self, user_id: str, profile_updates: Dict[str, Any]):
        """Update user conversation profile"""
        await self.memory_manager.update_user_profile(user_id, profile_updates)
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get chat engine status"""
        inference_status = await self.inference_engine.get_engine_status()
        memory_stats = await self.memory_manager.get_memory_stats()
        
        return {
            "engine_id": self.engine_id,
            "status": "running" if self.is_running else "stopped",
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "start_time": self.start_time.isoformat(),
            "chat_stats": self.chat_stats,
            "active_sessions": len(self.active_sessions),
            "inference_engine": inference_status,
            "memory_stats": memory_stats
        }
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed chat engine metrics"""
        inference_metrics = await self.inference_engine.get_detailed_metrics()
        
        return {
            "chat_stats": self.chat_stats,
            "inference_metrics": inference_metrics,
            "active_sessions": [
                {
                    "session_id": session_id,
                    "user_id": session["user_id"],
                    "duration": (datetime.utcnow() - session["start_time"]).total_seconds(),
                    "exchanges": session["exchange_count"]
                }
                for session_id, session in self.active_sessions.items()
            ],
            "memory_efficiency": {
                "cache_hit_rate": self.chat_stats["memory_hits"] / max(1, self.chat_stats["total_exchanges"]),
                "avg_memory_per_session": inference_metrics.get("cache_stats", {}).get("active_entries", 0) / max(1, len(self.active_sessions))
            }
        }
    
    async def configure_engine(self, config: Dict[str, Any]):
        """Configure chat engine parameters"""
        if "inference" in config:
            await self.inference_engine.configure_engine(config["inference"])
        
        # Update local config
        self.config.update(config)
        
        logger.info(f"Chat engine configuration updated: {config}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        inference_health = await self.inference_engine.health_check()
        
        health_status = {
            "engine_id": self.engine_id,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "inference_engine": inference_health["status"],
                "memory_manager": "running" if self.memory_manager.running else "stopped",
                "chat_engine": "running" if self.is_running else "stopped"
            },
            "metrics": {
                "active_sessions": len(self.active_sessions),
                "total_exchanges": self.chat_stats["total_exchanges"],
                "avg_response_time": self.chat_stats["avg_response_time"]
            }
        }
        
        # Determine overall health
        if (inference_health["status"] == "healthy" and 
            self.memory_manager.running and 
            self.is_running):
            health_status["status"] = "healthy"
        else:
            health_status["status"] = "degraded"
        
        return health_status


# Convenience function for easy usage
async def create_chat_engine(config: Optional[Dict[str, Any]] = None) -> ChatEngine:
    """Create and start chat engine"""
    engine = ChatEngine(config)
    await engine.start()
    return engine