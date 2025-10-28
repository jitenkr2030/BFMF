"""
Multi-Agent System for Bharat-FM
Implements collaborative problem-solving capabilities with multiple AI agents
"""

import numpy as np
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import uuid
from collections import defaultdict, deque
import pickle
import os

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Roles that agents can play in the system"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    COMMUNICATOR = "communicator"
    OPTIMIZER = "optimizer"
    VALIDATOR = "validator"
    LEARNER = "learner"

class TaskStatus(Enum):
    """Status of tasks in the system"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MessageType(Enum):
    """Types of messages between agents"""
    TASK_REQUEST = "task_request"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    COORDINATION_REQUEST = "coordination_request"
    KNOWLEDGE_SHARE = "knowledge_share"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"

@dataclass
class AgentCapability:
    """Capability that an agent possesses"""
    name: str
    description: str
    proficiency: float  # 0.0 to 1.0
    domain: str
    complexity_level: int  # 1 to 10

@dataclass
class Task:
    """Task that can be assigned to agents"""
    id: str
    name: str
    description: str
    required_capabilities: List[str]
    task_type: str
    priority: int  # 1 to 10
    estimated_duration: float  # in seconds
    deadline: Optional[float] = None  # timestamp
    dependencies: List[str] = None
    input_data: Any = None
    expected_output: Any = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class Message:
    """Message for communication between agents"""
    id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Any
    timestamp: float
    priority: int = 1  # 1 to 10
    requires_ack: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class Agent:
    """Individual agent in the multi-agent system"""
    id: str
    name: str
    role: AgentRole
    capabilities: List[AgentCapability]
    max_concurrent_tasks: int = 3
    current_tasks: List[str] = None
    performance_history: List[Dict] = None
    knowledge_base: Dict = None
    availability: float = 1.0  # 0.0 to 1.0
    reliability: float = 0.95  # 0.0 to 1.0
    learning_rate: float = 0.1  # 0.0 to 1.0
    
    def __post_init__(self):
        if self.current_tasks is None:
            self.current_tasks = []
        if self.performance_history is None:
            self.performance_history = []
        if self.knowledge_base is None:
            self.knowledge_base = {}
    
    def can_handle_task(self, task: Task) -> Tuple[bool, float]:
        """Check if agent can handle a task and return confidence score"""
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            return False, 0.0
        
        # Check capability match
        capability_scores = []
        for req_cap in task.required_capabilities:
            agent_cap = next((cap for cap in self.capabilities if cap.name == req_cap), None)
            if agent_cap:
                capability_scores.append(agent_cap.proficiency)
            else:
                capability_scores.append(0.0)
        
        if not capability_scores:
            return False, 0.0
        
        # Calculate overall confidence
        avg_proficiency = np.mean(capability_scores)
        availability_factor = self.availability
        reliability_factor = self.reliability
        
        confidence = avg_proficiency * availability_factor * reliability_factor
        return confidence > 0.5, confidence

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_config: Agent):
        self.config = agent_config
        self.message_queue = queue.Queue()
        self.task_queue = queue.Queue()
        self.running = False
        self.communication_handler = None
        self.learning_system = None
        
    @abstractmethod
    def process_task(self, task: Task) -> Any:
        """Process a task and return result"""
        pass
    
    @abstractmethod
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message and optionally return response"""
        pass
    
    def start(self):
        """Start the agent"""
        self.running = True
        self._run()
    
    def stop(self):
        """Stop the agent"""
        self.running = False
    
    def _run(self):
        """Main agent loop"""
        while self.running:
            try:
                # Process messages
                try:
                    message = self.message_queue.get(timeout=0.1)
                    response = self.handle_message(message)
                    if response and self.communication_handler:
                        self.communication_handler.send_message(response)
                except queue.Empty:
                    pass
                
                # Process tasks
                try:
                    task = self.task_queue.get(timeout=0.1)
                    result = self.process_task(task)
                    self._complete_task(task, result)
                except queue.Empty:
                    pass
                
                # Perform other agent activities
                self._perform_periodic_activities()
                
            except Exception as e:
                logger.error(f"Error in agent {self.config.id}: {e}")
    
    def _complete_task(self, task: Task, result: Any):
        """Complete a task and notify the system"""
        task.result = result
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        
        if task.id in self.config.current_tasks:
            self.config.current_tasks.remove(task.id)
        
        # Send completion message
        if self.communication_handler:
            completion_message = Message(
                id=str(uuid.uuid4()),
                sender_id=self.config.id,
                receiver_id="coordinator",
                message_type=MessageType.TASK_RESULT,
                content={
                    "task_id": task.id,
                    "result": result,
                    "completion_time": task.completed_at
                },
                timestamp=time.time()
            )
            self.communication_handler.send_message(completion_message)
    
    def _perform_periodic_activities(self):
        """Perform periodic agent activities"""
        # Learning and adaptation
        if self.learning_system:
            self.learning_system.update_knowledge(self.config)
        
        # Send heartbeat
        if self.communication_handler:
            heartbeat = Message(
                id=str(uuid.uuid4()),
                sender_id=self.config.id,
                receiver_id="coordinator",
                message_type=MessageType.HEARTBEAT,
                content={
                    "status": "active",
                    "current_tasks": len(self.config.current_tasks),
                    "availability": self.config.availability
                },
                timestamp=time.time()
            )
            self.communication_handler.send_message(heartbeat)

class SpecialistAgent(BaseAgent):
    """Specialist agent for specific domain tasks"""
    
    def __init__(self, agent_config: Agent):
        super().__init__(agent_config)
        self.domain_expertise = self._initialize_domain_expertise()
    
    def _initialize_domain_expertise(self) -> Dict:
        """Initialize domain-specific expertise"""
        expertise = {}
        for capability in self.config.capabilities:
            expertise[capability.name] = {
                "knowledge_level": capability.proficiency,
                "experience": 0,
                "success_rate": 0.0,
                "recent_tasks": deque(maxlen=10)
            }
        return expertise
    
    def process_task(self, task: Task) -> Any:
        """Process a specialized task"""
        logger.info(f"Specialist agent {self.config.id} processing task {task.id}")
        
        start_time = time.time()
        
        try:
            # Simulate task processing based on capabilities
            result = self._execute_specialized_task(task)
            
            # Update expertise
            self._update_expertise(task, True, time.time() - start_time)
            
            return result
            
        except Exception as e:
            # Update expertise with failure
            self._update_expertise(task, False, time.time() - start_time)
            raise e
    
    def _execute_specialized_task(self, task: Task) -> Any:
        """Execute specialized task based on agent capabilities"""
        # Simulate different types of specialized tasks
        if "analysis" in task.task_type.lower():
            return self._perform_analysis(task)
        elif "optimization" in task.task_type.lower():
            return self._perform_optimization(task)
        elif "prediction" in task.task_type.lower():
            return self._perform_prediction(task)
        elif "validation" in task.task_type.lower():
            return self._perform_validation(task)
        else:
            return self._perform_general_task(task)
    
    def _perform_analysis(self, task: Task) -> Dict:
        """Perform analysis task"""
        # Simulate analysis
        time.sleep(0.1)  # Simulate processing time
        
        return {
            "analysis_type": "specialized",
            "insights": f"Analysis completed by {self.config.name}",
            "confidence": np.random.uniform(0.7, 0.95),
            "details": {
                "agent_id": self.config.id,
                "capabilities_used": task.required_capabilities,
                "processing_time": time.time()
            }
        }
    
    def _perform_optimization(self, task: Task) -> Dict:
        """Perform optimization task"""
        # Simulate optimization
        time.sleep(0.2)  # Simulate processing time
        
        return {
            "optimization_type": "specialized",
            "improvement": np.random.uniform(0.1, 0.3),
            "solution": f"Optimization completed by {self.config.name}",
            "confidence": np.random.uniform(0.8, 0.95)
        }
    
    def _perform_prediction(self, task: Task) -> Dict:
        """Perform prediction task"""
        # Simulate prediction
        time.sleep(0.05)  # Simulate processing time
        
        return {
            "prediction_type": "specialized",
            "predicted_value": np.random.randn(),
            "confidence": np.random.uniform(0.6, 0.9),
            "method": f"Prediction by {self.config.name}"
        }
    
    def _perform_validation(self, task: Task) -> Dict:
        """Perform validation task"""
        # Simulate validation
        time.sleep(0.15)  # Simulate processing time
        
        return {
            "validation_type": "specialized",
            "is_valid": np.random.choice([True, False], p=[0.8, 0.2]),
            "confidence": np.random.uniform(0.85, 0.98),
            "validator": self.config.name
        }
    
    def _perform_general_task(self, task: Task) -> Dict:
        """Perform general task"""
        # Simulate general task
        time.sleep(0.1)  # Simulate processing time
        
        return {
            "task_type": "general",
            "result": f"Task completed by {self.config.name}",
            "success": True,
            "agent_id": self.config.id
        }
    
    def _update_expertise(self, task: Task, success: bool, duration: float):
        """Update domain expertise based on task performance"""
        for capability in task.required_capabilities:
            if capability in self.domain_expertise:
                expertise = self.domain_expertise[capability]
                expertise["experience"] += 1
                
                # Update success rate
                if expertise["experience"] == 1:
                    expertise["success_rate"] = 1.0 if success else 0.0
                else:
                    expertise["success_rate"] = (
                        expertise["success_rate"] * (expertise["experience"] - 1) + 
                        (1.0 if success else 0.0)
                    ) / expertise["experience"]
                
                # Add to recent tasks
                expertise["recent_tasks"].append({
                    "task_id": task.id,
                    "success": success,
                    "duration": duration,
                    "timestamp": time.time()
                })
                
                # Update proficiency based on performance
                if success:
                    performance_factor = min(duration / 10.0, 1.0)  # Normalize duration
                    expertise["knowledge_level"] = min(
                        1.0, 
                        expertise["knowledge_level"] + self.config.learning_rate * performance_factor
                    )
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages"""
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            task_data = message.content
            task = Task(**task_data)
            self.task_queue.put(task)
            
            # Add to current tasks
            self.config.current_tasks.append(task.id)
            
            # Send acknowledgment
            return Message(
                id=str(uuid.uuid4()),
                sender_id=self.config.id,
                receiver_id=message.sender_id,
                message_type=MessageType.STATUS_UPDATE,
                content={
                    "task_id": task.id,
                    "status": "accepted",
                    "estimated_completion": time.time() + task.estimated_duration
                },
                timestamp=time.time()
            )
        
        elif message.message_type == MessageType.KNOWLEDGE_SHARE:
            # Update knowledge base
            knowledge = message.content
            self.config.knowledge_base.update(knowledge)
        
        return None

class CoordinatorAgent(BaseAgent):
    """Coordinator agent that manages task distribution and coordination"""
    
    def __init__(self, agent_config: Agent):
        super().__init__(agent_config)
        self.task_registry = {}
        self.agent_registry = {}
        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.coordination_strategies = {
            "load_balancing": self._load_balancing_strategy,
            "capability_matching": self._capability_matching_strategy,
            "priority_scheduling": self._priority_scheduling_strategy
        }
    
    def register_agent(self, agent: Agent):
        """Register an agent with the coordinator"""
        self.agent_registry[agent.id] = agent
        logger.info(f"Registered agent {agent.id} with coordinator")
    
    def submit_task(self, task: Task) -> str:
        """Submit a new task to the system"""
        self.task_registry[task.id] = task
        self.task_queue.append(task)
        
        logger.info(f"Task {task.id} submitted to coordinator")
        return task.id
    
    def process_task(self, task: Task) -> Any:
        """Process coordination tasks"""
        # Coordinator doesn't process regular tasks
        pass
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages"""
        if message.message_type == MessageType.TASK_REQUEST:
            # Handle new task request
            task_data = message.content
            task = Task(**task_data)
            task_id = self.submit_task(task)
            
            return Message(
                id=str(uuid.uuid4()),
                sender_id=self.config.id,
                receiver_id=message.sender_id,
                message_type=MessageType.STATUS_UPDATE,
                content={
                    "task_id": task_id,
                    "status": "submitted",
                    "estimated_start": time.time() + 1.0
                },
                timestamp=time.time()
            )
        
        elif message.message_type == MessageType.TASK_RESULT:
            # Handle task completion
            result_data = message.content
            task_id = result_data["task_id"]
            
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                task.result = result_data["result"]
                task.status = TaskStatus.COMPLETED
                task.completed_at = result_data["completion_time"]
                
                self.completed_tasks.append(task)
                logger.info(f"Task {task_id} completed")
                
                # Handle task dependencies
                self._handle_task_dependencies(task)
        
        elif message.message_type == MessageType.HEARTBEAT:
            # Update agent status
            agent_id = message.sender_id
            if agent_id in self.agent_registry:
                agent = self.agent_registry[agent_id]
                agent.availability = message.content["availability"]
        
        elif message.message_type == MessageType.ERROR_REPORT:
            # Handle task failure
            error_data = message.content
            task_id = error_data["task_id"]
            
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                task.status = TaskStatus.FAILED
                task.error = error_data["error"]
                
                self.failed_tasks.append(task)
                logger.error(f"Task {task_id} failed: {error_data['error']}")
        
        return None
    
    def _perform_periodic_activities(self):
        """Perform periodic coordination activities"""
        super()._perform_periodic_activities()
        
        # Task scheduling and assignment
        self._schedule_tasks()
        
        # Monitor system health
        self._monitor_system_health()
        
        # Optimize agent performance
        self._optimize_agent_performance()
    
    def _schedule_tasks(self):
        """Schedule and assign tasks to agents"""
        if not self.task_queue:
            return
        
        # Sort tasks by priority and deadline
        ready_tasks = [
            task for task in self.task_queue 
            if task.status == TaskStatus.PENDING and self._are_dependencies_met(task)
        ]
        
        if not ready_tasks:
            return
        
        # Sort by priority (higher first) and deadline (earlier first)
        ready_tasks.sort(key=lambda t: (-t.priority, t.deadline or float('inf')))
        
        # Assign tasks using coordination strategy
        for task in ready_tasks[:5]:  # Limit assignments per cycle
            assigned_agent = self._assign_task_to_agent(task)
            if assigned_agent:
                self._send_task_assignment(task, assigned_agent)
    
    def _assign_task_to_agent(self, task: Task) -> Optional[str]:
        """Assign task to the most suitable agent"""
        best_agent_id = None
        best_score = 0.0
        
        for agent_id, agent in self.agent_registry.items():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle and confidence > best_score:
                best_score = confidence
                best_agent_id = agent_id
        
        return best_agent_id
    
    def _send_task_assignment(self, task: Task, agent_id: str):
        """Send task assignment to agent"""
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent_id
        task.started_at = time.time()
        
        assignment_message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.config.id,
            receiver_id=agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            content=asdict(task),
            timestamp=time.time()
        )
        
        if self.communication_handler:
            self.communication_handler.send_message(assignment_message)
        
        logger.info(f"Task {task.id} assigned to agent {agent_id}")
    
    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if task dependencies are met"""
        for dep_id in task.dependencies:
            if dep_id in self.task_registry:
                dep_task = self.task_registry[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        return True
    
    def _handle_task_dependencies(self, completed_task: Task):
        """Handle dependencies when a task is completed"""
        for task in self.task_queue:
            if completed_task.id in task.dependencies:
                if self._are_dependencies_met(task):
                    logger.info(f"Dependencies met for task {task.id}")
    
    def _monitor_system_health(self):
        """Monitor overall system health"""
        total_agents = len(self.agent_registry)
        active_agents = sum(1 for agent in self.agent_registry.values() if agent.availability > 0.5)
        
        pending_tasks = len([t for t in self.task_registry.values() if t.status == TaskStatus.PENDING])
        in_progress_tasks = len([t for t in self.task_registry.values() if t.status == TaskStatus.IN_PROGRESS])
        
        system_health = {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "success_rate": len(self.completed_tasks) / max(1, len(self.completed_tasks) + len(self.failed_tasks))
        }
        
        # Log system health periodically
        if int(time.time()) % 30 == 0:  # Every 30 seconds
            logger.info(f"System health: {system_health}")
    
    def _optimize_agent_performance(self):
        """Optimize agent performance based on historical data"""
        for agent_id, agent in self.agent_registry.items():
            # Analyze performance history
            if len(agent.performance_history) > 10:
                recent_performance = agent.performance_history[-10:]
                success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
                
                # Adjust agent parameters based on performance
                if success_rate < 0.7:
                    agent.max_concurrent_tasks = max(1, agent.max_concurrent_tasks - 1)
                    logger.info(f"Reduced max concurrent tasks for agent {agent_id} due to low performance")
                elif success_rate > 0.9 and agent.max_concurrent_tasks < 5:
                    agent.max_concurrent_tasks += 1
                    logger.info(f"Increased max concurrent tasks for agent {agent_id} due to high performance")
    
    def _load_balancing_strategy(self, task: Task) -> Optional[str]:
        """Load balancing strategy for task assignment"""
        # Find agent with least current tasks
        min_tasks = float('inf')
        best_agent_id = None
        
        for agent_id, agent in self.agent_registry.items():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle and len(agent.current_tasks) < min_tasks:
                min_tasks = len(agent.current_tasks)
                best_agent_id = agent_id
        
        return best_agent_id
    
    def _capability_matching_strategy(self, task: Task) -> Optional[str]:
        """Capability matching strategy for task assignment"""
        # Find agent with best capability match
        best_score = 0.0
        best_agent_id = None
        
        for agent_id, agent in self.agent_registry.items():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle and confidence > best_score:
                best_score = confidence
                best_agent_id = agent_id
        
        return best_agent_id
    
    def _priority_scheduling_strategy(self, task: Task) -> Optional[str]:
        """Priority scheduling strategy for task assignment"""
        # Consider both priority and agent capability
        best_score = 0.0
        best_agent_id = None
        
        for agent_id, agent in self.agent_registry.items():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle:
                # Combine priority and confidence
                score = task.priority * confidence
                if score > best_score:
                    best_score = score
                    best_agent_id = agent_id
        
        return best_agent_id
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            "coordinator_id": self.config.id,
            "registered_agents": len(self.agent_registry),
            "total_tasks": len(self.task_registry),
            "pending_tasks": len([t for t in self.task_registry.values() if t.status == TaskStatus.PENDING]),
            "in_progress_tasks": len([t for t in self.task_registry.values() if t.status == TaskStatus.IN_PROGRESS]),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "system_uptime": time.time() - (self.config.performance_history[0]["timestamp"] if self.config.performance_history else time.time())
        }

class CommunicationHandler:
    """Handles communication between agents"""
    
    def __init__(self):
        self.message_bus = defaultdict(list)
        self.subscribers = defaultdict(set)
        self.message_history = []
    
    def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """Subscribe agent to message types"""
        for msg_type in message_types:
            self.subscribers[msg_type].add(agent_id)
    
    def send_message(self, message: Message):
        """Send message to recipient"""
        # Store message
        self.message_history.append(message)
        
        # Deliver to recipient
        if message.receiver_id in self.subscribers.get(message.message_type, set()):
            self.message_bus[message.receiver_id].append(message)
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Get messages for an agent"""
        messages = self.message_bus[agent_id]
        self.message_bus[agent_id] = []
        return messages
    
    def broadcast_message(self, sender_id: str, message_type: MessageType, content: Any):
        """Broadcast message to all subscribers"""
        for receiver_id in self.subscribers.get(message_type, set()):
            message = Message(
                id=str(uuid.uuid4()),
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type,
                content=content,
                timestamp=time.time()
            )
            self.send_message(message)

class LearningSystem:
    """Learning and adaptation system for agents"""
    
    def __init__(self):
        self.knowledge_base = defaultdict(dict)
        self.performance_patterns = defaultdict(list)
        self.learning_strategies = {
            "reinforcement": self._reinforcement_learning,
            "imitation": self._imitation_learning,
            "collaborative": self._collaborative_learning
        }
    
    def update_knowledge(self, agent: Agent):
        """Update agent knowledge based on experience"""
        # Analyze performance history
        if len(agent.performance_history) > 5:
            recent_performance = agent.performance_history[-5:]
            
            # Identify patterns
            success_patterns = self._identify_success_patterns(recent_performance)
            failure_patterns = self._identify_failure_patterns(recent_performance)
            
            # Update knowledge base
            self.knowledge_base[agent.id]["success_patterns"] = success_patterns
            self.knowledge_base[agent.id]["failure_patterns"] = failure_patterns
            
            # Apply learning strategies
            self._apply_learning_strategies(agent)
    
    def _identify_success_patterns(self, performance_history: List[Dict]) -> List[Dict]:
        """Identify patterns in successful tasks"""
        successful_tasks = [p for p in performance_history if p["success"]]
        
        patterns = []
        if successful_tasks:
            # Analyze common characteristics
            common_capabilities = set()
            for task in successful_tasks:
                common_capabilities.update(task.get("capabilities_used", []))
            
            patterns.append({
                "type": "capability_preference",
                "capabilities": list(common_capabilities),
                "success_rate": len(successful_tasks) / len(performance_history)
            })
        
        return patterns
    
    def _identify_failure_patterns(self, performance_history: List[Dict]) -> List[Dict]:
        """Identify patterns in failed tasks"""
        failed_tasks = [p for p in performance_history if not p["success"]]
        
        patterns = []
        if failed_tasks:
            # Analyze common failure characteristics
            common_errors = defaultdict(int)
            for task in failed_tasks:
                error_type = task.get("error_type", "unknown")
                common_errors[error_type] += 1
            
            patterns.append({
                "type": "error_pattern",
                "common_errors": dict(common_errors),
                "failure_rate": len(failed_tasks) / len(performance_history)
            })
        
        return patterns
    
    def _apply_learning_strategies(self, agent: Agent):
        """Apply learning strategies to improve agent performance"""
        # Reinforcement learning
        self._reinforcement_learning(agent)
        
        # Collaborative learning
        self._collaborative_learning(agent)
    
    def _reinforcement_learning(self, agent: Agent):
        """Apply reinforcement learning to improve agent performance"""
        if agent.id in self.knowledge_base:
            knowledge = self.knowledge_base[agent.id]
            
            # Adjust parameters based on success patterns
            if "success_patterns" in knowledge:
                for pattern in knowledge["success_patterns"]:
                    if pattern["type"] == "capability_preference":
                        # Improve proficiency in preferred capabilities
                        for cap_name in pattern["capabilities"]:
                            for capability in agent.capabilities:
                                if capability.name == cap_name:
                                    capability.proficiency = min(
                                        1.0, 
                                        capability.proficiency + agent.learning_rate * 0.1
                                    )
    
    def _imitation_learning(self, agent: Agent):
        """Apply imitation learning from successful agents"""
        # Find successful agents with similar capabilities
        similar_agents = []
        for other_agent_id, knowledge in self.knowledge_base.items():
            if other_agent_id != agent.id and "success_patterns" in knowledge:
                # Check capability similarity
                agent_capabilities = {cap.name for cap in agent.capabilities}
                other_capabilities = set()
                for pattern in knowledge["success_patterns"]:
                    if pattern["type"] == "capability_preference":
                        other_capabilities.update(pattern["capabilities"])
                
                similarity = len(agent_capabilities & other_capabilities) / len(agent_capabilities | other_capabilities)
                if similarity > 0.5:
                    similar_agents.append((other_agent_id, similarity))
        
        # Learn from similar successful agents
        for other_agent_id, similarity in similar_agents[:3]:  # Top 3 similar agents
            other_knowledge = self.knowledge_base[other_agent_id]
            if "success_patterns" in other_knowledge:
                # Adapt successful strategies
                for pattern in other_knowledge["success_patterns"]:
                    if pattern["type"] == "capability_preference":
                        # Slightly adjust capabilities
                        for cap_name in pattern["capabilities"]:
                            for capability in agent.capabilities:
                                if capability.name == cap_name:
                                    capability.proficiency = min(
                                        1.0,
                                        capability.proficiency + agent.learning_rate * similarity * 0.05
                                    )
    
    def _collaborative_learning(self, agent: Agent):
        """Apply collaborative learning from agent interactions"""
        # Share knowledge between agents with complementary capabilities
        if agent.id in self.knowledge_base:
            agent_knowledge = self.knowledge_base[agent.id]
            
            # Find complementary agents
            complementary_agents = []
            for other_agent_id, other_knowledge in self.knowledge_base.items():
                if other_agent_id != agent.id:
                    # Check capability complementarity
                    agent_capabilities = {cap.name for cap in agent.capabilities}
                    other_capabilities = set()
                    if "success_patterns" in other_knowledge:
                        for pattern in other_knowledge["success_patterns"]:
                            if pattern["type"] == "capability_preference":
                                other_capabilities.update(pattern["capabilities"])
                    
                    # Complementary if they have different but non-overlapping capabilities
                    intersection = agent_capabilities & other_capabilities
                    union = agent_capabilities | other_capabilities
                    
                    if len(intersection) / len(union) < 0.3:  # Low overlap
                        complementary_agents.append(other_agent_id)
            
            # Learn from complementary agents
            for other_agent_id in complementary_agents[:2]:  # Top 2 complementary agents
                other_knowledge = self.knowledge_base[other_agent_id]
                if "success_patterns" in other_knowledge:
                    # Learn new strategies
                    for pattern in other_knowledge["success_patterns"]:
                        if pattern["type"] == "capability_preference":
                            # Consider adding new capabilities or improving existing ones
                            for cap_name in pattern["capabilities"]:
                                # Check if agent has this capability
                                has_capability = any(cap.name == cap_name for cap in agent.capabilities)
                                if has_capability:
                                    # Improve existing capability
                                    for capability in agent.capabilities:
                                        if capability.name == cap_name:
                                            capability.proficiency = min(
                                                1.0,
                                                capability.proficiency + agent.learning_rate * 0.02
                                            )

class MultiAgentSystem:
    """Main multi-agent system coordinator"""
    
    def __init__(self):
        self.coordinator = None
        self.agents = {}
        self.communication_handler = CommunicationHandler()
        self.learning_system = LearningSystem()
        self.running = False
        self.system_metrics = {
            "tasks_processed": 0,
            "average_completion_time": 0.0,
            "success_rate": 0.0,
            "agent_utilization": {}
        }
    
    def initialize_system(self):
        """Initialize the multi-agent system"""
        # Create coordinator agent
        coordinator_config = Agent(
            id="coordinator",
            name="System Coordinator",
            role=AgentRole.COORDINATOR,
            capabilities=[
                AgentCapability("coordination", "Task coordination and management", 0.95, "system", 8),
                AgentCapability("scheduling", "Task scheduling and optimization", 0.90, "system", 7),
                AgentCapability("monitoring", "System monitoring and health checks", 0.85, "system", 6)
            ]
        )
        
        self.coordinator = CoordinatorAgent(coordinator_config)
        self.coordinator.communication_handler = self.communication_handler
        self.coordinator.learning_system = self.learning_system
        
        # Register coordinator with itself
        self.coordinator.register_agent(coordinator_config)
        self.agents["coordinator"] = self.coordinator
        
        # Create specialist agents
        self._create_specialist_agents()
        
        # Set up communication
        self._setup_communication()
        
        logger.info("Multi-agent system initialized")
    
    def _create_specialist_agents(self):
        """Create specialist agents for different domains"""
        specialist_configs = [
            {
                "id": "analyst_agent",
                "name": "Data Analyst",
                "role": AgentRole.SPECIALIST,
                "capabilities": [
                    AgentCapability("data_analysis", "Data analysis and insight generation", 0.85, "analytics", 7),
                    AgentCapability("statistical_analysis", "Statistical analysis and modeling", 0.80, "analytics", 8),
                    AgentCapability("visualization", "Data visualization and reporting", 0.75, "analytics", 6)
                ]
            },
            {
                "id": "optimizer_agent",
                "name": "System Optimizer",
                "role": AgentRole.SPECIALIST,
                "capabilities": [
                    AgentCapability("optimization", "System optimization and tuning", 0.90, "optimization", 9),
                    AgentCapability("performance_analysis", "Performance analysis and improvement", 0.85, "optimization", 8),
                    AgentCapability("resource_allocation", "Resource allocation and management", 0.80, "optimization", 7)
                ]
            },
            {
                "id": "predictor_agent",
                "name": "Prediction Specialist",
                "role": AgentRole.SPECIALIST,
                "capabilities": [
                    AgentCapability("prediction", "Predictive modeling and forecasting", 0.88, "prediction", 8),
                    AgentCapability("machine_learning", "Machine learning model training", 0.82, "prediction", 9),
                    AgentCapability("time_series", "Time series analysis and forecasting", 0.78, "prediction", 7)
                ]
            },
            {
                "id": "validator_agent",
                "name": "Quality Validator",
                "role": AgentRole.VALIDATOR,
                "capabilities": [
                    AgentCapability("validation", "Quality validation and testing", 0.92, "validation", 8),
                    AgentCapability("error_detection", "Error detection and correction", 0.87, "validation", 7),
                    AgentCapability("compliance", "Compliance checking and validation", 0.83, "validation", 6)
                ]
            }
        ]
        
        for config_data in specialist_configs:
            agent_config = Agent(**config_data)
            agent = SpecialistAgent(agent_config)
            agent.communication_handler = self.communication_handler
            agent.learning_system = self.learning_system
            
            # Register agent with coordinator
            self.coordinator.register_agent(agent_config)
            self.agents[agent_config.id] = agent
    
    def _setup_communication(self):
        """Set up communication between agents"""
        # Subscribe agents to relevant message types
        for agent_id, agent in self.agents.items():
            if isinstance(agent, CoordinatorAgent):
                self.communication_handler.subscribe(agent_id, [
                    MessageType.TASK_REQUEST,
                    MessageType.TASK_RESULT,
                    MessageType.HEARTBEAT,
                    MessageType.ERROR_REPORT
                ])
            else:
                self.communication_handler.subscribe(agent_id, [
                    MessageType.TASK_ASSIGNMENT,
                    MessageType.KNOWLEDGE_SHARE
                ])
    
    def start_system(self):
        """Start the multi-agent system"""
        logger.info("Starting multi-agent system...")
        
        self.running = True
        
        # Start all agents
        for agent in self.agents.values():
            agent.start()
        
        # Start system monitor
        self._start_system_monitor()
        
        logger.info("Multi-agent system started")
    
    def stop_system(self):
        """Stop the multi-agent system"""
        logger.info("Stopping multi-agent system...")
        
        self.running = False
        
        # Stop all agents
        for agent in self.agents.values():
            agent.stop()
        
        logger.info("Multi-agent system stopped")
    
    def submit_task(self, task: Task) -> str:
        """Submit a task to the multi-agent system"""
        if not self.running:
            raise RuntimeError("Multi-agent system is not running")
        
        # Send task request to coordinator
        task_request = Message(
            id=str(uuid.uuid4()),
            sender_id="user",
            receiver_id="coordinator",
            message_type=MessageType.TASK_REQUEST,
            content=asdict(task),
            timestamp=time.time()
        )
        
        self.communication_handler.send_message(task_request)
        
        # Submit task directly to coordinator
        return self.coordinator.submit_task(task)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        base_status = self.coordinator.get_system_status()
        
        # Add agent-specific status
        agent_status = {}
        for agent_id, agent in self.agents.items():
            agent_status[agent_id] = {
                "name": agent.config.name,
                "role": agent.config.role.value,
                "availability": agent.config.availability,
                "current_tasks": len(agent.config.current_tasks),
                "capabilities": [cap.name for cap in agent.config.capabilities],
                "performance_score": self._calculate_agent_performance_score(agent.config)
            }
        
        # Add system metrics
        system_metrics = self.system_metrics.copy()
        
        return {
            **base_status,
            "agent_status": agent_status,
            "system_metrics": system_metrics,
            "learning_system_status": {
                "knowledge_base_size": len(self.learning_system.knowledge_base),
                "active_learning_strategies": len(self.learning_system.learning_strategies)
            }
        }
    
    def _calculate_agent_performance_score(self, agent: Agent) -> float:
        """Calculate performance score for an agent"""
        if not agent.performance_history:
            return 0.5
        
        # Calculate success rate
        recent_history = agent.performance_history[-10:]  # Last 10 tasks
        success_rate = sum(1 for p in recent_history if p.get("success", False)) / len(recent_history)
        
        # Calculate efficiency (tasks completed per time unit)
        completion_times = [p.get("completion_time", 0) for p in recent_history if p.get("success", False)]
        avg_completion_time = np.mean(completion_times) if completion_times else 0
        efficiency = 1.0 / (1.0 + avg_completion_time / 60.0)  # Normalize to 1 minute
        
        # Calculate reliability
        reliability = agent.reliability
        
        # Combine metrics
        performance_score = (success_rate * 0.4 + efficiency * 0.3 + reliability * 0.3)
        
        return performance_score
    
    def _start_system_monitor(self):
        """Start system monitoring in a separate thread"""
        def monitor():
            while self.running:
                time.sleep(0.1)  # Check every 100ms for faster message delivery
                
                # Deliver messages to agents
                self._deliver_messages()
                
                # Update system metrics every 10 seconds
                if int(time.time()) % 10 == 0:
                    self._update_system_metrics()
                
                # Check for system optimization opportunities
                self._optimize_system()
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _deliver_messages(self):
        """Deliver messages from communication handler to agents"""
        for agent_id, agent in self.agents.items():
            messages = self.communication_handler.get_messages(agent_id)
            for message in messages:
                agent.message_queue.put(message)
    
    def _update_system_metrics(self):
        """Update system performance metrics"""
        # Calculate metrics from coordinator data
        completed_tasks = self.coordinator.completed_tasks
        
        if completed_tasks:
            # Average completion time
            completion_times = [
                task.completed_at - task.started_at 
                for task in completed_tasks 
                if task.completed_at and task.started_at
            ]
            avg_completion_time = np.mean(completion_times) if completion_times else 0
            
            # Success rate
            successful_tasks = len(completed_tasks)
            total_tasks = successful_tasks + len(self.coordinator.failed_tasks)
            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
            
            # Update metrics
            self.system_metrics["tasks_processed"] = len(completed_tasks)
            self.system_metrics["average_completion_time"] = avg_completion_time
            self.system_metrics["success_rate"] = success_rate
            
            # Agent utilization
            for agent_id, agent in self.agents.items():
                if agent_id != "coordinator":
                    utilization = len(agent.current_tasks) / agent.max_concurrent_tasks
                    self.system_metrics["agent_utilization"][agent_id] = utilization
    
    def _optimize_system(self):
        """Optimize system performance"""
        # Analyze system performance and make improvements
        if self.system_metrics["success_rate"] < 0.8:
            # Low success rate - consider adjusting agent parameters
            for agent_id, agent in self.agents.items():
                if agent_id != "coordinator":
                    # Reduce max concurrent tasks for struggling agents
                    if agent.reliability < 0.8:
                        agent.max_concurrent_tasks = max(1, agent.max_concurrent_tasks - 1)
        
        # Balance workload across agents
        self._balance_workload()
    
    def _balance_workload(self):
        """Balance workload across agents"""
        # Find overutilized and underutilized agents
        overutilized = []
        underutilized = []
        
        for agent_id, agent in self.agents.items():
            if agent_id != "coordinator":
                utilization = len(agent.current_tasks) / agent.max_concurrent_tasks
                if utilization > 0.8:
                    overutilized.append((agent_id, utilization))
                elif utilization < 0.3:
                    underutilized.append((agent_id, utilization))
        
        # Suggest workload redistribution
        if overutilized and underutilized:
            logger.info(f"Workload imbalance detected. Overutilized: {overutilized}, Underutilized: {underutilized}")
            
            # In a real implementation, this would trigger task redistribution
            # For now, just log the recommendation

# Factory functions
def create_multi_agent_system() -> MultiAgentSystem:
    """Create and initialize a multi-agent system"""
    system = MultiAgentSystem()
    system.initialize_system()
    return system

def create_task(name: str, description: str, required_capabilities: List[str], 
               task_type: str, priority: int = 5, estimated_duration: float = 60.0,
               input_data: Any = None) -> Task:
    """Create a new task"""
    return Task(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        required_capabilities=required_capabilities,
        task_type=task_type,
        priority=priority,
        estimated_duration=estimated_duration,
        input_data=input_data
    )

# Example usage and testing
def test_multi_agent_system():
    """Test the multi-agent system functionality"""
    print("Testing Multi-Agent System...")
    
    # Create and initialize system
    system = create_multi_agent_system()
    
    # Start the system
    system.start_system()
    
    # Create sample tasks
    tasks = [
        create_task(
            name="Data Analysis Task",
            description="Analyze sales data and generate insights",
            required_capabilities=["data_analysis", "statistical_analysis"],
            task_type="analysis",
            priority=8,
            estimated_duration=30.0
        ),
        create_task(
            name="System Optimization Task",
            description="Optimize system performance parameters",
            required_capabilities=["optimization", "performance_analysis"],
            task_type="optimization",
            priority=7,
            estimated_duration=45.0
        ),
        create_task(
            name="Prediction Task",
            description="Predict future sales trends",
            required_capabilities=["prediction", "machine_learning"],
            task_type="prediction",
            priority=6,
            estimated_duration=60.0
        ),
        create_task(
            name="Validation Task",
            description="Validate data quality and consistency",
            required_capabilities=["validation", "error_detection"],
            task_type="validation",
            priority=5,
            estimated_duration=20.0
        )
    ]
    
    # Submit tasks to the system
    task_ids = []
    for task in tasks:
        task_id = system.submit_task(task)
        task_ids.append(task_id)
        print(f"Submitted task: {task.name} (ID: {task_id})")
    
    # Wait for tasks to complete
    print("Waiting for tasks to complete...")
    time.sleep(10)  # Wait for processing
    
    # Get system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"Registered agents: {status['registered_agents']}")
    print(f"Completed tasks: {status['completed_tasks']}")
    print(f"Failed tasks: {status['failed_tasks']}")
    print(f"Success rate: {status['system_metrics']['success_rate']:.2%}")
    
    # Display agent status
    print(f"\nAgent Status:")
    for agent_id, agent_info in status['agent_status'].items():
        print(f"  {agent_info['name']} ({agent_id}):")
        print(f"    Role: {agent_info['role']}")
        print(f"    Availability: {agent_info['availability']:.2%}")
        print(f"    Current tasks: {agent_info['current_tasks']}")
        print(f"    Performance score: {agent_info['performance_score']:.2%}")
    
    # Stop the system
    system.stop_system()
    
    print("Multi-agent system tests completed!")

if __name__ == "__main__":
    test_multi_agent_system()