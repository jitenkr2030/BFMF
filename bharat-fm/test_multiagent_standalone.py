#!/usr/bin/env python3
"""
Standalone test script for Multi-Agent System components
"""

import sys
import os
import json
import time
import logging
import threading
import queue
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        avg_proficiency = sum(capability_scores) / len(capability_scores)
        availability_factor = self.availability
        reliability_factor = self.reliability
        
        confidence = avg_proficiency * availability_factor * reliability_factor
        return confidence > 0.5, confidence

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
    
    def update_knowledge(self, agent_config: Agent):
        """Update agent knowledge based on performance"""
        # Simple learning implementation
        for capability in agent_config.capabilities:
            if capability.name not in self.knowledge_base[agent_config.id]:
                self.knowledge_base[agent_config.id][capability.name] = {
                    "experience": 0,
                    "success_rate": 0.0,
                    "avg_performance": 0.0
                }
    
    def _reinforcement_learning(self, agent_id: str, performance_data: Dict):
        """Reinforcement learning strategy"""
        pass
    
    def _imitation_learning(self, agent_id: str, performance_data: Dict):
        """Imitation learning strategy"""
        pass
    
    def _collaborative_learning(self, agent_id: str, performance_data: Dict):
        """Collaborative learning strategy"""
        pass

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
    
    def _execute_specialized_task(self, task: Task) -> Dict:
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
            "confidence": random.uniform(0.7, 0.95),
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
            "improvement": random.uniform(0.1, 0.3),
            "solution": f"Optimization completed by {self.config.name}",
            "confidence": random.uniform(0.8, 0.95)
        }
    
    def _perform_prediction(self, task: Task) -> Dict:
        """Perform prediction task"""
        # Simulate prediction
        time.sleep(0.05)  # Simulate processing time
        
        return {
            "prediction_type": "specialized",
            "predicted_value": random.random(),
            "confidence": random.uniform(0.6, 0.9),
            "method": f"Prediction by {self.config.name}"
        }
    
    def _perform_validation(self, task: Task) -> Dict:
        """Perform validation task"""
        # Simulate validation
        time.sleep(0.15)  # Simulate processing time
        
        return {
            "validation_type": "specialized",
            "is_valid": random.choice([True, False], p=[0.8, 0.2]),
            "confidence": random.uniform(0.85, 0.98),
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
        """Submit a task to the coordinator"""
        self.task_registry[task.id] = task
        self.task_queue.append(task)
        logger.info(f"Task {task.id} submitted to coordinator")
        return task.id
    
    def process_task(self, task: Task) -> Any:
        """Process task coordination"""
        # Assign task to suitable agent
        assigned_agent = self._assign_task(task)
        
        if assigned_agent:
            # Send task to agent
            assignment_message = Message(
                id=str(uuid.uuid4()),
                sender_id=self.config.id,
                receiver_id=assigned_agent.id,
                message_type=MessageType.TASK_ASSIGNMENT,
                content=task.__dict__,
                timestamp=time.time()
            )
            self.communication_handler.send_message(assignment_message)
            
            return {"status": "assigned", "agent": assigned_agent.id}
        else:
            return {"status": "no_suitable_agent"}
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages"""
        if message.message_type == MessageType.TASK_REQUEST:
            # Handle task request
            task_data = message.content
            task = Task(**task_data)
            return self.submit_task(task)
        
        elif message.message_type == MessageType.TASK_RESULT:
            # Handle task completion
            result_data = message.content
            task_id = result_data.get("task_id")
            
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                task.status = TaskStatus.COMPLETED
                task.result = result_data.get("result")
                task.completed_at = result_data.get("completion_time")
                
                self.completed_tasks.append(task)
                logger.info(f"Task {task_id} completed successfully")
        
        elif message.message_type == MessageType.HEARTBEAT:
            # Handle agent heartbeat
            agent_id = message.sender_id
            heartbeat_data = message.content
            
            # Update agent status
            if agent_id in self.agent_registry:
                agent = self.agent_registry[agent_id]
                agent.availability = heartbeat_data.get("availability", 1.0)
        
        return None
    
    def _assign_task(self, task: Task) -> Optional[Agent]:
        """Assign task to suitable agent using coordination strategies"""
        # Use capability matching strategy
        return self._capability_matching_strategy(task)
    
    def _capability_matching_strategy(self, task: Task) -> Optional[Agent]:
        """Assign task based on capability matching"""
        best_agent = None
        best_score = 0.0
        
        for agent in self.agent_registry.values():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle and confidence > best_score:
                best_score = confidence
                best_agent = agent
        
        return best_agent
    
    def _load_balancing_strategy(self, task: Task) -> Optional[Agent]:
        """Assign task based on load balancing"""
        available_agents = []
        
        for agent in self.agent_registry.values():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle:
                available_agents.append((agent, len(agent.current_tasks)))
        
        if available_agents:
            # Select agent with least load
            return min(available_agents, key=lambda x: x[1])[0]
        
        return None
    
    def _priority_scheduling_strategy(self, task: Task) -> Optional[Agent]:
        """Assign task based on priority scheduling"""
        # For high priority tasks, use best available agent
        if task.priority >= 8:
            return self._capability_matching_strategy(task)
        else:
            return self._load_balancing_strategy(task)

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
        
        return self.coordinator.submit_task(task)
    
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
            avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
            
            # Success rate
            successful_tasks = len(completed_tasks)
            total_tasks = successful_tasks + len(self.coordinator.failed_tasks)
            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
            
            # Update metrics
            self.system_metrics["tasks_processed"] = successful_tasks
            self.system_metrics["average_completion_time"] = avg_completion_time
            self.system_metrics["success_rate"] = success_rate
    
    def _optimize_system(self):
        """Optimize system performance"""
        # Simple optimization - adjust agent availability based on load
        for agent_id, agent in self.agents.items():
            if agent_id != "coordinator":
                load_factor = len(agent.current_tasks) / agent.max_concurrent_tasks
                if load_factor > 0.8:
                    agent.availability = max(0.5, agent.availability - 0.05)
                elif load_factor < 0.3:
                    agent.availability = min(1.0, agent.availability + 0.05)

def test_system_initialization():
    """Test multi-agent system initialization"""
    logger.info("=== Testing System Initialization ===")
    
    try:
        # Create and initialize system
        system = MultiAgentSystem()
        system.initialize_system()
        
        # Check if agents were created
        expected_agents = ["coordinator", "analyst_agent", "optimizer_agent", "predictor_agent", "validator_agent"]
        actual_agents = list(system.agents.keys())
        
        logger.info(f"Expected agents: {expected_agents}")
        logger.info(f"Actual agents: {actual_agents}")
        
        # Check if all expected agents are present
        missing_agents = set(expected_agents) - set(actual_agents)
        if missing_agents:
            raise Exception(f"Missing agents: {missing_agents}")
        
        # Check agent capabilities
        for agent_id, agent in system.agents.items():
            logger.info(f"Agent {agent_id}: {len(agent.capabilities)} capabilities")
            for cap in agent.capabilities:
                logger.info(f"  - {cap.name}: {cap.proficiency}")
        
        return {
            "test_type": "system_initialization",
            "success": True,
            "agents_created": len(system.agents),
            "expected_agents": len(expected_agents),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"System initialization test failed: {str(e)}")
        return {
            "test_type": "system_initialization",
            "success": False,
            "error": str(e)
        }

def test_task_submission():
    """Test task submission and processing"""
    logger.info("=== Testing Task Submission ===")
    
    try:
        # Create and initialize system
        system = MultiAgentSystem()
        system.initialize_system()
        system.start_system()
        
        # Create sample tasks
        tasks = [
            Task(
                id="task_1",
                name="Data Analysis Task",
                description="Analyze sample dataset",
                required_capabilities=["data_analysis"],
                task_type="analysis",
                priority=5,
                estimated_duration=2.0
            ),
            Task(
                id="task_2",
                name="Optimization Task",
                description="Optimize system performance",
                required_capabilities=["optimization"],
                task_type="optimization",
                priority=7,
                estimated_duration=3.0
            ),
            Task(
                id="task_3",
                name="Prediction Task",
                description="Predict future values",
                required_capabilities=["prediction"],
                task_type="prediction",
                priority=6,
                estimated_duration=1.5
            )
        ]
        
        # Submit tasks
        submitted_tasks = []
        for task in tasks:
            task_id = system.submit_task(task)
            submitted_tasks.append(task_id)
            logger.info(f"Submitted task {task_id}")
        
        # Wait for tasks to complete
        time.sleep(5)
        
        # Check completed tasks
        completed_tasks = len(system.coordinator.completed_tasks)
        logger.info(f"Completed tasks: {completed_tasks}")
        
        # Stop system
        system.stop_system()
        
        return {
            "test_type": "task_submission",
            "success": True,
            "tasks_submitted": len(submitted_tasks),
            "tasks_completed": completed_tasks,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Task submission test failed: {str(e)}")
        return {
            "test_type": "task_submission",
            "success": False,
            "error": str(e)
        }

def test_agent_communication():
    """Test agent communication"""
    logger.info("=== Testing Agent Communication ===")
    
    try:
        # Create and initialize system
        system = MultiAgentSystem()
        system.initialize_system()
        system.start_system()
        
        # Test message sending
        test_message = Message(
            id="test_msg_1",
            sender_id="coordinator",
            receiver_id="analyst_agent",
            message_type=MessageType.KNOWLEDGE_SHARE,
            content={"test_data": "Hello from coordinator!"},
            timestamp=time.time()
        )
        
        # Send message
        system.communication_handler.send_message(test_message)
        logger.info("Sent test message")
        
        # Wait for message processing
        time.sleep(1)
        
        # Check message history
        message_history = len(system.communication_handler.message_history)
        logger.info(f"Message history length: {message_history}")
        
        # Stop system
        system.stop_system()
        
        return {
            "test_type": "agent_communication",
            "success": True,
            "messages_sent": 1,
            "message_history_length": message_history,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Agent communication test failed: {str(e)}")
        return {
            "test_type": "agent_communication",
            "success": False,
            "error": str(e)
        }

def main():
    """Run all multi-agent system tests"""
    logger.info("Starting Multi-Agent System Tests for Phase 5")
    
    test_results = []
    
    # Run tests
    test_results.append(test_system_initialization())
    test_results.append(test_task_submission())
    test_results.append(test_agent_communication())
    
    # Summary
    logger.info("=== Test Summary ===")
    successful_tests = sum(1 for result in test_results if result['success'])
    total_tests = len(test_results)
    
    logger.info(f"Tests completed: {successful_tests}/{total_tests}")
    
    for result in test_results:
        if result['success']:
            logger.info(f"✓ {result['test_type']}: PASSED")
        else:
            logger.info(f"✗ {result['test_type']}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Save results
    with open('/home/z/my-project/bharat-fm/multiagent_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    logger.info("Test results saved to multiagent_test_results.json")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)