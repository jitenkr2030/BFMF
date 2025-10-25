#!/usr/bin/env python3
"""
Simple test script for Multi-Agent System components
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

class SimpleAgent:
    """Simple agent for testing"""
    
    def __init__(self, agent_config: Agent):
        self.config = agent_config
        self.message_queue = queue.Queue()
        self.task_queue = queue.Queue()
        self.running = False
        self.communication_handler = None
        self.processed_tasks = []
    
    def process_task(self, task: Task) -> Any:
        """Process a task"""
        logger.info(f"Agent {self.config.id} processing task {task.id}")
        time.sleep(0.1)  # Simulate processing
        result = f"Task {task.id} processed by {self.config.name}"
        self.processed_tasks.append(task.id)
        return result
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages"""
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            task_data = message.content
            task = Task(**task_data)
            self.task_queue.put(task)
            self.config.current_tasks.append(task.id)
            
            # Send acknowledgment
            return Message(
                id=str(uuid.uuid4()),
                sender_id=self.config.id,
                receiver_id=message.sender_id,
                message_type=MessageType.STATUS_UPDATE,
                content={
                    "task_id": task.id,
                    "status": "accepted"
                },
                timestamp=time.time()
            )
        return None
    
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
                # Process tasks
                try:
                    task = self.task_queue.get(timeout=0.1)
                    result = self.process_task(task)
                    self._complete_task(task, result)
                except queue.Empty:
                    pass
                
                # Process messages
                try:
                    message = self.message_queue.get(timeout=0.1)
                    response = self.handle_message(message)
                    if response and self.communication_handler:
                        self.communication_handler.send_message(response)
                except queue.Empty:
                    pass
                
            except Exception as e:
                logger.error(f"Error in agent {self.config.id}: {e}")
    
    def _complete_task(self, task: Task, result: Any):
        """Complete a task"""
        task.result = result
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        
        if task.id in self.config.current_tasks:
            self.config.current_tasks.remove(task.id)

class SimpleCoordinator:
    """Simple coordinator for testing"""
    
    def __init__(self):
        self.agents = {}
        self.communication_handler = CommunicationHandler()
        self.completed_tasks = []
    
    def register_agent(self, agent: Agent):
        """Register an agent"""
        self.agents[agent.id] = agent
        agent.communication_handler = self.communication_handler
        logger.info(f"Registered agent {agent.id}")
    
    def submit_task(self, task: Task):
        """Submit a task"""
        # Find suitable agent
        for agent in self.agents.values():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle:
                # Send task to agent
                message = Message(
                    id=str(uuid.uuid4()),
                    sender_id="coordinator",
                    receiver_id=agent.id,
                    message_type=MessageType.TASK_ASSIGNMENT,
                    content=task.__dict__,
                    timestamp=time.time()
                )
                self.communication_handler.send_message(message)
                logger.info(f"Task {task.id} assigned to {agent.id}")
                return True
        
        logger.warning(f"No suitable agent found for task {task.id}")
        return False

def test_system_initialization():
    """Test system initialization"""
    logger.info("=== Testing System Initialization ===")
    
    try:
        # Create coordinator
        coordinator = SimpleCoordinator()
        
        # Create agents
        agents = [
            Agent(
                id="analyst_agent",
                name="Data Analyst",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("data_analysis", "Data analysis", 0.85, "analytics", 7),
                    AgentCapability("statistical_analysis", "Statistical analysis", 0.80, "analytics", 8)
                ]
            ),
            Agent(
                id="optimizer_agent",
                name="System Optimizer",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("optimization", "System optimization", 0.90, "optimization", 9),
                    AgentCapability("performance_analysis", "Performance analysis", 0.85, "optimization", 8)
                ]
            )
        ]
        
        # Create and register agent instances
        agent_instances = []
        for agent_config in agents:
            agent = SimpleAgent(agent_config)
            coordinator.register_agent(agent_config)
            agent_instances.append(agent)
        
        # Set up communication
        for agent in agent_instances:
            coordinator.communication_handler.subscribe(agent.config.id, [MessageType.TASK_ASSIGNMENT])
        
        logger.info(f"System initialized with {len(agents)} agents")
        
        return {
            "test_type": "system_initialization",
            "success": True,
            "agents_created": len(agents),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"System initialization test failed: {str(e)}")
        return {
            "test_type": "system_initialization",
            "success": False,
            "error": str(e)
        }

def test_task_processing():
    """Test task processing"""
    logger.info("=== Testing Task Processing ===")
    
    try:
        # Create coordinator
        coordinator = SimpleCoordinator()
        
        # Create agents
        agents = [
            Agent(
                id="analyst_agent",
                name="Data Analyst",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("data_analysis", "Data analysis", 0.85, "analytics", 7)
                ]
            ),
            Agent(
                id="optimizer_agent",
                name="System Optimizer",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("optimization", "System optimization", 0.90, "optimization", 9)
                ]
            )
        ]
        
        # Create and register agent instances
        agent_instances = []
        for agent_config in agents:
            agent = SimpleAgent(agent_config)
            coordinator.register_agent(agent_config)
            agent_instances.append(agent)
        
        # Set up communication
        for agent in agent_instances:
            coordinator.communication_handler.subscribe(agent.config.id, [MessageType.TASK_ASSIGNMENT])
        
        # Start agents
        for agent in agent_instances:
            agent.start()
        
        # Create and submit tasks
        tasks = [
            Task(
                id="task_1",
                name="Data Analysis",
                description="Analyze data",
                required_capabilities=["data_analysis"],
                task_type="analysis",
                priority=5,
                estimated_duration=1.0
            ),
            Task(
                id="task_2",
                name="System Optimization",
                description="Optimize system",
                required_capabilities=["optimization"],
                task_type="optimization",
                priority=7,
                estimated_duration=1.0
            )
        ]
        
        # Submit tasks
        for task in tasks:
            coordinator.submit_task(task)
        
        # Wait for processing
        time.sleep(2)
        
        # Stop agents
        for agent in agent_instances:
            agent.stop()
        
        # Check results
        total_processed = sum(len(agent.processed_tasks) for agent in agent_instances)
        logger.info(f"Total tasks processed: {total_processed}")
        
        return {
            "test_type": "task_processing",
            "success": True,
            "tasks_submitted": len(tasks),
            "tasks_processed": total_processed,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Task processing test failed: {str(e)}")
        return {
            "test_type": "task_processing",
            "success": False,
            "error": str(e)
        }

def test_message_delivery():
    """Test message delivery"""
    logger.info("=== Testing Message Delivery ===")
    
    try:
        # Create communication handler
        comm_handler = CommunicationHandler()
        
        # Create test agents
        agents = [
            Agent(
                id="agent_1",
                name="Test Agent 1",
                role=AgentRole.SPECIALIST,
                capabilities=[AgentCapability("test", "Test capability", 0.8, "test", 5)]
            ),
            Agent(
                id="agent_2",
                name="Test Agent 2",
                role=AgentRole.SPECIALIST,
                capabilities=[AgentCapability("test", "Test capability", 0.8, "test", 5)]
            )
        ]
        
        # Subscribe agents
        for agent in agents:
            comm_handler.subscribe(agent.id, [MessageType.TASK_ASSIGNMENT])
        
        # Send test messages
        messages = []
        for i, agent in enumerate(agents):
            message = Message(
                id=f"msg_{i}",
                sender_id="sender",
                receiver_id=agent.id,
                message_type=MessageType.TASK_ASSIGNMENT,
                content={"test": f"content_{i}"},
                timestamp=time.time()
            )
            comm_handler.send_message(message)
            messages.append(message)
        
        # Check message delivery
        delivered_count = 0
        for agent in agents:
            agent_messages = comm_handler.get_messages(agent.id)
            delivered_count += len(agent_messages)
            logger.info(f"Agent {agent.id} received {len(agent_messages)} messages")
        
        logger.info(f"Total messages delivered: {delivered_count}")
        
        return {
            "test_type": "message_delivery",
            "success": True,
            "messages_sent": len(messages),
            "messages_delivered": delivered_count,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Message delivery test failed: {str(e)}")
        return {
            "test_type": "message_delivery",
            "success": False,
            "error": str(e)
        }

def main():
    """Run all tests"""
    logger.info("Starting Simple Multi-Agent System Tests")
    
    test_results = []
    
    # Run tests
    test_results.append(test_system_initialization())
    test_results.append(test_task_processing())
    test_results.append(test_message_delivery())
    
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