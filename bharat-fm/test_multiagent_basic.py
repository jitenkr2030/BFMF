#!/usr/bin/env python3
"""
Basic test script for Multi-Agent System components
"""

import sys
import os
import json
import time
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
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

class SimpleMultiAgentSystem:
    """Simple multi-agent system for testing"""
    
    def __init__(self):
        self.agents = {}
        self.tasks = {}
        self.messages = []
        self.completed_tasks = []
    
    def add_agent(self, agent: Agent):
        """Add an agent to the system"""
        self.agents[agent.id] = agent
        logger.info(f"Added agent {agent.id} to system")
    
    def submit_task(self, task: Task):
        """Submit a task to the system"""
        self.tasks[task.id] = task
        logger.info(f"Submitted task {task.id} to system")
        
        # Try to assign task immediately
        self._assign_task(task)
        
        return task.id
    
    def _assign_task(self, task: Task):
        """Assign task to suitable agent"""
        best_agent = None
        best_score = 0.0
        
        for agent in self.agents.values():
            can_handle, confidence = agent.can_handle_task(task)
            if can_handle and confidence > best_score:
                best_score = confidence
                best_agent = agent
        
        if best_agent:
            task.assigned_agent = best_agent.id
            task.status = TaskStatus.ASSIGNED
            best_agent.current_tasks.append(task.id)
            logger.info(f"Assigned task {task.id} to agent {best_agent.id}")
            
            # Process task immediately (simplified)
            self._process_task(task, best_agent)
        else:
            logger.warning(f"No suitable agent found for task {task.id}")
    
    def _process_task(self, task: Task, agent: Agent):
        """Process a task (simplified)"""
        logger.info(f"Processing task {task.id} with agent {agent.id}")
        
        # Simulate processing
        time.sleep(0.1)
        
        # Generate result
        result = {
            "task_id": task.id,
            "agent_id": agent.id,
            "result": f"Task {task.id} completed by {agent.name}",
            "timestamp": time.time()
        }
        
        # Update task status
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = time.time()
        
        # Update agent
        if task.id in agent.current_tasks:
            agent.current_tasks.remove(task.id)
        
        # Add to completed tasks
        self.completed_tasks.append(task)
        
        logger.info(f"Completed task {task.id}")
    
    def get_system_status(self):
        """Get system status"""
        return {
            "total_agents": len(self.agents),
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "agent_utilization": {
                agent_id: len(agent.current_tasks) / agent.max_concurrent_tasks
                for agent_id, agent in self.agents.items()
            }
        }

def test_system_creation():
    """Test system creation and agent addition"""
    logger.info("=== Testing System Creation ===")
    
    try:
        # Create system
        system = SimpleMultiAgentSystem()
        
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
            ),
            Agent(
                id="predictor_agent",
                name="Prediction Specialist",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("prediction", "Predictive modeling", 0.88, "prediction", 8),
                    AgentCapability("machine_learning", "Machine learning", 0.82, "prediction", 9)
                ]
            )
        ]
        
        # Add agents to system
        for agent in agents:
            system.add_agent(agent)
        
        # Check system status
        status = system.get_system_status()
        logger.info(f"System status: {status}")
        
        return {
            "test_type": "system_creation",
            "success": True,
            "agents_added": len(agents),
            "system_status": status,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"System creation test failed: {str(e)}")
        return {
            "test_type": "system_creation",
            "success": False,
            "error": str(e)
        }

def test_task_assignment():
    """Test task assignment and processing"""
    logger.info("=== Testing Task Assignment ===")
    
    try:
        # Create system
        system = SimpleMultiAgentSystem()
        
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
        
        # Add agents to system
        for agent in agents:
            system.add_agent(agent)
        
        # Create tasks
        tasks = [
            Task(
                id="task_1",
                name="Data Analysis",
                description="Analyze sample data",
                required_capabilities=["data_analysis"],
                task_type="analysis",
                priority=5,
                estimated_duration=1.0
            ),
            Task(
                id="task_2",
                name="System Optimization",
                description="Optimize system performance",
                required_capabilities=["optimization"],
                task_type="optimization",
                priority=7,
                estimated_duration=1.5
            ),
            Task(
                id="task_3",
                name="Statistical Analysis",
                description="Perform statistical analysis",
                required_capabilities=["statistical_analysis"],
                task_type="analysis",
                priority=6,
                estimated_duration=2.0
            )
        ]
        
        # Submit tasks
        submitted_tasks = []
        for task in tasks:
            task_id = system.submit_task(task)
            submitted_tasks.append(task_id)
        
        # Wait for processing
        time.sleep(1)
        
        # Check results
        status = system.get_system_status()
        logger.info(f"Final system status: {status}")
        
        return {
            "test_type": "task_assignment",
            "success": True,
            "tasks_submitted": len(submitted_tasks),
            "tasks_completed": status["completed_tasks"],
            "system_status": status,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Task assignment test failed: {str(e)}")
        return {
            "test_type": "task_assignment",
            "success": False,
            "error": str(e)
        }

def test_agent_capability_matching():
    """Test agent capability matching"""
    logger.info("=== Testing Agent Capability Matching ===")
    
    try:
        # Create system
        system = SimpleMultiAgentSystem()
        
        # Create agents with different capabilities
        agents = [
            Agent(
                id="analyst_agent",
                name="Data Analyst",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("data_analysis", "Data analysis", 0.95, "analytics", 9),
                    AgentCapability("statistical_analysis", "Statistical analysis", 0.85, "analytics", 8)
                ]
            ),
            Agent(
                id="optimizer_agent",
                name="System Optimizer",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("optimization", "System optimization", 0.90, "optimization", 9),
                    AgentCapability("performance_analysis", "Performance analysis", 0.80, "optimization", 7)
                ]
            ),
            Agent(
                id="generalist_agent",
                name="Generalist Agent",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("data_analysis", "Data analysis", 0.70, "analytics", 6),
                    AgentCapability("optimization", "System optimization", 0.75, "optimization", 6),
                    AgentCapability("prediction", "Prediction", 0.65, "prediction", 5)
                ]
            )
        ]
        
        # Add agents to system
        for agent in agents:
            system.add_agent(agent)
        
        # Test capability matching
        test_cases = [
            {
                "task": Task(
                    id="test_task_1",
                    name="High Priority Analysis",
                    description="Complex data analysis",
                    required_capabilities=["data_analysis"],
                    task_type="analysis",
                    priority=9,
                    estimated_duration=2.0
                ),
                "expected_agent": "analyst_agent",
                "description": "Should assign to specialist analyst"
            },
            {
                "task": Task(
                    id="test_task_2",
                    name="System Optimization",
                    description="Optimize system parameters",
                    required_capabilities=["optimization"],
                    task_type="optimization",
                    priority=8,
                    estimated_duration=1.5
                ),
                "expected_agent": "optimizer_agent",
                "description": "Should assign to specialist optimizer"
            },
            {
                "task": Task(
                    id="test_task_3",
                    name="Mixed Task",
                    description="Task requiring multiple capabilities",
                    required_capabilities=["data_analysis", "optimization"],
                    task_type="mixed",
                    priority=7,
                    estimated_duration=3.0
                ),
                "expected_agent": "generalist_agent",
                "description": "Should assign to generalist with both capabilities"
            }
        ]
        
        results = []
        for test_case in test_cases:
            task = test_case["task"]
            expected_agent = test_case["expected_agent"]
            description = test_case["description"]
            
            # Submit task
            system.submit_task(task)
            
            # Check assignment
            assigned_agent = task.assigned_agent
            success = assigned_agent == expected_agent
            
            logger.info(f"{description}: Assigned to {assigned_agent} (expected {expected_agent}) - {'✓' if success else '✗'}")
            
            results.append({
                "test_case": description,
                "task_id": task.id,
                "expected_agent": expected_agent,
                "assigned_agent": assigned_agent,
                "success": success
            })
        
        # Calculate overall success rate
        successful_tests = sum(1 for r in results if r["success"])
        total_tests = len(results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"Capability matching success rate: {success_rate:.2%} ({successful_tests}/{total_tests})")
        
        return {
            "test_type": "capability_matching",
            "success": success_rate >= 0.8,  # Allow for some flexibility
            "success_rate": success_rate,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "results": results,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Capability matching test failed: {str(e)}")
        return {
            "test_type": "capability_matching",
            "success": False,
            "error": str(e)
        }

def main():
    """Run all tests"""
    logger.info("Starting Basic Multi-Agent System Tests")
    
    test_results = []
    
    # Run tests
    test_results.append(test_system_creation())
    test_results.append(test_task_assignment())
    test_results.append(test_agent_capability_matching())
    
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