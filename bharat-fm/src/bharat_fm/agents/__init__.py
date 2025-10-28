"""
Multi-Agent System Module for Bharat-FM
Provides collaborative problem-solving capabilities with multiple AI agents
"""

from .multi_agent_system import (
    AgentRole,
    TaskStatus,
    MessageType,
    AgentCapability,
    Task,
    Message,
    Agent,
    BaseAgent,
    SpecialistAgent,
    CoordinatorAgent,
    CommunicationHandler,
    LearningSystem,
    MultiAgentSystem,
    create_multi_agent_system,
    create_task
)

__all__ = [
    'AgentRole',
    'TaskStatus',
    'MessageType',
    'AgentCapability',
    'Task',
    'Message',
    'Agent',
    'BaseAgent',
    'SpecialistAgent',
    'CoordinatorAgent',
    'CommunicationHandler',
    'LearningSystem',
    'MultiAgentSystem',
    'create_multi_agent_system',
    'create_task'
]