"""
Governance AI Applications

Ready-to-deploy applications for Digital Governance & Public Policy AI use cases
"""

from .policy_drafting import PolicyDraftingApp
from .rti_assistant import RTIAssistantApp
from .grievance_system import GrievanceSystemApp
from .audit_automation import AuditAutomationApp

__all__ = [
    'PolicyDraftingApp',
    'RTIAssistantApp',
    'GrievanceSystemApp',
    'AuditAutomationApp'
]