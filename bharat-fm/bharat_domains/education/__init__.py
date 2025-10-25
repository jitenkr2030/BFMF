"""
Bharat Education AI Module

Support for Education & Skill Development use cases including:
- AI Teachers (VidyaYantra) for personalized learning
- Automated question generation from educational content
- Exam evaluation and feedback bots
- AI curriculum translation across regional boards
"""

from .models import BharatEdu, VidyaYantra
from .datasets import EducationalDatasets, NCERTData, SWAYAMData, BoardData
from .preprocessors import ContentPreprocessor, QuestionGenerator, ExamEvaluator
from .evaluators import EducationEvaluator, LearningAssessment

__all__ = [
    'BharatEdu',
    'VidyaYantra',
    'EducationalDatasets',
    'NCERTData', 
    'SWAYAMData',
    'BoardData',
    'ContentPreprocessor',
    'QuestionGenerator',
    'ExamEvaluator',
    'EducationEvaluator',
    'LearningAssessment'
]