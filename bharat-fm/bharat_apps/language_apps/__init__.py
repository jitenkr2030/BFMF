"""
Language AI Applications

Ready-to-deploy applications for National & Regional Language AI use cases
"""

from .multilingual_chatbot import MultilingualChatbotApp
from .translation_engine import TranslationEngineApp
from .speech_transcription import SpeechTranscriptionApp
from .news_summarizer import NewsSummarizerApp

__all__ = [
    'MultilingualChatbotApp',
    'TranslationEngineApp',
    'SpeechTranscriptionApp',
    'NewsSummarizerApp'
]