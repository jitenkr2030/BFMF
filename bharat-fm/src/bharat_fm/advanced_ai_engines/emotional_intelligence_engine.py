"""
Emotional Intelligence and Empathy Module for Bharat-FM MLOps Platform

This module implements sophisticated emotional intelligence capabilities that enable the AI to:
- Recognize and understand human emotions
- Respond with appropriate empathy and emotional support
- Adapt communication style based on emotional context
- Build emotional rapport with users
- Handle sensitive emotional situations with care

Author: Advanced AI Team
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json
import time
from datetime import datetime, timedelta
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Emotion(Enum):
    """Primary emotions that can be recognized and expressed"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"
    EXCITEMENT = "excitement"
    ANXIETY = "anxiety"
    GRATITUDE = "gratitude"
    DISAPPOINTMENT = "disappointment"

class EmpathyLevel(Enum):
    """Levels of empathetic response"""
    BASIC = "basic"  # Acknowledge emotion
    MODERATE = "moderate"  # Understand and validate
    DEEP = "deep"  # Fully connect and support
    THERAPEUTIC = "therapeutic"  # Professional emotional support

class CommunicationStyle(Enum):
    """Communication styles based on emotional context"""
    SUPPORTIVE = "supportive"
    ENCOURAGING = "encouraging"
    CALMING = "calming"
    ENERGIZING = "energizing"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"
    DIRECT = "direct"

@dataclass
class EmotionalState:
    """Represents the emotional state of a user"""
    user_id: str
    primary_emotion: Emotion
    secondary_emotions: List[Emotion]
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    duration: timedelta = field(default=timedelta(0))

@dataclass
class EmpathyResponse:
    """Represents an empathetic response"""
    response_id: str
    target_emotion: Emotion
    empathy_level: EmpathyLevel
    response_text: str
    tone: str
    non_verbal_cues: List[str]
    follow_up_suggestions: List[str]
    confidence: float
    strategy: str

@dataclass
class EmotionalProfile:
    """Represents the emotional profile of a user"""
    user_id: str
    typical_emotions: List[Emotion]
    emotional_triggers: Dict[str, List[Emotion]]
    coping_strategies: List[str]
    communication_preferences: CommunicationStyle
    empathy_sensitivity: float  # 0.0 to 1.0
    emotional_patterns: Dict[str, Any]
    last_updated: datetime

class EmotionalIntelligenceEngine:
    """
    Advanced emotional intelligence and empathy engine
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Emotional analysis models
        self.emotion_classifier = None
        self.empathy_generator = None
        self.sentiment_analyzer = None
        
        # Data structures
        self.emotional_states = defaultdict(list)  # user_id -> emotional states
        self.emotional_profiles = {}  # user_id -> emotional profile
        self.empathy_responses = defaultdict(list)  # emotion -> response templates
        self.emotional_history = defaultdict(deque)  # user_id -> emotional history
        
        # Emotional intelligence parameters
        self.empathy_threshold = self.config.get('empathy_threshold', 0.6)
        self.emotion_sensitivity = self.config.get('emotion_sensitivity', 0.7)
        self.response_adaptation_rate = self.config.get('response_adaptation_rate', 0.1)
        
        # Initialize components
        self._initialize_emotional_models()
        self._load_empathy_templates()
        self._initialize_emotional_analysis()
        
        logger.info("Emotional Intelligence Engine initialized successfully")
    
    def _initialize_emotional_models(self):
        """Initialize emotional analysis models"""
        try:
            # Load pre-trained emotion classification model
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            self.emotion_classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.emotion_classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.emotion_classifier.eval()
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.emotion_classifier.to(self.device)
            
            logger.info("Emotional analysis models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load emotional models: {e}")
            # Fallback to rule-based emotion detection
            self.emotion_classifier = None
    
    def _load_empathy_templates(self):
        """Load empathy response templates"""
        # Basic empathy templates for different emotions
        self.empathy_templates = {
            Emotion.JOY: [
                {
                    'level': EmpathyLevel.BASIC,
                    'template': "That's wonderful! I'm happy to hear you're feeling {emotion}.",
                    'tone': 'cheerful',
                    'follow_up': ['Tell me more about what made you happy', 'How can we maintain this positive feeling?']
                },
                {
                    'level': EmpathyLevel.MODERATE,
                    'template': "I can feel your {emotion}! It's great to see you experiencing such positive emotions. What's contributing to this wonderful feeling?",
                    'tone': 'enthusiastic',
                    'follow_up': ['Share more about this experience', 'Let\'s celebrate this moment together']
                }
            ],
            Emotion.SADNESS: [
                {
                    'level': EmpathyLevel.BASIC,
                    'template': "I understand you're feeling {emotion}. That must be difficult.",
                    'tone': 'gentle',
                    'follow_up': ['Would you like to talk about it?', 'I\'m here to listen']
                },
                {
                    'level': EmpathyLevel.MODERATE,
                    'template': "I can sense your {emotion}, and I want you to know that your feelings are completely valid. It's okay to feel this way.",
                    'tone': 'supportive',
                    'follow_up': ['Take your time to express what\'s on your mind', 'What would be most helpful right now?']
                },
                {
                    'level': EmpathyLevel.DEEP,
                    'template': "Your {emotion} is palpable, and I'm here to walk alongside you through this difficult time. Your emotions matter, and I'm committed to supporting you.",
                    'tone': 'compassionate',
                    'follow_up': ['Let\'s explore ways to cope together', 'Remember, you\'re not alone in this']
                }
            ],
            Emotion.ANGER: [
                {
                    'level': EmpathyLevel.BASIC,
                    'template': "I can see you're feeling {emotion}. That's a valid emotion.",
                    'tone': 'calm',
                    'follow_up': ['Would you like to discuss what triggered this?', 'Let\'s find a constructive way to address this']
                },
                {
                    'level': EmpathyLevel.MODERATE,
                    'template': "I understand your {emotion}, and it makes sense given the situation. Let's work together to understand and address what's causing these feelings.",
                    'tone': 'understanding',
                    'follow_up': ['Take a deep breath with me', 'What would help you feel more in control?']
                }
            ],
            Emotion.FEAR: [
                {
                    'level': EmpathyLevel.BASIC,
                    'template': "It's okay to feel {emotion}. I'm here with you.",
                    'tone': 'reassuring',
                    'follow_up': ['What specifically are you afraid of?', 'Let\'s break this down together']
                },
                {
                    'level': EmpathyLevel.MODERATE,
                    'template': "Your {emotion} is completely understandable, and I want you to know you're safe here. We can face this together.",
                    'tone': 'comforting',
                    'follow_up': ['Let\'s identify what we can control', 'I\'ll help you develop a plan']
                }
            ],
            Emotion.ANXIETY: [
                {
                    'level': EmpathyLevel.BASIC,
                    'template': "I can sense your {emotion}. Let's take this one step at a time.",
                    'tone': 'calming',
                    'follow_up': ['Would you like to try a breathing exercise?', 'What\'s worrying you most right now?']
                },
                {
                    'level': EmpathyLevel.MODERATE,
                    'template': "I understand your {emotion}, and I want to help you find some calm. Let's work together to manage these feelings.",
                    'tone': 'soothing',
                    'follow_up': ['Let\'s practice mindfulness together', 'What would help you feel more grounded?']
                }
            ],
            Emotion.NEUTRAL: [
                {
                    'level': EmpathyLevel.BASIC,
                    'template': "I'm here to support you. How are you feeling right now?",
                    'tone': 'neutral',
                    'follow_up': ['Is there anything on your mind?', 'How can I assist you today?']
                }
            ]
        }
        
        # Add templates for other emotions
        for emotion in Emotion:
            if emotion not in self.empathy_templates:
                self.empathy_templates[emotion] = [
                    {
                        'level': EmpathyLevel.BASIC,
                        'template': f"I understand you're feeling {{emotion}}. I'm here to support you.",
                        'tone': 'supportive',
                        'follow_up': ['Would you like to talk about it?', 'How can I help?']
                    }
                ]
    
    def _initialize_emotional_analysis(self):
        """Initialize emotional analysis components"""
        # Emotion keywords for rule-based detection
        self.emotion_keywords = {
            Emotion.JOY: ['happy', 'joy', 'excited', 'delighted', 'cheerful', 'pleased', 'glad', 'wonderful', 'amazing', 'great'],
            Emotion.SADNESS: ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'heartbroken', 'disappointed', 'grief', 'sorrow'],
            Emotion.ANGER: ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'rage', 'upset', 'livid'],
            Emotion.FEAR: ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 'nervous', 'panicked'],
            Emotion.SURPRISE: ['surprised', 'shocked', 'amazed', 'astonished', 'startled', 'stunned'],
            Emotion.DISGUST: ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled'],
            Emotion.ANXIETY: ['anxious', 'worried', 'nervous', 'stressed', 'overwhelmed', 'tense', 'uneasy'],
            Emotion.CONFUSION: ['confused', 'uncertain', 'unsure', 'puzzled', 'bewildered', 'lost'],
            Emotion.FRUSTRATION: ['frustrated', 'stuck', 'annoyed', 'impatient', 'irritated'],
            Emotion.EXCITEMENT: ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped', 'jazzed'],
            Emotion.GRATITUDE: ['grateful', 'thankful', 'appreciative', 'blessed', 'indebted'],
            Emotion.DISAPPOINTMENT: ['disappointed', 'let down', 'unsatisfied', 'unfulfilled']
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'very': 1.5, 'extremely': 1.7, 'incredibly': 1.8, 'absolutely': 1.6,
            'quite': 1.2, 'rather': 1.1, 'somewhat': 0.8, 'slightly': 0.6,
            'a bit': 0.5, 'kind of': 0.7, 'pretty': 1.3
        }
    
    def analyze_emotion(self, text: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> EmotionalState:
        """
        Analyze emotion from text input
        
        Args:
            text: Text to analyze for emotion
            user_id: User identifier
            context: Additional context
            
        Returns:
            EmotionalState: Detected emotional state
        """
        # Use ML model if available, otherwise fall back to rule-based
        if self.emotion_classifier:
            emotion_data = self._analyze_emotion_ml(text)
        else:
            emotion_data = self._analyze_emotion_rule_based(text)
        
        # Create emotional state
        emotional_state = EmotionalState(
            user_id=user_id,
            primary_emotion=emotion_data['primary_emotion'],
            secondary_emotions=emotion_data['secondary_emotions'],
            intensity=emotion_data['intensity'],
            confidence=emotion_data['confidence'],
            timestamp=datetime.now(),
            context=context or {},
            triggers=emotion_data.get('triggers', [])
        )
        
        # Store emotional state
        self.emotional_states[user_id].append(emotional_state)
        self.emotional_history[user_id].append(emotional_state)
        
        # Update user's emotional profile
        self._update_emotional_profile(user_id, emotional_state)
        
        logger.debug(f"Analyzed emotion for user {user_id}: {emotional_state.primary_emotion.value}")
        
        return emotional_state
    
    def _analyze_emotion_ml(self, text: str) -> Dict[str, Any]:
        """Analyze emotion using machine learning model"""
        try:
            # Tokenize and classify
            inputs = self.emotion_classifier_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.emotion_classifier(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()[0]
            
            # Map predictions to emotions
            emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            emotion_scores = {emotion: score for emotion, score in zip(emotion_labels, predictions)}
            
            # Find primary and secondary emotions
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            primary_emotion = Emotion(sorted_emotions[0][0])
            secondary_emotions = [Emotion(emotion) for emotion, score in sorted_emotions[1:3] if score > 0.1]
            
            intensity = sorted_emotions[0][1]
            confidence = intensity
            
            return {
                'primary_emotion': primary_emotion,
                'secondary_emotions': secondary_emotions,
                'intensity': intensity,
                'confidence': confidence,
                'triggers': self._extract_triggers(text)
            }
            
        except Exception as e:
            logger.error(f"Error in ML emotion analysis: {e}")
            return self._analyze_emotion_rule_based(text)
    
    def _analyze_emotion_rule_based(self, text: str) -> Dict[str, Any]:
        """Analyze emotion using rule-based approach"""
        text_lower = text.lower()
        emotion_scores = defaultdict(float)
        
        # Check for emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1
        
        # Apply intensity modifiers
        for modifier, multiplier in self.intensity_modifiers.items():
            if modifier in text_lower:
                for emotion in emotion_scores:
                    emotion_scores[emotion] *= multiplier
        
        # Normalize scores
        if emotion_scores:
            max_score = max(emotion_scores.values())
            if max_score > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= max_score
        
        # Find primary and secondary emotions
        if emotion_scores:
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            primary_emotion = sorted_emotions[0][0]
            secondary_emotions = [emotion for emotion, score in sorted_emotions[1:3] if score > 0.3]
            intensity = sorted_emotions[0][1]
            confidence = min(intensity * 0.8, 0.9)  # Rule-based is less confident
        else:
            primary_emotion = Emotion.NEUTRAL
            secondary_emotions = []
            intensity = 0.5
            confidence = 0.6
        
        return {
            'primary_emotion': primary_emotion,
            'secondary_emotions': secondary_emotions,
            'intensity': intensity,
            'confidence': confidence,
            'triggers': self._extract_triggers(text)
        }
    
    def _extract_triggers(self, text: str) -> List[str]:
        """Extract potential emotional triggers from text"""
        triggers = []
        
        # Common trigger patterns
        trigger_patterns = [
            r'because\s+(.+)',
            r'since\s+(.+)',
            r'due to\s+(.+)',
            r'when\s+(.+)',
            r'after\s+(.+)',
            r'about\s+(.+)'
        ]
        
        for pattern in trigger_patterns:
            matches = re.findall(pattern, text.lower())
            triggers.extend(matches)
        
        return triggers[:3]  # Return top 3 triggers
    
    def generate_empathy_response(self, emotional_state: EmotionalState, 
                                empathy_level: Optional[EmpathyLevel] = None) -> EmpathyResponse:
        """
        Generate empathetic response based on emotional state
        
        Args:
            emotional_state: User's emotional state
            empathy_level: Desired level of empathy
            
        Returns:
            EmpathyResponse: Generated empathetic response
        """
        # Determine empathy level
        if empathy_level is None:
            empathy_level = self._determine_empathy_level(emotional_state)
        
        # Select appropriate template
        templates = self.empathy_templates.get(emotional_state.primary_emotion, [])
        
        # Filter templates by empathy level
        suitable_templates = [t for t in templates if t['level'] == empathy_level]
        
        if not suitable_templates:
            suitable_templates = templates  # Use any available template
        
        if not suitable_templates:
            suitable_templates = self.empathy_templates[Emotion.NEUTRAL]
        
        # Select template
        template = suitable_templates[0]
        
        # Generate response
        response_text = template['template'].format(
            emotion=emotional_state.primary_emotion.value
        )
        
        # Personalize response based on user profile
        if emotional_state.user_id in self.emotional_profiles:
            response_text = self._personalize_response(
                response_text, 
                emotional_state.user_id,
                emotional_state
            )
        
        # Generate response ID
        response_id = f"empathy_{int(time.time())}_{hashlib.md5(response_text.encode()).hexdigest()[:8]}"
        
        # Create empathy response
        empathy_response = EmpathyResponse(
            response_id=response_id,
            target_emotion=emotional_state.primary_emotion,
            empathy_level=empathy_level,
            response_text=response_text,
            tone=template['tone'],
            non_verbal_cues=self._generate_non_verbal_cues(emotional_state),
            follow_up_suggestions=template['follow_up'],
            confidence=emotional_state.confidence,
            strategy=f"empathy_{empathy_level.value}"
        )
        
        return empathy_response
    
    def _determine_empathy_level(self, emotional_state: EmotionalState) -> EmpathyLevel:
        """Determine appropriate empathy level based on emotional state"""
        # High intensity emotions require deeper empathy
        if emotional_state.intensity > 0.8:
            return EmpathyLevel.DEEP
        
        # Negative emotions typically need more empathy
        negative_emotions = [Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR, Emotion.ANXIETY]
        if emotional_state.primary_emotion in negative_emotions:
            return EmpathyLevel.MODERATE if emotional_state.intensity < 0.7 else EmpathyLevel.DEEP
        
        # Consider user's empathy sensitivity
        if emotional_state.user_id in self.emotional_profiles:
            profile = self.emotional_profiles[emotional_state.user_id]
            if profile.empathy_sensitivity > 0.8:
                return EmpathyLevel.DEEP
            elif profile.empathy_sensitivity > 0.6:
                return EmpathyLevel.MODERATE
        
        # Default to basic empathy
        return EmpathyLevel.BASIC
    
    def _personalize_response(self, response_text: str, user_id: str, 
                            emotional_state: EmotionalState) -> str:
        """Personalize empathy response based on user profile"""
        profile = self.emotional_profiles[user_id]
        
        # Adapt to communication preferences
        if profile.communication_preferences == CommunicationStyle.CASUAL:
            response_text = response_text.replace("I understand", "I get")
            response_text = response_text.replace("completely valid", "totally valid")
        
        elif profile.communication_preferences == CommunicationStyle.PROFESSIONAL:
            response_text = response_text.replace("I'm here", "I am available")
            response_text = response_text.replace("Let's", "I suggest we")
        
        # Add personalization based on emotional patterns
        if 'calming_phrases' in profile.emotional_patterns:
            if emotional_state.primary_emotion in [Emotion.ANXIETY, Emotion.FEAR]:
                calming_phrase = np.random.choice(profile.emotional_patterns['calming_phrases'])
                response_text += f" {calming_phrase}"
        
        return response_text
    
    def _generate_non_verbal_cues(self, emotional_state: EmotionalState) -> List[str]:
        """Generate non-verbal communication cues"""
        cues = []
        
        # Base cues on emotion and intensity
        if emotional_state.primary_emotion == Emotion.JOY:
            cues.extend(["warm smile", "upbeat tone", "open posture"])
        elif emotional_state.primary_emotion == Emotion.SADNESS:
            cues.extend(["gentle tone", "soft voice", "compassionate expression"])
        elif emotional_state.primary_emotion == Emotion.ANGER:
            cues.extend(["calm demeanor", "steady voice", "patient expression"])
        elif emotional_state.primary_emotion == Emotion.FEAR:
            cues.extend(["reassuring presence", "soothing voice", "safe space"])
        elif emotional_state.primary_emotion == Emotion.ANXIETY:
            cues.extend(["calming presence", "measured pace", "grounded energy"])
        
        # Adjust based on intensity
        if emotional_state.intensity > 0.7:
            cues.append("heightened attentiveness")
        elif emotional_state.intensity < 0.3:
            cues.append("subtle support")
        
        return cues[:3]  # Return top 3 cues
    
    def adapt_communication_style(self, user_id: str, emotional_state: EmotionalState) -> CommunicationStyle:
        """
        Adapt communication style based on emotional state and user preferences
        
        Args:
            user_id: User identifier
            emotional_state: Current emotional state
            
        Returns:
            CommunicationStyle: Recommended communication style
        """
        # Get user profile
        profile = self.emotional_profiles.get(user_id)
        
        # Base style on emotional state
        if emotional_state.primary_emotion in [Emotion.JOY, Emotion.EXCITEMENT]:
            base_style = CommunicationStyle.ENERGIZING
        elif emotional_state.primary_emotion in [Emotion.SADNESS, Emotion.FEAR, Emotion.ANXIETY]:
            base_style = CommunicationStyle.SUPPORTIVE
        elif emotional_state.primary_emotion in [Emotion.ANGER, Emotion.FRUSTRATION]:
            base_style = CommunicationStyle.CALMING
        else:
            base_style = CommunicationStyle.NEUTRAL
        
        # Override with user preferences if available
        if profile and profile.communication_preferences != CommunicationStyle.NEUTRAL:
            # Blend user preference with emotional needs
            if emotional_state.intensity > 0.7:
                # High intensity: prioritize emotional needs
                return base_style
            else:
                # Lower intensity: use user preference
                return profile.communication_preferences
        
        return base_style
    
    def build_emotional_rapport(self, user_id: str, interaction_history: List[Dict[str, Any]]) -> float:
        """
        Build and measure emotional rapport with user
        
        Args:
            user_id: User identifier
            interaction_history: History of interactions
            
        Returns:
            float: Rapport score (0.0 to 1.0)
        """
        if user_id not in self.emotional_history:
            return 0.0
        
        # Calculate rapport based on emotional alignment
        rapport_score = 0.0
        total_interactions = 0
        
        for interaction in interaction_history:
            if 'user_emotion' in interaction and 'system_response' in interaction:
                user_emotion = interaction['user_emotion']
                system_empathy = interaction.get('system_empathy_level', EmpathyLevel.BASIC)
                
                # Score based on empathy appropriateness
                empathy_scores = {
                    EmpathyLevel.BASIC: 0.5,
                    EmpathyLevel.MODERATE: 0.7,
                    EmpathyLevel.DEEP: 0.9,
                    EmpathyLevel.THERAPEUTIC: 1.0
                }
                
                rapport_score += empathy_scores.get(system_empathy, 0.5)
                total_interactions += 1
        
        # Normalize score
        if total_interactions > 0:
            rapport_score /= total_interactions
        
        # Consider emotional consistency
        emotional_consistency = self._calculate_emotional_consistency(user_id)
        rapport_score = (rapport_score + emotional_consistency) / 2
        
        return min(rapport_score, 1.0)
    
    def _calculate_emotional_consistency(self, user_id: str) -> float:
        """Calculate emotional consistency for a user"""
        if user_id not in self.emotional_history or len(self.emotional_history[user_id]) < 3:
            return 0.5
        
        # Analyze emotional patterns
        emotions = [state.primary_emotion for state in self.emotional_history[user_id][-10:]]
        
        # Calculate consistency (how predictable the patterns are)
        emotion_counts = defaultdict(int)
        for emotion in emotions:
            emotion_counts[emotion] += 1
        
        # Consistency based on dominant emotion frequency
        if emotion_counts:
            max_count = max(emotion_counts.values())
            consistency = max_count / len(emotions)
        else:
            consistency = 0.5
        
        return consistency
    
    def handle_emotional_crisis(self, user_id: str, emotional_state: EmotionalState) -> Dict[str, Any]:
        """
        Handle emotional crisis situations
        
        Args:
            user_id: User identifier
            emotional_state: Current emotional state
            
        Returns:
            Dict: Crisis handling strategy
        """
        crisis_emotions = [Emotion.FEAR, Emotion.ANXIETY, Emotion.SADNESS, Emotion.ANGER]
        
        if (emotional_state.primary_emotion in crisis_emotions and 
            emotional_state.intensity > 0.8):
            
            # Generate crisis response
            crisis_response = {
                'priority': 'high',
                'immediate_actions': [
                    'Validate emotions',
                    'Ensure safety',
                    'Provide support resources',
                    'Offer professional help if needed'
                ],
                'communication_strategy': 'therapeutic',
                'follow_up_required': True,
                'support_resources': self._get_support_resources(emotional_state.primary_emotion),
                'de_escalation_techniques': self._get_de_escalation_techniques(emotional_state.primary_emotion)
            }
            
            # Log crisis event
            self._log_crisis_event(user_id, emotional_state, crisis_response)
            
            return crisis_response
        
        return {'priority': 'normal', 'immediate_actions': []}
    
    def _get_support_resources(self, emotion: Emotion) -> List[str]:
        """Get support resources for specific emotions"""
        resources = {
            Emotion.FEAR: [
                "Grounding exercises",
                "Breathing techniques",
                "Safety planning",
                "Professional counseling"
            ],
            Emotion.ANXIETY: [
                "Anxiety management strategies",
                "Mindfulness exercises",
                "Stress reduction techniques",
                "Therapy options"
            ],
            Emotion.SADNESS: [
                "Emotional support resources",
                "Depression screening",
                "Support groups",
                "Mental health professionals"
            ],
            Emotion.ANGER: [
                "Anger management techniques",
                "Conflict resolution strategies",
                "Stress management",
                "Counseling services"
            ]
        }
        
        return resources.get(emotion, ["General support resources"])
    
    def _get_de_escalation_techniques(self, emotion: Emotion) -> List[str]:
        """Get de-escalation techniques for specific emotions"""
        techniques = {
            Emotion.FEAR: [
                "Create safe environment",
                "Use calm, reassuring language",
                "Validate feelings",
                "Offer choices and control"
            ],
            Emotion.ANXIETY: [
                "Guide through breathing exercises",
                "Use structured, clear communication",
                "Break down overwhelming thoughts",
                "Focus on present moment"
            ],
            Emotion.SADNESS: [
                "Provide emotional validation",
                "Offer compassionate presence",
                "Use gentle, supportive language",
                "Allow emotional expression"
            ],
            Emotion.ANGER: [
                "Maintain calm demeanor",
                "Use non-confrontational language",
                "Acknowledge feelings without judgment",
                "Provide space and time"
            ]
        }
        
        return techniques.get(emotion, ["General de-escalation techniques"])
    
    def _log_crisis_event(self, user_id: str, emotional_state: EmotionalState, response: Dict[str, Any]):
        """Log emotional crisis event"""
        crisis_log = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'emotion': emotional_state.primary_emotion.value,
            'intensity': emotional_state.intensity,
            'response_priority': response['priority'],
            'actions_taken': response['immediate_actions']
        }
        
        # Store crisis log (in a real system, this would go to a database)
        crisis_log_file = f"crisis_log_{user_id}.json"
        
        try:
            if Path(crisis_log_file).exists():
                with open(crisis_log_file, 'r') as f:
                    existing_logs = json.load(f)
            else:
                existing_logs = []
            
            existing_logs.append(crisis_log)
            
            with open(crisis_log_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging crisis event: {e}")
    
    def _update_emotional_profile(self, user_id: str, emotional_state: EmotionalState):
        """Update user's emotional profile"""
        if user_id not in self.emotional_profiles:
            # Create new profile
            self.emotional_profiles[user_id] = EmotionalProfile(
                user_id=user_id,
                typical_emotions=[emotional_state.primary_emotion],
                emotional_triggers={},
                coping_strategies=[],
                communication_preferences=CommunicationStyle.NEUTRAL,
                empathy_sensitivity=0.5,
                emotional_patterns={},
                last_updated=datetime.now()
            )
        
        profile = self.emotional_profiles[user_id]
        
        # Update typical emotions
        if emotional_state.primary_emotion not in profile.typical_emotions:
            profile.typical_emotions.append(emotional_state.primary_emotion)
        
        # Update emotional triggers
        for trigger in emotional_state.triggers:
            if trigger not in profile.emotional_triggers:
                profile.emotional_triggers[trigger] = []
            if emotional_state.primary_emotion not in profile.emotional_triggers[trigger]:
                profile.emotional_triggers[trigger].append(emotional_state.primary_emotion)
        
        # Update empathy sensitivity based on response patterns
        if hasattr(self, 'response_history'):
            recent_responses = [r for r in self.response_history.get(user_id, [])[-10:]]
            if recent_responses:
                avg_empathy = np.mean([r.get('empathy_score', 0.5) for r in recent_responses])
                profile.empathy_sensitivity = avg_empathy
        
        # Update last modified
        profile.last_updated = datetime.now()
    
    def get_emotional_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive emotional insights for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict: Emotional insights and analysis
        """
        if user_id not in self.emotional_profiles:
            return {'error': 'No emotional profile found for user'}
        
        profile = self.emotional_profiles[user_id]
        emotional_history = list(self.emotional_history.get(user_id, []))
        
        insights = {
            'user_id': user_id,
            'typical_emotions': [e.value for e in profile.typical_emotions],
            'emotional_triggers': profile.emotional_triggers,
            'communication_preferences': profile.communication_preferences.value,
            'empathy_sensitivity': profile.empathy_sensitivity,
            'emotional_patterns': profile.emotional_patterns,
            'recent_emotional_trends': self._analyze_emotional_trends(emotional_history),
            'emotional_consistency': self._calculate_emotional_consistency(user_id),
            'rapport_score': self.build_emotional_rapport(user_id, []),
            'recommendations': self._generate_emotional_recommendations(profile, emotional_history)
        }
        
        return insights
    
    def _analyze_emotional_trends(self, emotional_history: List[EmotionalState]) -> Dict[str, Any]:
        """Analyze emotional trends over time"""
        if len(emotional_history) < 3:
            return {'insufficient_data': True}
        
        # Group emotions by time periods
        recent_emotions = emotional_history[-10:]  # Last 10 states
        emotion_counts = defaultdict(int)
        intensity_trend = []
        
        for state in recent_emotions:
            emotion_counts[state.primary_emotion] += 1
            intensity_trend.append(state.intensity)
        
        # Calculate trends
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0].value
        intensity_trend_direction = "increasing" if intensity_trend[-1] > intensity_trend[0] else "decreasing"
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': {e.value: c for e, c in emotion_counts.items()},
            'intensity_trend': intensity_trend_direction,
            'average_intensity': np.mean(intensity_trend),
            'emotional_volatility': np.std(intensity_trend)
        }
    
    def _generate_emotional_recommendations(self, profile: EmotionalProfile, 
                                           emotional_history: List[EmotionalState]) -> List[str]:
        """Generate recommendations based on emotional profile"""
        recommendations = []
        
        # Analyze recent emotional patterns
        if emotional_history:
            recent_emotions = [state.primary_emotion for state in emotional_history[-5:]]
            
            # Check for concerning patterns
            negative_emotions = [Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR, Emotion.ANXIETY]
            negative_count = sum(1 for e in recent_emotions if e in negative_emotions)
            
            if negative_count >= 3:
                recommendations.append("Consider seeking additional emotional support")
                recommendations.append("Practice stress-reduction techniques regularly")
            
            # Check for emotional volatility
            if len(emotional_history) >= 5:
                recent_intensities = [state.intensity for state in emotional_history[-5:]]
                if np.std(recent_intensities) > 0.3:
                    recommendations.append("Focus on emotional regulation strategies")
        
        # Communication recommendations
        if profile.empathy_sensitivity > 0.8:
            recommendations.append("Continue using deep empathy in interactions")
        elif profile.empathy_sensitivity < 0.4:
            recommendations.append("Gradually increase empathy levels in responses")
        
        # Trigger-based recommendations
        if profile.emotional_triggers:
            common_triggers = list(profile.emotional_triggers.keys())[:3]
            recommendations.append(f"Be mindful of common triggers: {', '.join(common_triggers)}")
        
        return recommendations
    
    def save_emotional_data(self, filepath: str):
        """Save emotional data to file"""
        try:
            data = {
                'emotional_profiles': {
                    user_id: {
                        'typical_emotions': [e.value for e in profile.typical_emotions],
                        'emotional_triggers': profile.emotional_triggers,
                        'coping_strategies': profile.coping_strategies,
                        'communication_preferences': profile.communication_preferences.value,
                        'empathy_sensitivity': profile.empathy_sensitivity,
                        'emotional_patterns': profile.emotional_patterns,
                        'last_updated': profile.last_updated.isoformat()
                    }
                    for user_id, profile in self.emotional_profiles.items()
                },
                'empathy_templates': {
                    emotion.value: [
                        {
                            'level': template['level'].value,
                            'template': template['template'],
                            'tone': template['tone'],
                            'follow_up': template['follow_up']
                        }
                        for template in templates
                    ]
                    for emotion, templates in self.empathy_templates.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Emotional data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving emotional data: {e}")
    
    def load_emotional_data(self, filepath: str):
        """Load emotional data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load emotional profiles
            self.emotional_profiles = {}
            for user_id, profile_data in data['emotional_profiles'].items():
                self.emotional_profiles[user_id] = EmotionalProfile(
                    user_id=user_id,
                    typical_emotions=[Emotion(e) for e in profile_data['typical_emotions']],
                    emotional_triggers=profile_data['emotional_triggers'],
                    coping_strategies=profile_data['coping_strategies'],
                    communication_preferences=CommunicationStyle(profile_data['communication_preferences']),
                    empathy_sensitivity=profile_data['empathy_sensitivity'],
                    emotional_patterns=profile_data['emotional_patterns'],
                    last_updated=datetime.fromisoformat(profile_data['last_updated'])
                )
            
            # Load empathy templates (merge with existing)
            for emotion_value, templates in data['empathy_templates'].items():
                emotion = Emotion(emotion_value)
                if emotion not in self.empathy_templates:
                    self.empathy_templates[emotion] = []
                
                for template_data in templates:
                    self.empathy_templates[emotion].append({
                        'level': EmpathyLevel(template_data['level']),
                        'template': template_data['template'],
                        'tone': template_data['tone'],
                        'follow_up': template_data['follow_up']
                    })
            
            logger.info(f"Emotional data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading emotional data: {e}")

# Example usage and demonstration
def demonstrate_emotional_intelligence():
    """Demonstrate the emotional intelligence capabilities"""
    print("=== Emotional Intelligence Engine Demonstration ===")
    
    # Initialize emotional intelligence engine
    ei_engine = EmotionalIntelligenceEngine()
    
    # Test texts with different emotions
    test_texts = [
        ("I'm so happy today! Everything is going perfectly.", "user_1"),
        ("I feel really sad about what happened yesterday.", "user_2"),
        ("I'm so angry right now, I can't believe this happened!", "user_3"),
        ("I'm really worried about the presentation tomorrow.", "user_4"),
        ("I'm feeling pretty neutral about the whole situation.", "user_5")
    ]
    
    print("\nAnalyzing emotions from text...")
    
    for text, user_id in test_texts:
        print(f"\nText: '{text}'")
        print(f"User: {user_id}")
        print("-" * 50)
        
        # Analyze emotion
        emotional_state = ei_engine.analyze_emotion(text, user_id)
        
        print(f"Primary Emotion: {emotional_state.primary_emotion.value}")
        print(f"Secondary Emotions: {[e.value for e in emotional_state.secondary_emotions]}")
        print(f"Intensity: {emotional_state.intensity:.3f}")
        print(f"Confidence: {emotional_state.confidence:.3f}")
        print(f"Triggers: {emotional_state.triggers}")
        
        # Generate empathy response
        empathy_response = ei_engine.generate_empathy_response(emotional_state)
        
        print(f"\nEmpathy Response:")
        print(f"  Text: {empathy_response.response_text}")
        print(f"  Level: {empathy_response.empathy_level.value}")
        print(f"  Tone: {empathy_response.tone}")
        print(f"  Non-verbal cues: {', '.join(empathy_response.non_verbal_cues)}")
        
        # Adapt communication style
        comm_style = ei_engine.adapt_communication_style(user_id, emotional_state)
        print(f"  Recommended communication style: {comm_style.value}")
    
    # Test emotional crisis handling
    print("\n=== Crisis Handling Test ===")
    crisis_state = EmotionalState(
        user_id="user_6",
        primary_emotion=Emotion.ANXIETY,
        secondary_emotions=[Emotion.FEAR],
        intensity=0.9,
        confidence=0.8,
        timestamp=datetime.now(),
        context={'situation': 'panic_attack'}
    )
    
    crisis_response = ei_engine.handle_emotional_crisis("user_6", crisis_state)
    print(f"Crisis Priority: {crisis_response['priority']}")
    print(f"Immediate Actions: {', '.join(crisis_response['immediate_actions'])}")
    print(f"Support Resources: {', '.join(crisis_response['support_resources'])}")
    
    # Get emotional insights
    print("\n=== Emotional Insights ===")
    insights = ei_engine.get_emotional_insights("user_1")
    print(f"Typical Emotions: {insights.get('typical_emotions', [])}")
    print(f"Communication Preferences: {insights.get('communication_preferences', 'neutral')}")
    print(f"Empathy Sensitivity: {insights.get('empathy_sensitivity', 0.5):.3f}")
    print(f"Emotional Consistency: {insights.get('emotional_consistency', 0.5):.3f}")
    print(f"Recommendations: {insights.get('recommendations', [])}")
    
    # Save emotional data
    ei_engine.save_emotional_data('emotional_data.json')
    print("\nEmotional data saved successfully!")

if __name__ == "__main__":
    import hashlib
    demonstrate_emotional_intelligence()