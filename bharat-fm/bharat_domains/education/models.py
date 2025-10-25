"""
Domain-specific models for Education AI use cases
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from bharat_model.modeling_llama import BharatLlamaForCausalLM


class BharatEdu(BharatLlamaForCausalLM):
    """
    BharatEdu: AI model for educational content and learning
    
    Fine-tuned on educational datasets including:
    - NCERT textbooks and materials
    - IGNOU course content
    - SWAYAM MOOC content
    - State board curriculum
    - Educational standards and frameworks
    
    Capabilities:
    - Educational content generation
    - Curriculum alignment analysis
    - Learning objective creation
    - Educational resource recommendation
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Specialized layers for education tasks
        self.subject_classifier = nn.Linear(config.hidden_size, config.num_subjects)
        self.grade_level_classifier = nn.Linear(config.hidden_size, config.num_grade_levels)
        self.learning_objective_generator = nn.Linear(config.hidden_size, config.learning_objective_dim)
        self.difficulty_assessor = nn.Linear(config.hidden_size, config.num_difficulty_levels)
        
        # Educational content understanding
        self.curriculum_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size
            ),
            num_layers=4
        )
        
        # Learning style adaptation
        self.learning_style_adapter = nn.ModuleDict({
            style: nn.Linear(config.hidden_size, config.hidden_size)
            for style in ['visual', 'auditory', 'kinesthetic', 'reading']
        })
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        learning_style: Optional[str] = None,
        **kwargs
    ):
        """
        Forward pass with education-specific processing
        
        Args:
            input_ids: Token input ids
            attention_mask: Attention mask
            task_type: Type of education task (content, assessment, curriculum)
            learning_style: Learning style for content adaptation
        """
        # Base model forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply education-specific processing based on task type
        if task_type == 'content':
            subject_logits = self.subject_classifier(hidden_states.mean(dim=1))
            grade_logits = self.grade_level_classifier(hidden_states.mean(dim=1))
            outputs.subject_logits = subject_logits
            outputs.grade_logits = grade_logits
            
        elif task_type == 'assessment':
            difficulty_logits = self.difficulty_assessor(hidden_states.mean(dim=1))
            outputs.difficulty_logits = difficulty_logits
            
        elif task_type == 'curriculum':
            curriculum_encoded = self.curriculum_encoder(hidden_states)
            outputs.curriculum_hidden_states = curriculum_encoded
        
        # Apply learning style adaptation
        if learning_style and learning_style in self.learning_style_adapter:
            adapted_states = self.learning_style_adapter[learning_style](hidden_states)
            outputs.adapted_hidden_states = adapted_states
        
        return outputs
    
    def generate_educational_content(
        self,
        topic: str,
        subject: str,
        grade_level: str,
        content_type: str = "explanation",
        learning_style: str = "visual",
        max_length: int = 1024
    ) -> str:
        """
        Generate educational content tailored to specific requirements
        
        Args:
            topic: Topic to cover
            subject: Subject area
            grade_level: Target grade level
            content_type: Type of content (explanation, example, exercise, etc.)
            learning_style: Learning style adaptation
            max_length: Maximum content length
        """
        prompt = f"""
        Generate educational content with the following specifications:
        
        Topic: {topic}
        Subject: {subject}
        Grade Level: {grade_level}
        Content Type: {content_type}
        Learning Style: {learning_style}
        
        Content should be:
        1. Age-appropriate and engaging
        2. Aligned with curriculum standards
        3. Adapted for {learning_style} learning style
        4. Include relevant examples and applications
        5. Use clear and accessible language
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=0.4,
            do_sample=True,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def create_learning_objectives(
        self,
        topic: str,
        subject: str,
        grade_level: str,
        duration_hours: int = 1
    ) -> List[Dict[str, str]]:
        """
        Create learning objectives for a lesson
        
        Args:
            topic: Lesson topic
            subject: Subject area
            grade_level: Target grade level
            duration_hours: Lesson duration in hours
        """
        prompt = f"""
        Create learning objectives for a lesson with the following details:
        
        Topic: {topic}
        Subject: {subject}
        Grade Level: {grade_level}
        Duration: {duration_hours} hours
        
        Generate 3-5 specific, measurable, achievable, relevant, and time-bound (SMART) learning objectives.
        Include objectives for knowledge, skills, and attitudes.
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            temperature=0.3,
            do_sample=True,
            num_return_sequences=1
        )
        
        objectives_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse objectives into structured format
        objectives = self._parse_learning_objectives(objectives_text)
        
        return objectives
    
    def assess_content_difficulty(
        self,
        content: str,
        target_grade: str
    ) -> Dict[str, Union[str, float]]:
        """
        Assess difficulty level of educational content
        
        Args:
            content: Educational content to assess
            target_grade: Target grade level
        """
        prompt = f"""
        Assess the difficulty level of the following educational content:
        
        Target Grade: {target_grade}
        Content: {content[:1000]}...
        
        Provide difficulty assessment covering:
        1. Reading level complexity
        2. Concept complexity
        3. Vocabulary difficulty
        4. Overall difficulty rating (Easy/Medium/Hard)
        5. Recommendations for adaptation
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_type='assessment'
        )
        
        assessment = self.tokenizer.decode(
            self.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                temperature=0.3
            )[0],
            skip_special_tokens=True
        )
        
        # Get difficulty score from classifier
        difficulty_logits = outputs.difficulty_logits
        difficulty_levels = ['Easy', 'Medium', 'Hard']
        predicted_difficulty = difficulty_levels[torch.argmax(difficulty_logits, dim=-1).item()]
        
        return {
            'assessment': assessment,
            'difficulty_level': predicted_difficulty,
            'difficulty_score': torch.softmax(difficulty_logits, dim=-1).max().item()
        }
    
    def align_with_curriculum(
        self,
        content: str,
        curriculum_framework: str,
        subject: str
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Align content with curriculum standards
        
        Args:
            content: Educational content to align
            curriculum_framework: Curriculum framework (NCERT, CBSE, State Board, etc.)
            subject: Subject area
        """
        prompt = f"""
        Align the following content with curriculum standards:
        
        Curriculum Framework: {curriculum_framework}
        Subject: {subject}
        Content: {content[:1500]}...
        
        Provide curriculum alignment analysis covering:
        1. Relevant curriculum standards
        2. Alignment strength (High/Medium/Low)
        3. Gaps in coverage
        4. Recommendations for improvement
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_type='curriculum'
        )
        
        alignment_analysis = self.tokenizer.decode(
            self.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                temperature=0.3
            )[0],
            skip_special_tokens=True
        )
        
        # Extract curriculum standards (simplified)
        curriculum_standards = self._extract_curriculum_standards(alignment_analysis)
        
        return {
            'alignment_analysis': alignment_analysis,
            'curriculum_standards': curriculum_standards,
            'alignment_score': self._calculate_alignment_score(alignment_analysis)
        }
    
    def recommend_educational_resources(
        self,
        topic: str,
        subject: str,
        grade_level: str,
        learning_objectives: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        Recommend educational resources for learning
        
        Args:
            topic: Learning topic
            subject: Subject area
            grade_level: Target grade level
            learning_objectives: Specific learning objectives
        """
        objectives_text = "\\n".join(learning_objectives) if learning_objectives else "General understanding"
        
        prompt = f"""
        Recommend educational resources for the following learning scenario:
        
        Topic: {topic}
        Subject: {subject}
        Grade Level: {grade_level}
        Learning Objectives: {objectives_text}
        
        Recommend 3-5 resources including:
        1. Textbooks and reference materials
        2. Online resources and videos
        3. Interactive activities and exercises
        4. Assessment tools
        5. Supplementary materials
        
        For each resource, include:
        - Resource type
        - Title/name
        - Brief description
        - How it supports learning objectives
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=1024,
            temperature=0.4,
            do_sample=True,
            num_return_sequences=1
        )
        
        recommendations_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse recommendations into structured format
        recommendations = self._parse_recommendations(recommendations_text)
        
        return recommendations
    
    def _parse_learning_objectives(self, text: str) -> List[Dict[str, str]]:
        """
        Parse learning objectives from generated text
        """
        objectives = []
        lines = text.split('\\n')
        
        current_objective = {}
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_objective:
                    objectives.append(current_objective)
                current_objective = {'objective': line[2:].strip()}
            elif line.startswith('-') or line.startswith('*'):
                if 'details' not in current_objective:
                    current_objective['details'] = []
                current_objective['details'].append(line[1:].strip())
        
        if current_objective:
            objectives.append(current_objective)
        
        return objectives
    
    def _extract_curriculum_standards(self, text: str) -> List[str]:
        """
        Extract curriculum standards from alignment analysis
        """
        standards = []
        lines = text.split('\\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['standard', 'objective', 'outcome', 'competency']):
                standards.append(line.strip())
        
        return standards[:5]  # Return top 5 standards
    
    def _calculate_alignment_score(self, text: str) -> float:
        """
        Calculate curriculum alignment score
        """
        # Simple heuristic based on alignment keywords
        alignment_keywords = ['aligned', 'covers', 'addresses', 'meets', 'satisfies', 'complies']
        text_lower = text.lower()
        
        keyword_count = sum(1 for keyword in alignment_keywords if keyword in text_lower)
        alignment_score = min(keyword_count / len(alignment_keywords), 1.0)
        
        return alignment_score
    
    def _parse_recommendations(self, text: str) -> List[Dict[str, str]]:
        """
        Parse resource recommendations from generated text
        """
        recommendations = []
        lines = text.split('\\n')
        
        current_resource = {}
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_resource:
                    recommendations.append(current_resource)
                current_resource = {'title': line[2:].strip()}
            elif line.startswith('Type:'):
                current_resource['type'] = line[5:].strip()
            elif line.startswith('Description:'):
                current_resource['description'] = line[12:].strip()
            elif line.startswith('Supports:'):
                current_resource['supports'] = line[9:].strip()
        
        if current_resource:
            recommendations.append(current_resource)
        
        return recommendations


class VidyaYantra(nn.Module):
    """
    VidyaYantra: AI Teacher for personalized learning
    
    Specialized for:
    - Personalized learning paths
    - Adaptive content delivery
    - Real-time feedback and assessment
    - Learning progress tracking
    - Interactive tutoring sessions
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base model for conversation
        self.base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        
        # Student modeling
        self.student_profiler = nn.Sequential(
            nn.Linear(config.student_features, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )
        
        # Learning path generator
        self.path_generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size
            ),
            num_layers=3
        )
        
        # Assessment and feedback
        self.assessment_engine = nn.Linear(config.hidden_size, config.num_assessment_types)
        self.feedback_generator = nn.Linear(config.hidden_size, config.feedback_dim)
        
        # Progress tracking
        self.progress_tracker = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            bidirectional=True
        )
        
    def forward(
        self,
        student_input: str,
        student_profile: Optional[torch.Tensor] = None,
        learning_context: Optional[Dict] = None,
        **kwargs
    ):
        """
        Forward pass for AI teacher interaction
        
        Args:
            student_input: Student's question or response
            student_profile: Student profile features
            learning_context: Current learning context
        """
        outputs = {}
        
        # Process student input
        inputs = self.base_model.tokenizer(
            student_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        base_outputs = self.base_model(**inputs)
        input_hidden = base_outputs.last_hidden_state
        
        outputs['input_hidden_states'] = input_hidden
        
        # Process student profile if provided
        if student_profile is not None:
            profile_hidden = self.student_profiler(student_profile)
            outputs['student_profile_hidden'] = profile_hidden
        
        # Generate learning path if context provided
        if learning_context:
            path_outputs = self._generate_learning_path(
                input_hidden, 
                profile_hidden if student_profile is not None else None,
                learning_context
            )
            outputs.update(path_outputs)
        
        return outputs
    
    def create_personalized_session(
        self,
        student_id: str,
        subject: str,
        topic: str,
        learning_goals: List[str],
        session_duration: int = 30
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
        Create personalized learning session
        
        Args:
            student_id: Student identifier
            subject: Subject area
            topic: Session topic
            learning_goals: Learning goals for the session
            session_duration: Session duration in minutes
        """
        goals_text = "\\n".join(learning_goals)
        
        prompt = f"""
        Create a personalized learning session with the following details:
        
        Student ID: {student_id}
        Subject: {subject}
        Topic: {topic}
        Learning Goals: {goals_text}
        Session Duration: {session_duration} minutes
        
        Design the session to include:
        1. Welcome and introduction (2-3 minutes)
        2. Knowledge assessment (5 minutes)
        3. Main content delivery (15-20 minutes)
        4. Interactive activities (5-7 minutes)
        5. Summary and next steps (3-5 minutes)
        
        Adapt the session based on student's learning style and pace.
        """
        
        outputs = self.forward(student_input=prompt)
        
        session_plan = self.base_model.tokenizer.decode(
            self.base_model.generate(
                input_ids=outputs['input_hidden_states'].argmax(dim=-1),
                max_length=1024,
                temperature=0.4
            )[0],
            skip_special_tokens=True
        )
        
        # Parse session plan into structured format
        session_structure = self._parse_session_plan(session_plan)
        
        return {
            'session_plan': session_plan,
            'session_structure': session_structure,
            'estimated_duration': session_duration,
            'subject': subject,
            'topic': topic
        }
    
    def provide_real_time_feedback(
        self,
        student_response: str,
        expected_answer: str,
        question_type: str = "multiple_choice"
    ) -> Dict[str, Union[str, float]]:
        """
        Provide real-time feedback on student response
        
        Args:
            student_response: Student's answer
            expected_answer: Correct answer
            question_type: Type of question
        """
        prompt = f"""
        Provide feedback on the following student response:
        
        Question Type: {question_type}
        Expected Answer: {expected_answer}
        Student Response: {student_response}
        
        Provide feedback that includes:
        1. Accuracy assessment
        2. Strengths in the response
        3. Areas for improvement
        4. Specific suggestions for better understanding
        5. Encouraging and supportive tone
        """
        
        outputs = self.forward(student_input=prompt)
        
        feedback = self.base_model.tokenizer.decode(
            self.base_model.generate(
                input_ids=outputs['input_hidden_states'].argmax(dim=-1),
                max_length=512,
                temperature=0.3
            )[0],
            skip_special_tokens=True
        )
        
        # Assess response accuracy
        accuracy_score = self._assess_response_accuracy(student_response, expected_answer)
        
        return {
            'feedback': feedback,
            'accuracy_score': accuracy_score,
            'response_quality': 'Excellent' if accuracy_score > 0.9 else 'Good' if accuracy_score > 0.7 else 'Needs Improvement'
        }
    
    def adapt_learning_path(
        self,
        student_progress: Dict[str, float],
        current_difficulty: str,
        performance_trend: str
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Adapt learning path based on student progress
        
        Args:
            student_progress: Dictionary of topic-wise progress scores
            current_difficulty: Current difficulty level
            performance_trend: Performance trend (improving, stable, declining)
        """
        progress_text = "\\n".join([f"{topic}: {score:.2f}" for topic, score in student_progress.items()])
        
        prompt = f"""
        Adapt learning path based on student progress:
        
        Current Progress: {progress_text}
        Current Difficulty: {current_difficulty}
        Performance Trend: {performance_trend}
        
        Provide adaptation recommendations:
        1. Difficulty level adjustment
        2. Topics needing reinforcement
        3. Topics ready for advancement
        4. Learning strategy changes
        5. Additional support needed
        """
        
        outputs = self.forward(student_input=prompt)
        
        adaptation_plan = self.base_model.tokenizer.decode(
            self.base_model.generate(
                input_ids=outputs['input_hidden_states'].argmax(dim=-1),
                max_length=512,
                temperature=0.4
            )[0],
            skip_special_tokens=True
        )
        
        # Extract adaptation recommendations
        recommendations = self._extract_adaptation_recommendations(adaptation_plan)
        
        return {
            'adaptation_plan': adaptation_plan,
            'recommendations': recommendations,
            'new_difficulty': self._determine_new_difficulty(student_progress, performance_trend)
        }
    
    def generate_progress_report(
        self,
        student_id: str,
        session_history: List[Dict],
        time_period: str = "last_week"
    ) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Generate comprehensive progress report
        
        Args:
            student_id: Student identifier
            session_history: List of session data
            time_period: Time period for report
        """
        # Process session history
        progress_metrics = self._calculate_progress_metrics(session_history)
        
        prompt = f"""
        Generate progress report for the following student data:
        
        Student ID: {student_id}
        Time Period: {time_period}
        Sessions Completed: {len(session_history)}
        Progress Metrics: {progress_metrics}
        
        Include in the report:
        1. Overall progress summary
        2. Subject-wise performance
        3. Strengths and achievements
        4. Areas needing improvement
        5. Recommendations for next steps
        6. Learning goals for next period
        """
        
        outputs = self.forward(student_input=prompt)
        
        progress_report = self.base_model.tokenizer.decode(
            self.base_model.generate(
                input_ids=outputs['input_hidden_states'].argmax(dim=-1),
                max_length=1024,
                temperature=0.3
            )[0],
            skip_special_tokens=True
        )
        
        return {
            'progress_report': progress_report,
            'progress_metrics': progress_metrics,
            'sessions_completed': len(session_history),
            'time_period': time_period
        }
    
    def _generate_learning_path(
        self,
        input_hidden: torch.Tensor,
        profile_hidden: Optional[torch.Tensor],
        learning_context: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Generate personalized learning path
        """
        # Combine input and profile information
        combined_hidden = input_hidden
        if profile_hidden is not None:
            combined_hidden = torch.cat([input_hidden.mean(dim=1), profile_hidden], dim=-1)
        
        # Generate learning path
        path_hidden = self.path_generator(
            combined_hidden.unsqueeze(1),
            combined_hidden.unsqueeze(1)
        )
        
        return {
            'learning_path_hidden': path_hidden
        }
    
    def _assess_response_accuracy(self, student_response: str, expected_answer: str) -> float:
        """
        Assess accuracy of student response
        """
        # Simple keyword matching (can be enhanced with semantic similarity)
        student_words = set(student_response.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        if not expected_words:
            return 0.0
        
        intersection = student_words.intersection(expected_words)
        accuracy = len(intersection) / len(expected_words)
        
        return accuracy
    
    def _parse_session_plan(self, text: str) -> List[Dict[str, str]]:
        """
        Parse session plan into structured format
        """
        sections = []
        lines = text.split('\\n')
        
        current_section = {}
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_section:
                    sections.append(current_section)
                current_section = {'title': line[2:].strip()}
            elif line.startswith('Duration:'):
                current_section['duration'] = line[9:].strip()
            elif line.startswith('Activities:'):
                current_section['activities'] = line[11:].strip()
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _extract_adaptation_recommendations(self, text: str) -> List[str]:
        """
        Extract adaptation recommendations from text
        """
        recommendations = []
        lines = text.split('\\n')
        
        for line in lines:
            if line.startswith('-') or line.startswith('*'):
                recommendations.append(line[1:].strip())
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _determine_new_difficulty(self, progress: Dict[str, float], trend: str) -> str:
        """
        Determine new difficulty level based on progress and trend
        """
        avg_progress = sum(progress.values()) / len(progress) if progress else 0
        
        if trend == 'improving' and avg_progress > 0.8:
            return 'Advanced'
        elif trend == 'improving' and avg_progress > 0.6:
            return 'Intermediate'
        elif trend == 'declining' and avg_progress < 0.4:
            return 'Beginner'
        else:
            return 'Intermediate'  # Default
    
    def _calculate_progress_metrics(self, session_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate progress metrics from session history
        """
        if not session_history:
            return {}
        
        metrics = {
            'average_score': sum(s.get('score', 0) for s in session_history) / len(session_history),
            'completion_rate': sum(1 for s in session_history if s.get('completed', False)) / len(session_history),
            'engagement_level': sum(s.get('engagement', 0) for s in session_history) / len(session_history),
            'improvement_rate': self._calculate_improvement_rate(session_history)
        }
        
        return metrics
    
    def _calculate_improvement_rate(self, session_history: List[Dict]) -> float:
        """
        Calculate improvement rate over sessions
        """
        if len(session_history) < 2:
            return 0.0
        
        scores = [s.get('score', 0) for s in session_history]
        improvement = scores[-1] - scores[0]
        
        return improvement / len(session_history)


# Model configurations
BHARAT_EDU_CONFIG = {
    'vocab_size': 65000,
    'hidden_size': 2048,
    'intermediate_size': 8192,
    'num_hidden_layers': 24,
    'num_attention_heads': 32,
    'max_position_embeddings': 2048,
    'num_subjects': 20,
    'num_grade_levels': 12,
    'learning_objective_dim': 512,
    'num_difficulty_levels': 3
}

VIDYA_YANTRA_CONFIG = {
    'hidden_size': 1024,
    'num_attention_heads': 16,
    'intermediate_size': 4096,
    'student_features': 50,
    'num_assessment_types': 10,
    'feedback_dim': 512
}