/**
 * Education AI domain-specific client for BFMF
 */

import { BharatAIClient } from '../client';
import { GenerationRequest, GenerationResponse } from '../types';
import { BharatAIError } from '../errors';

/**
 * Tutoring session request
 */
export interface TutoringRequest {
  /** Subject */
  subject: string;
  /** Topic/Concept */
  topic: string;
  /** Student level */
  studentLevel: 'primary' | 'secondary' | 'higher-secondary' | 'undergraduate' | 'postgraduate';
  /** Learning style */
  learningStyle?: 'visual' | 'auditory' | 'kinesthetic' | 'reading';
  /** Language preference */
  language?: string;
  /** Previous knowledge */
  previousKnowledge?: string[];
  /** Learning objectives */
  learningObjectives?: string[];
}

/**
 * Tutoring session response
 */
export interface TutoringResponse {
  /** Session ID */
  sessionId: string;
  /** Tutor response */
  tutorResponse: string;
  /** Key concepts covered */
  keyConcepts: string[];
  /** Learning resources */
  learningResources: Array<{
    type: 'video' | 'article' | 'exercise' | 'simulation';
    title: string;
    url?: string;
    description: string;
  }>;
  /** Practice questions */
  practiceQuestions: Array<{
    question: string;
    type: 'multiple-choice' | 'short-answer' | 'essay' | 'problem-solving';
    difficulty: 'easy' | 'medium' | 'hard';
    answer?: string;
    explanation?: string;
  }>;
  /** Progress assessment */
  progressAssessment: {
    understanding: number; // 0-100
    confidence: number; // 0-100
    areasForImprovement: string[];
  };
  /** Session time */
  sessionTime: number;
}

/**
 * Content generation request
 */
export interface ContentGenerationRequest {
  /** Subject */
  subject: string;
  /** Topic */
  topic: string;
  /** Content type */
  contentType: 'lesson-plan' | 'study-material' | 'assignment' | 'quiz' | 'presentation';
  ** Target audience */
  targetAudience: {
    level: 'primary' | 'secondary' | 'higher-secondary' | 'undergraduate' | 'postgraduate';
    age?: number;
    background?: string;
  };
  ** Learning objectives */
  learningObjectives?: string[];
  ** Duration (in minutes) */
  duration?: number;
  ** Language */
  language?: string;
  ** Include visual aids */
  includeVisualAids?: boolean;
}

/**
 * Content generation response
 */
export interface ContentGenerationResponse {
  /** Generated content */
  content: string;
  /** Content structure */
  structure: Array<{
    type: 'heading' | 'paragraph' | 'list' | 'table' | 'image' | 'video' | 'exercise';
    title?: string;
    content: string;
    order: number;
  }>;
  /** Learning outcomes */
  learningOutcomes: string[];
  /** Assessment criteria */
  assessmentCriteria?: string[];
  /** Estimated completion time */
  estimatedCompletionTime: number;
  /** Generation time */
  generationTime: number;
}

/**
 * Assessment generation request
 */
export interface AssessmentGenerationRequest {
  /** Subject */
  subject: string;
  /** Topic */
  topic: string;
  /** Assessment type */
  assessmentType: 'quiz' | 'assignment' | 'exam' | 'project';
  /** Difficulty level */
  difficulty: 'easy' | 'medium' | 'hard' | 'mixed';
  /** Number of questions */
  numberOfQuestions: number;
  /** Question types */
  questionTypes?: ('multiple-choice' | 'short-answer' | 'essay' | 'problem-solving' | 'true-false')[];
  /** Student level */
  studentLevel: 'primary' | 'secondary' | 'higher-secondary' | 'undergraduate' | 'postgraduate';
  ** Time limit (in minutes) */
  timeLimit?: number;
  ** Total marks */
  totalMarks?: number;
  ** Language */
  language?: string;
}

/**
 * Assessment generation response
 */
export interface AssessmentGenerationResponse {
  /** Assessment ID */
  assessmentId: string;
  /** Questions */
  questions: Array<{
    id: string;
    question: string;
    type: 'multiple-choice' | 'short-answer' | 'essay' | 'problem-solving' | 'true-false';
    difficulty: 'easy' | 'medium' | 'hard';
    marks: number;
    options?: string[]; // For multiple choice
    correctAnswer?: string;
    explanation?: string;
  }>;
  /** Answer key */
  answerKey: Record<string, string>;
  /** Rubric */
  rubric?: {
    criteria: Array<{
      name: string;
      description: string;
      maxMarks: number;
    }>;
  };
  /** Instructions */
  instructions: string;
  /** Estimated completion time */
  estimatedCompletionTime: number;
  /** Generation time */
  generationTime: number;
}

/**
 * Progress tracking request
 */
export interface ProgressTrackingRequest {
  /** Student ID */
  studentId: string;
  ** Subject */
  subject: string;
  ** Topics covered */
  topicsCovered: string[];
  ** Assessment scores */
  assessmentScores: Array<{
    assessmentId: string;
    score: number;
    maxScore: number;
    date: string;
  }>;
  ** Learning goals */
  learningGoals?: string[];
  ** Time spent (in minutes) */
  timeSpent?: number;
}

/**
 * Progress tracking response
 */
export interface ProgressTrackingResponse {
  /** Student ID */
  studentId: string;
  ** Overall progress */
  overallProgress: {
    percentage: number;
    grade: 'A+' | 'A' | 'B' | 'C' | 'D' | 'F';
    status: 'excellent' | 'good' | 'average' | 'needs-improvement';
  };
  ** Topic-wise progress */
  topicProgress: Array<{
    topic: string;
    mastery: number; // 0-100
    status: 'mastered' | 'in-progress' | 'needs-attention';
    lastStudied: string;
    timeSpent: number;
  }>;
  ** Strengths */
  strengths: string[];
  ** Areas for improvement */
  areasForImprovement: string[];
  ** Recommendations */
  recommendations: Array<{
    type: 'study-material' | 'practice' | 'tutoring' | 'assessment';
    title: string;
    description: string;
    priority: 'high' | 'medium' | 'low';
  }>;
  ** Predicted performance */
  predictedPerformance: {
    nextAssessment: number; // Predicted score
    confidence: number; // 0-100
  };
  /** Analysis time */
  analysisTime: number;
}

/**
 * Education AI domain-specific client
 */
export class EducationAIClient {
  private client: BharatAIClient;
  private domainEndpoint: string;

  /**
   * Create a new Education AI client
   */
  constructor(client: BharatAIClient, domainEndpoint: string = '/education') {
    this.client = client;
    this.domainEndpoint = domainEndpoint;
  }

  /**
   * Start tutoring session
   */
  async startTutoringSession(request: TutoringRequest): Promise<TutoringResponse> {
    const response = await this.client['makeRequest']<TutoringResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/tutoring`,
      data: {
        subject: request.subject,
        topic: request.topic,
        student_level: request.studentLevel,
        learning_style: request.learningStyle,
        language: request.language,
        previous_knowledge: request.previousKnowledge,
        learning_objectives: request.learningObjectives
      }
    });

    return response;
  }

  /**
   * Continue tutoring session
   */
  async continueTutoringSession(sessionId: string, userMessage: string): Promise<TutoringResponse> {
    const response = await this.client['makeRequest']<TutoringResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/tutoring/${sessionId}/continue`,
      data: {
        user_message: userMessage
      }
    });

    return response;
  }

  /**
   * Generate educational content
   */
  async generateContent(request: ContentGenerationRequest): Promise<ContentGenerationResponse> {
    const response = await this.client['makeRequest']<ContentGenerationResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/generate-content`,
      data: {
        subject: request.subject,
        topic: request.topic,
        content_type: request.contentType,
        target_audience: request.targetAudience,
        learning_objectives: request.learningObjectives,
        duration: request.duration,
        language: request.language,
        include_visual_aids: request.includeVisualAids
      }
    });

    return response;
  }

  /**
   * Generate assessment
   */
  async generateAssessment(request: AssessmentGenerationRequest): Promise<AssessmentGenerationResponse> {
    const response = await this.client['makeRequest']<AssessmentGenerationResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/generate-assessment`,
      data: {
        subject: request.subject,
        topic: request.topic,
        assessment_type: request.assessmentType,
        difficulty: request.difficulty,
        number_of_questions: request.numberOfQuestions,
        question_types: request.questionTypes,
        student_level: request.studentLevel,
        time_limit: request.timeLimit,
        total_marks: request.totalMarks,
        language: request.language
      }
    });

    return response;
  }

  /**
   * Track student progress
   */
  async trackProgress(request: ProgressTrackingRequest): Promise<ProgressTrackingResponse> {
    const response = await this.client['makeRequest']<ProgressTrackingResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/track-progress`,
      data: {
        student_id: request.studentId,
        subject: request.subject,
        topics_covered: request.topicsCovered,
        assessment_scores: request.assessmentScores,
        learning_goals: request.learningGoals,
        time_spent: request.timeSpent
      }
    });

    return response;
  }

  /**
   * Get personalized learning path
   */
  async getPersonalizedLearningPath(params: {
    studentId: string;
    subject: string;
    currentLevel: string;
    goals: string[];
    availableTime: number; // hours per week
    preferredLanguage?: string;
  }): Promise<{
    pathId: string;
    weeks: Array<{
      week: number;
      topics: Array<{
        topic: string;
        estimatedHours: number;
        resources: Array<{
          type: 'video' | 'reading' | 'exercise' | 'project';
          title: string;
          url?: string;
          description: string;
        }>;
        assessments: Array<{
          type: 'quiz' | 'assignment' | 'project';
          title: string;
          estimatedTime: number;
        }>;
      }>;
      milestones: string[];
    }>;
    totalEstimatedHours: number;
    completionDate: string;
  }> {
    const response = await this.client['makeRequest']<{
      pathId: string;
      weeks: Array<{
        week: number;
        topics: Array<{
          topic: string;
          estimatedHours: number;
          resources: Array<{
            type: 'video' | 'reading' | 'exercise' | 'project';
            title: string;
            url?: string;
            description: string;
          }>;
          assessments: Array<{
            type: 'quiz' | 'assignment' | 'project';
            title: string;
            estimatedTime: number;
          }>;
        }>;
        milestones: string[];
      }>;
      totalEstimatedHours: number;
      completionDate: string;
    }>({
      method: 'POST',
      url: `${this.domainEndpoint}/learning-path`,
      data: params
    });

    return response;
  }

  /**
   * Grade assessment automatically
   */
  async autoGradeAssessment(assessmentId: string, submissions: Array<{
    studentId: string;
    answers: Record<string, string>;
    submissionTime: string;
  }>): Promise<Array<{
    studentId: string;
    score: number;
    maxScore: number;
    percentage: number;
    grade: string;
    feedback: string;
    questionFeedback: Record<string, {
      score: number;
      feedback: string;
      correct: boolean;
    }>;
    timeSpent: number;
  }>> {
    const response = await this.client['makeRequest']<Array<{
      studentId: string;
      score: number;
      maxScore: number;
      percentage: number;
      grade: string;
      feedback: string;
      questionFeedback: Record<string, {
        score: number;
        feedback: string;
        correct: boolean;
      }>;
      timeSpent: number;
    }>>({
      method: 'POST',
      url: `${this.domainEndpoint}/grade-assessment/${assessmentId}`,
      data: { submissions }
    });

    return response;
  }

  /**
   * Get subject curriculum
   */
  async getSubjectCurriculum(subject: string, level: string): Promise<{
    subject: string;
    level: string;
    units: Array<{
      name: string;
      topics: string[];
      learningObjectives: string[];
      estimatedHours: number;
      prerequisites?: string[];
    }>;
    totalHours: number;
    skills: string[];
  }> {
    const response = await this.client['makeRequest']<{
      subject: string;
      level: string;
      units: Array<{
        name: string;
        topics: string[];
        learningObjectives: string[];
        estimatedHours: number;
        prerequisites?: string[];
      }>;
      totalHours: number;
      skills: string[];
    }>({
      method: 'GET',
      url: `${this.domainEndpoint}/curriculum/${subject}/${level}`
    });

    return response;
  }

  /**
   * Generate interactive exercise
   */
  async generateInteractiveExercise(params: {
    subject: string;
    topic: string;
    exerciseType: 'simulation' | 'game' | 'lab' | 'case-study';
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    duration?: number;
    learningObjectives?: string[];
  }): Promise<{
    exerciseId: string;
    title: string;
    description: string;
    instructions: string;
    interactiveElements: Array<{
      type: 'drag-drop' | 'multiple-choice' | 'input' | 'simulation';
      question: string;
      options?: string[];
      correctAnswer?: string;
      explanation?: string;
    }>;
    learningOutcomes: string[];
    estimatedTime: number;
  }> {
    const response = await this.client['makeRequest']<{
      exerciseId: string;
      title: string;
      description: string;
      instructions: string;
      interactiveElements: Array<{
        type: 'drag-drop' | 'multiple-choice' | 'input' | 'simulation';
        question: string;
        options?: string[];
        correctAnswer?: string;
        explanation?: string;
      }>;
      learningOutcomes: string[];
      estimatedTime: number;
    }>({
      method: 'POST',
      url: `${this.domainEndpoint}/interactive-exercise`,
      data: params
    });

    return response;
  }
}