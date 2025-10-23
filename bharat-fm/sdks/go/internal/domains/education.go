package domains

import (
    "context"
    "fmt"

    "github.com/bharat-ai/bharat-fm/sdks/go/internal/client"
    "github.com/bharat-ai/bharat-fm/sdks/go/internal/models"
)

// EducationAIClient provides education-specific AI capabilities
type EducationAIClient struct {
    client *client.Client
}

// NewEducationAIClient creates a new education AI client
func NewEducationAIClient(client *client.Client) *EducationAIClient {
    return &EducationAIClient{
        client: client,
    }
}

// TutoringRequest represents a tutoring session request
type TutoringRequest struct {
    Subject          string   `json:"subject"`
    Topic            string   `json:"topic"`
    StudentLevel     string   `json:"student_level"`
    LearningStyle    string   `json:"learning_style,omitempty"`
    Language         string   `json:"language,omitempty"`
    PreviousKnowledge []string `json:"previous_knowledge,omitempty"`
    LearningObjectives []string `json:"learning_objectives,omitempty"`
}

// TutoringResponse represents a tutoring session response
type TutoringResponse struct {
    SessionID        string `json:"session_id"`
    TutorResponse    string `json:"tutor_response"`
    KeyConcepts      []string `json:"key_concepts"`
    LearningResources []struct {
        Type        string `json:"type"`
        Title       string `json:"title"`
        URL         string `json:"url,omitempty"`
        Description string `json:"description"`
    } `json:"learning_resources"`
    PracticeQuestions []struct {
        Question    string `json:"question"`
        Type        string `json:"type"`
        Difficulty  string `json:"difficulty"`
        Answer      string `json:"answer,omitempty"`
        Explanation string `json:"explanation,omitempty"`
    } `json:"practice_questions"`
    ProgressAssessment struct {
        Understanding      float64   `json:"understanding"`
        Confidence         float64   `json:"confidence"`
        AreasForImprovement []string `json:"areas_for_improvement"`
    } `json:"progress_assessment"`
    SessionTime float64 `json:"session_time"`
}

// ContentGenerationRequest represents a content generation request
type ContentGenerationRequest struct {
    Subject           string                 `json:"subject"`
    Topic             string                 `json:"topic"`
    ContentType       string                 `json:"content_type"`
    TargetAudience    map[string]interface{} `json:"target_audience"`
    LearningObjectives []string               `json:"learning_objectives,omitempty"`
    Duration          int                    `json:"duration,omitempty"`
    Language          string                 `json:"language,omitempty"`
    IncludeVisualAids  bool                   `json:"include_visual_aids,omitempty"`
}

// ContentGenerationResponse represents a content generation response
type ContentGenerationResponse struct {
    Content               string `json:"content"`
    Structure             []struct {
        Type     string `json:"type"`
        Title    string `json:"title,omitempty"`
        Content  string `json:"content"`
        Order    int    `json:"order"`
    } `json:"structure"`
    LearningOutcomes     []string `json:"learning_outcomes"`
    AssessmentCriteria  []string `json:"assessment_criteria,omitempty"`
    EstimatedCompletionTime float64 `json:"estimated_completion_time"`
    GenerationTime       float64 `json:"generation_time"`
}

// StartTutoringSession starts a tutoring session
func (c *EducationAIClient) StartTutoringSession(ctx context.Context, req *TutoringRequest) (*TutoringResponse, error) {
    var result TutoringResponse
    
    apiReq := map[string]interface{}{
        "subject":       req.Subject,
        "topic":         req.Topic,
        "student_level": req.StudentLevel,
    }
    
    if req.LearningStyle != "" {
        apiReq["learning_style"] = req.LearningStyle
    }
    if req.Language != "" {
        apiReq["language"] = req.Language
    }
    if len(req.PreviousKnowledge) > 0 {
        apiReq["previous_knowledge"] = req.PreviousKnowledge
    }
    if len(req.LearningObjectives) > 0 {
        apiReq["learning_objectives"] = req.LearningObjectives
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/education/tutoring")
    
    if err != nil {
        return nil, fmt.Errorf("tutoring session request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("tutoring session API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// GenerateContent generates educational content
func (c *EducationAIClient) GenerateContent(ctx context.Context, req *ContentGenerationRequest) (*ContentGenerationResponse, error) {
    var result ContentGenerationResponse
    
    apiReq := map[string]interface{}{
        "subject":         req.Subject,
        "topic":           req.Topic,
        "content_type":    req.ContentType,
        "target_audience": req.TargetAudience,
    }
    
    if len(req.LearningObjectives) > 0 {
        apiReq["learning_objectives"] = req.LearningObjectives
    }
    if req.Duration > 0 {
        apiReq["duration"] = req.Duration
    }
    if req.Language != "" {
        apiReq["language"] = req.Language
    }
    apiReq["include_visual_aids"] = req.IncludeVisualAids

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/education/generate-content")
    
    if err != nil {
        return nil, fmt.Errorf("content generation request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("content generation API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// GetSubjectCurriculum gets subject curriculum
func (c *EducationAIClient) GetSubjectCurriculum(ctx context.Context, subject, level string) (*struct {
    Subject      string `json:"subject"`
    Level        string `json:"level"`
    Units        []struct {
        Name               string   `json:"name"`
        Topics             []string `json:"topics"`
        LearningObjectives []string `json:"learning_objectives"`
        EstimatedHours     int      `json:"estimated_hours"`
        Prerequisites      []string `json:"prerequisites,omitempty"`
    } `json:"units"`
    TotalHours   int      `json:"total_hours"`
    Skills       []string `json:"skills"`
}, error) {
    var result struct {
        Subject      string `json:"subject"`
        Level        string `json:"level"`
        Units        []struct {
            Name               string   `json:"name"`
            Topics             []string `json:"topics"`
            LearningObjectives []string `json:"learning_objectives"`
            EstimatedHours     int      `json:"estimated_hours"`
            Prerequisites      []string `json:"prerequisites,omitempty"`
        } `json:"units"`
        TotalHours   int      `json:"total_hours"`
        Skills       []string `json:"skills"`
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetResult(&result).
        Get(fmt.Sprintf("/education/curriculum/%s/%s", subject, level))
    
    if err != nil {
        return nil, fmt.Errorf("get curriculum request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("get curriculum API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// SimpleStartTutoringSession provides a simple tutoring session interface
func (c *EducationAIClient) SimpleStartTutoringSession(ctx context.Context, subject, topic, studentLevel string) (string, error) {
    req := &TutoringRequest{
        Subject:      subject,
        Topic:        topic,
        StudentLevel: studentLevel,
    }
    
    resp, err := c.StartTutoringSession(ctx, req)
    if err != nil {
        return "", err
    }
    
    return resp.TutorResponse, nil
}

// SimpleGenerateContent provides a simple content generation interface
func (c *EducationAIClient) SimpleGenerateContent(ctx context.Context, subject, topic, contentType string) (string, error) {
    targetAudience := map[string]interface{}{
        "level": "secondary",
        "age":   14,
    }
    
    req := &ContentGenerationRequest{
        Subject:        subject,
        Topic:          topic,
        ContentType:    contentType,
        TargetAudience: targetAudience,
    }
    
    resp, err := c.GenerateContent(ctx, req)
    if err != nil {
        return "", err
    }
    
    return resp.Content, nil
}