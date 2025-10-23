/**
 * Tutoring Component for Bharat AI SDK
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  ScrollView,
  Alert,
  Modal,
  FlatList,
} from 'react-native';

import { BharatAIClient } from '../client/BharatAIClient';
import { EducationAIClient, TutoringSessionRequest, TutoringSessionResponse } from '../domains/EducationAIClient';
import { BharatAIError } from '../errors/BharatAIError';

interface SubjectOption {
  id: string;
  name: string;
  icon: string;
}

interface GradeLevelOption {
  value: string;
  label: string;
}

interface TutoringComponentProps {
  /** Bharat AI client instance */
  client: BharatAIClient;
  /** Available subjects */
  subjects?: SubjectOption[];
  /** Available grade levels */
  gradeLevels?: GradeLevelOption[];
  ** On session generated callback */
  onSessionGenerated?: (session: TutoringSessionResponse) => void;
  ** On error callback */
  onError?: (error: BharatAIError) => void;
  ** Custom styles */
  style?: any;
}

export const TutoringComponent: React.FC<TutoringComponentProps> = ({
  client,
  subjects = [
    { id: 'math', name: 'Mathematics', icon: '🔢' },
    { id: 'science', name: 'Science', icon: '🔬' },
    { id: 'english', name: 'English', icon: '📚' },
    { id: 'history', name: 'History', icon: '📜' },
    { id: 'geography', name: 'Geography', icon: '🌍' },
    { id: 'computer', name: 'Computer Science', icon: '💻' },
  ],
  gradeLevels = [
    { value: 'elementary', label: 'Elementary (K-5)' },
    { value: 'middle', label: 'Middle School (6-8)' },
    { value: 'high', label: 'High School (9-12)' },
    { value: 'undergraduate', label: 'Undergraduate' },
    { value: 'graduate', label: 'Graduate' },
  ],
  onSessionGenerated,
  onError,
  style,
}) => {
  const [selectedSubject, setSelectedSubject] = useState<SubjectOption | null>(null);
  const [selectedGrade, setSelectedGrade] = useState<GradeLevelOption | null>(null);
  const [topic, setTopic] = useState('');
  const [currentUnderstanding, setCurrentUnderstanding] = useState<'beginner' | 'intermediate' | 'advanced'>('beginner');
  const [learningObjectives, setLearningObjectives] = useState<string[]>(['']);
  const [teachingStyle, setTeachingStyle] = useState<'visual' | 'auditory' | 'kinesthetic' | 'reading'>('visual');
  const [language, setLanguage] = useState('en');
  const [duration, setDuration] = useState(30);
  const [context, setContext] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentSession, setCurrentSession] = useState<TutoringSessionResponse | null>(null);
  const [showObjectivesModal, setShowObjectivesModal] = useState(false);

  const educationClient = new EducationAIClient(client);

  useEffect(() => {
    // Initialize client if not already initialized
    if (!client.isClientInitialized()) {
      client.initialize().catch(error => {
        console.error('Failed to initialize client:', error);
        onError?.(error);
      });
    }
  }, [client]);

  const handleGenerateSession = async () => {
    if (!selectedSubject || !selectedGrade || !topic.trim()) {
      Alert.alert('Missing Information', 'Please select a subject, grade level, and enter a topic.');
      return;
    }

    setIsGenerating(true);

    try {
      const request: TutoringSessionRequest = {
        subject: selectedSubject.name,
        gradeLevel: selectedGrade.label,
        topic: topic.trim(),
        currentUnderstanding,
        learningObjectives: learningObjectives.filter(obj => obj.trim()),
        teachingStyle,
        language,
        duration,
        context: context.trim() || undefined,
      };

      const response = await educationClient.generateTutoringSession(request);
      
      setCurrentSession(response);
      onSessionGenerated?.(response);
    } catch (error) {
      console.error('Session generation error:', error);
      const aiError = error instanceof BharatAIError ? error : new BharatAIError('TUTORING_ERROR', 'Failed to generate tutoring session', error);
      onError?.(aiError);
      
      Alert.alert('Generation Error', 'Failed to generate tutoring session. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const addLearningObjective = () => {
    setLearningObjectives([...learningObjectives, '']);
  };

  const updateLearningObjective = (index: number, value: string) => {
    const newObjectives = [...learningObjectives];
    newObjectives[index] = value;
    setLearningObjectives(newObjectives);
  };

  const removeLearningObjective = (index: number) => {
    if (learningObjectives.length > 1) {
      const newObjectives = learningObjectives.filter((_, i) => i !== index);
      setLearningObjectives(newObjectives);
    }
  };

  const resetForm = () => {
    setSelectedSubject(null);
    setSelectedGrade(null);
    setTopic('');
    setCurrentUnderstanding('beginner');
    setLearningObjectives(['']);
    setTeachingStyle('visual');
    setLanguage('en');
    setDuration(30);
    setContext('');
    setCurrentSession(null);
  };

  const renderSessionContent = () => {
    if (!currentSession) return null;

    return (
      <ScrollView style={styles.sessionContainer}>
        <View style={styles.sessionHeader}>
          <Text style={styles.sessionTitle}>{selectedSubject?.name} - {topic}</Text>
          <Text style={styles.sessionSubtitle}>{selectedGrade?.label} • {duration} minutes</Text>
        </View>

        <View style={styles.sessionSection}>
          <Text style={styles.sectionTitle}>Introduction</Text>
          <Text style={styles.sectionContent}>{currentSession.content.introduction}</Text>
        </View>

        <View style={styles.sessionSection}>
          <Text style={styles.sectionTitle}>Main Content</Text>
          <Text style={styles.sectionContent}>{currentSession.content.mainContent}</Text>
        </View>

        {currentSession.content.examples.length > 0 && (
          <View style={styles.sessionSection}>
            <Text style={styles.sectionTitle}>Examples</Text>
            {currentSession.content.examples.map((example, index) => (
              <View key={index} style={styles.exampleItem}>
                <Text style={styles.exampleText}>{example}</Text>
              </View>
            ))}
          </View>
        )}

        {currentSession.content.practiceQuestions.length > 0 && (
          <View style={styles.sessionSection}>
            <Text style={styles.sectionTitle}>Practice Questions</Text>
            {currentSession.content.practiceQuestions.map((question, index) => (
              <View key={index} style={styles.questionItem}>
                <Text style={styles.questionText}>
                  Q{index + 1}: {question.question}
                </Text>
                <Text style={styles.answerText}>
                  A: {question.answer}
                </Text>
                <Text style={styles.difficultyText}>
                  Difficulty: {question.difficulty}
                </Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.sessionSection}>
          <Text style={styles.sectionTitle}>Summary</Text>
          <Text style={styles.sectionContent}>{currentSession.content.summary}</Text>
        </View>

        {currentSession.resources && currentSession.resources.length > 0 && (
          <View style={styles.sessionSection}>
            <Text style={styles.sectionTitle}>Learning Resources</Text>
            {currentSession.resources.map((resource, index) => (
              <View key={index} style={styles.resourceItem}>
                <Text style={styles.resourceType}>{resource.type}</Text>
                <Text style={styles.resourceTitle}>{resource.title}</Text>
                <Text style={styles.resourceDescription}>{resource.description}</Text>
              </View>
            ))}
          </View>
        )}

        <TouchableOpacity style={styles.newSessionButton} onPress={resetForm}>
          <Text style={styles.newSessionButtonText}>Start New Session</Text>
        </TouchableOpacity>
      </ScrollView>
    );
  };

  const renderForm = () => (
    <ScrollView style={styles.formContainer}>
      <Text style={styles.formTitle}>Create Tutoring Session</Text>

      <View style={styles.formSection}>
        <Text style={styles.label}>Subject</Text>
        <View style={styles.subjectGrid}>
          {subjects.map(subject => (
            <TouchableOpacity
              key={subject.id}
              style={[
                styles.subjectCard,
                selectedSubject?.id === subject.id && styles.selectedSubjectCard,
              ]}
              onPress={() => setSelectedSubject(subject)}
            >
              <Text style={styles.subjectIcon}>{subject.icon}</Text>
              <Text style={styles.subjectName}>{subject.name}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Grade Level</Text>
        <View style={styles.gradeContainer}>
          {gradeLevels.map(grade => (
            <TouchableOpacity
              key={grade.value}
              style={[
                styles.gradeOption,
                selectedGrade?.value === grade.value && styles.selectedGradeOption,
              ]}
              onPress={() => setSelectedGrade(grade)}
            >
              <Text style={styles.gradeText}>{grade.label}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Topic</Text>
        <TextInput
          style={styles.input}
          value={topic}
          onChangeText={setTopic}
          placeholder="Enter the topic you want to learn..."
          placeholderTextColor="#999"
        />
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Current Understanding</Text>
        <View style={styles.understandingContainer}>
          {(['beginner', 'intermediate', 'advanced'] as const).map(level => (
            <TouchableOpacity
              key={level}
              style={[
                styles.understandingOption,
                currentUnderstanding === level && styles.selectedUnderstandingOption,
              ]}
              onPress={() => setCurrentUnderstanding(level)}
            >
              <Text style={styles.understandingText}>
                {level.charAt(0).toUpperCase() + level.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.formSection}>
        <View style={styles.objectivesHeader}>
          <Text style={styles.label}>Learning Objectives</Text>
          <TouchableOpacity style={styles.addObjectiveButton} onPress={addLearningObjective}>
            <Text style={styles.addObjectiveText}>+ Add</Text>
          </TouchableOpacity>
        </View>
        <TouchableOpacity
          style={styles.objectivesPreview}
          onPress={() => setShowObjectivesModal(true)}
        >
          <Text style={styles.objectivesPreviewText}>
            {learningObjectives.filter(obj => obj.trim()).length} objectives
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Teaching Style</Text>
        <View style={styles.styleContainer}>
          {(['visual', 'auditory', 'kinesthetic', 'reading'] as const).map(style => (
            <TouchableOpacity
              key={style}
              style={[
                styles.styleOption,
                teachingStyle === style && styles.selectedStyleOption,
              ]}
              onPress={() => setTeachingStyle(style)}
            >
              <Text style={styles.styleText}>
                {style.charAt(0).toUpperCase() + style.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Duration (minutes)</Text>
        <TextInput
          style={styles.input}
          value={duration.toString()}
          onChangeText={(text) => setDuration(parseInt(text) || 30)}
          placeholder="30"
          placeholderTextColor="#999"
          keyboardType="numeric"
        />
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Additional Context (Optional)</Text>
        <TextInput
          style={[styles.input, styles.textArea]}
          value={context}
          onChangeText={setContext}
          placeholder="Any additional context or specific requirements..."
          placeholderTextColor="#999"
          multiline
          numberOfLines={4}
        />
      </View>

      <TouchableOpacity
        style={[
          styles.generateButton,
          (!selectedSubject || !selectedGrade || !topic.trim() || isGenerating) && styles.generateButtonDisabled,
        ]}
        onPress={handleGenerateSession}
        disabled={!selectedSubject || !selectedGrade || !topic.trim() || isGenerating}
      >
        {isGenerating ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Text style={styles.generateButtonText}>Generate Session</Text>
        )}
      </TouchableOpacity>
    </ScrollView>
  );

  return (
    <View style={[styles.container, style]}>
      {currentSession ? renderSessionContent() : renderForm()}

      <Modal
        visible={showObjectivesModal}
        animationType="slide"
        presentationStyle="pageSheet"
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Learning Objectives</Text>
            <TouchableOpacity onPress={() => setShowObjectivesModal(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>
          
          <FlatList
            data={learningObjectives}
            keyExtractor={(_, index) => index.toString()}
            renderItem={({ item, index }) => (
              <View style={styles.objectiveItem}>
                <TextInput
                  style={styles.objectiveInput}
                  value={item}
                  onChangeText={(value) => updateLearningObjective(index, value)}
                  placeholder="Enter learning objective..."
                  placeholderTextColor="#999"
                />
                {learningObjectives.length > 1 && (
                  <TouchableOpacity
                    style={styles.removeObjectiveButton}
                    onPress={() => removeLearningObjective(index)}
                  >
                    <Text style={styles.removeObjectiveText}>✕</Text>
                  </TouchableOpacity>
                )}
              </View>
            )}
          />
          
          <TouchableOpacity
            style={styles.modalDoneButton}
            onPress={() => setShowObjectivesModal(false)}
          >
            <Text style={styles.modalDoneButtonText}>Done</Text>
          </TouchableOpacity>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  formContainer: {
    flex: 1,
    padding: 16,
  },
  formTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 24,
    textAlign: 'center',
  },
  formSection: {
    marginBottom: 24,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  subjectGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -4,
  },
  subjectCard: {
    width: '48%',
    marginHorizontal: '1%',
    aspectRatio: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
    marginBottom: 8,
  },
  selectedSubjectCard: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  subjectIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  subjectName: {
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#333',
  },
  gradeContainer: {
    flexDirection: 'column',
  },
  gradeOption: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  selectedGradeOption: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  gradeText: {
    fontSize: 14,
    color: '#333',
  },
  input: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  understandingContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  understandingOption: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginHorizontal: 4,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  selectedUnderstandingOption: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  understandingText: {
    fontSize: 14,
    color: '#333',
  },
  objectivesHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  addObjectiveButton: {
    backgroundColor: '#007AFF',
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  addObjectiveText: {
    color: '#fff',
    fontSize: 12,
  },
  objectivesPreview: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  objectivesPreviewText: {
    fontSize: 14,
    color: '#666',
  },
  styleContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -2,
  },
  styleOption: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 8,
    marginHorizontal: 2,
    marginBottom: 4,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  selectedStyleOption: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  styleText: {
    fontSize: 12,
    color: '#333',
  },
  generateButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
  },
  generateButtonDisabled: {
    backgroundColor: '#ccc',
  },
  generateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  sessionContainer: {
    flex: 1,
    padding: 16,
  },
  sessionHeader: {
    marginBottom: 24,
    alignItems: 'center',
  },
  sessionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
    textAlign: 'center',
  },
  sessionSubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  sessionSection: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  sectionContent: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  },
  exampleItem: {
    backgroundColor: '#f0f8ff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  exampleText: {
    fontSize: 14,
    color: '#333',
    fontStyle: 'italic',
  },
  questionItem: {
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  questionText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  answerText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  difficultyText: {
    fontSize: 12,
    color: '#999',
  },
  resourceItem: {
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  resourceType: {
    fontSize: 12,
    color: '#666',
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  resourceTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  resourceDescription: {
    fontSize: 12,
    color: '#666',
  },
  newSessionButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  newSessionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  modalClose: {
    fontSize: 20,
    color: '#666',
  },
  objectiveItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    marginHorizontal: 16,
  },
  objectiveInput: {
    flex: 1,
    fontSize: 16,
    color: '#333',
    marginRight: 8,
  },
  removeObjectiveButton: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#ff4444',
    justifyContent: 'center',
    alignItems: 'center',
  },
  removeObjectiveText: {
    color: '#fff',
    fontSize: 12,
  },
  modalDoneButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    padding: 16,
    margin: 16,
    alignItems: 'center',
  },
  modalDoneButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});