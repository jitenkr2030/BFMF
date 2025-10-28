/**
 * RTI Assistant Component for Bharat AI SDK
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
import { GovernanceAIClient, RTIRequest, RTIResponse } from '../domains/GovernanceAIClient';
import { BharatAIError } from '../errors/BharatAIError';

interface PublicAuthority {
  id: string;
  name: string;
  description: string;
  category: string;
}

interface RTIAssistantComponentProps {
  /** Bharat AI client instance */
  client: BharatAIClient;
  /** Available public authorities */
  publicAuthorities?: PublicAuthority[];
  /** Common RTI categories */
  categories?: string[];
  ** On RTI generated callback */
  onRTIGenerated?: (rti: RTIResponse) => void;
  ** On error callback */
  onError?: (error: BharatAIError) => void;
  ** Custom styles */
  style?: any;
}

export const RTIAssistantComponent: React.FC<RTIAssistantComponentProps> = ({
  client,
  publicAuthorities = [
    {
      id: 'mha',
      name: 'Ministry of Home Affairs',
      description: 'Internal security, border management, etc.',
      category: 'Central Government',
    },
    {
      id: 'mof',
      name: 'Ministry of Finance',
      description: 'Financial policies, taxation, budget, etc.',
      category: 'Central Government',
    },
    {
      id: 'moe',
      name: 'Ministry of Education',
      description: 'Education policies, schools, universities, etc.',
      category: 'Central Government',
    },
    {
      id: 'moh',
      name: 'Ministry of Health',
      description: 'Health policies, hospitals, medical services, etc.',
      category: 'Central Government',
    },
    {
      id: 'state_admin',
      name: 'State Administration',
      description: 'State-level governance and services',
      category: 'State Government',
    },
    {
      id: 'municipal',
      name: 'Municipal Corporation',
      description: 'Local civic services and administration',
      category: 'Local Government',
    },
  ],
  categories = [
    'Administrative',
    'Financial',
    'Personal',
    'Policy',
    'Service Delivery',
    'Infrastructure',
    'Legal',
    'Environmental',
  ],
  onRTIGenerated,
  onError,
  style,
}) => {
  const [selectedAuthority, setSelectedAuthority] = useState<PublicAuthority | null>(null);
  const [subject, setSubject] = useState('');
  const [informationSought, setInformationSought] = useState('');
  const [applicantName, setApplicantName] = useState('');
  const [applicantAddress, setApplicantAddress] = useState('');
  const [applicantEmail, setApplicantEmail] = useState('');
  const [applicantPhone, setApplicantPhone] = useState('');
  const [priority, setPriority] = useState<'normal' | 'urgent' | 'emergency'>('normal');
  const [category, setCategory] = useState('');
  const [context, setContext] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentRTI, setCurrentRTI] = useState<RTIResponse | null>(null);
  const [showAuthorityModal, setShowAuthorityModal] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

  const governanceClient = new GovernanceAIClient(client);

  useEffect(() => {
    // Initialize client if not already initialized
    if (!client.isClientInitialized()) {
      client.initialize().catch(error => {
        console.error('Failed to initialize client:', error);
        onError?.(error);
      });
    }
  }, [client]);

  const handleGenerateRTI = async () => {
    if (!selectedAuthority || !subject.trim() || !informationSought.trim() || !applicantName.trim() || !applicantAddress.trim()) {
      Alert.alert('Missing Information', 'Please fill in all required fields.');
      return;
    }

    setIsGenerating(true);

    try {
      const request: RTIRequest = {
        subject: subject.trim(),
        publicAuthority: selectedAuthority.name,
        informationSought: informationSought.trim(),
        applicantDetails: {
          name: applicantName.trim(),
          address: applicantAddress.trim(),
          email: applicantEmail.trim() || undefined,
          phone: applicantPhone.trim() || undefined,
        },
        priority,
        category: category.trim() || undefined,
        context: context.trim() || undefined,
      };

      const response = await governanceClient.generateRTI(request);
      
      setCurrentRTI(response);
      setShowPreview(true);
      onRTIGenerated?.(response);
    } catch (error) {
      console.error('RTI generation error:', error);
      const aiError = error instanceof BharatAIError ? error : new BharatAIError('RTI_ERROR', 'Failed to generate RTI application', error);
      onError?.(aiError);
      
      Alert.alert('Generation Error', 'Failed to generate RTI application. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const resetForm = () => {
    setSelectedAuthority(null);
    setSubject('');
    setInformationSought('');
    setApplicantName('');
    setApplicantAddress('');
    setApplicantEmail('');
    setApplicantPhone('');
    setPriority('normal');
    setCategory('');
    setContext('');
    setCurrentRTI(null);
    setShowPreview(false);
  };

  const downloadRTI = () => {
    if (!currentRTI) return;
    
    // In a real implementation, this would generate and download a PDF
    Alert.alert('Download', 'RTI application downloaded successfully!');
  };

  const shareRTI = () => {
    if (!currentRTI) return;
    
    // In a real implementation, this would share the RTI application
    Alert.alert('Share', 'RTI application shared successfully!');
  };

  const renderRTIPreview = () => {
    if (!currentRTI) return null;

    return (
      <ScrollView style={styles.previewContainer}>
        <View style={styles.previewHeader}>
          <Text style={styles.previewTitle}>RTI Application Preview</Text>
          <View style={styles.previewActions}>
            <TouchableOpacity style={styles.previewButton} onPress={downloadRTI}>
              <Text style={styles.previewButtonText}>📥 Download</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.previewButton} onPress={shareRTI}>
              <Text style={styles.previewButtonText}>📤 Share</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.previewContent}>
          <View style={styles.previewSection}>
            <Text style={styles.previewLabel}>To:</Text>
            <Text style={styles.previewValue}>{selectedAuthority?.name}</Text>
            <Text style={styles.previewSubValue}>{selectedAuthority?.description}</Text>
          </View>

          <View style={styles.previewSection}>
            <Text style={styles.previewLabel}>Subject:</Text>
            <Text style={styles.previewValue}>{subject}</Text>
          </View>

          <View style={styles.previewSection}>
            <Text style={styles.previewLabel}>From:</Text>
            <Text style={styles.previewValue}>{applicantName}</Text>
            <Text style={styles.previewSubValue}>{applicantAddress}</Text>
            {applicantEmail && (
              <Text style={styles.previewSubValue}>Email: {applicantEmail}</Text>
            )}
            {applicantPhone && (
              <Text style={styles.previewSubValue}>Phone: {applicantPhone}</Text>
            )}
          </View>

          <View style={styles.previewSection}>
            <Text style={styles.previewLabel}>Information Sought:</Text>
            <Text style={styles.previewValue}>{informationSought}</Text>
          </View>

          <View style={styles.previewSection}>
            <Text style={styles.previewLabel}>Application Text:</Text>
            <Text style={styles.previewApplication}>{currentRTI.applicationText}</Text>
          </View>

          {currentRTI.suggestions && currentRTI.suggestions.length > 0 && (
            <View style={styles.previewSection}>
              <Text style={styles.previewLabel}>Suggestions:</Text>
              {currentRTI.suggestions.map((suggestion, index) => (
                <Text key={index} style={styles.previewSuggestion}>• {suggestion}</Text>
              ))}
            </View>
          )}

          {currentRTI.complianceChecklist && currentRTI.complianceChecklist.length > 0 && (
            <View style={styles.previewSection}>
              <Text style={styles.previewLabel}>Compliance Checklist:</Text>
              {currentRTI.complianceChecklist.map((item, index) => (
                <Text key={index} style={styles.previewChecklist}>☐ {item}</Text>
              ))}
            </View>
          )}

          {currentRTI.relevantSections && currentRTI.relevantSections.length > 0 && (
            <View style={styles.previewSection}>
              <Text style={styles.previewLabel}>Relevant RTI Act Sections:</Text>
              {currentRTI.relevantSections.map((section, index) => (
                <Text key={index} style={styles.previewSection}>• {section}</Text>
              ))}
            </View>
          )}

          <View style={styles.previewFooter}>
            <Text style={styles.previewConfidence}>
              Confidence: {(currentRTI.confidence * 100).toFixed(1)}%
            </Text>
            {currentRTI.estimatedProcessingTime && (
              <Text style={styles.previewProcessingTime}>
                Estimated Processing Time: {currentRTI.estimatedProcessingTime}
              </Text>
            )}
          </View>
        </View>

        <View style={styles.previewActionsBottom}>
          <TouchableOpacity style={styles.editButton} onPress={() => setShowPreview(false)}>
            <Text style={styles.editButtonText}>Edit</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.newButton} onPress={resetForm}>
            <Text style={styles.newButtonText}>New RTI</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    );
  };

  const renderForm = () => (
    <ScrollView style={styles.formContainer}>
      <Text style={styles.formTitle}>RTI Application Assistant</Text>
      <Text style={styles.formSubtitle}>Generate a Right to Information application under RTI Act, 2005</Text>

      <View style={styles.formSection}>
        <Text style={styles.label}>Public Authority *</Text>
        <TouchableOpacity
          style={[
            styles.authoritySelector,
            !selectedAuthority && styles.placeholderSelector,
          ]}
          onPress={() => setShowAuthorityModal(true)}
        >
          <Text style={[
            styles.authorityText,
            !selectedAuthority && styles.placeholderText,
          ]}>
            {selectedAuthority ? selectedAuthority.name : 'Select Public Authority'}
          </Text>
          <Text style={styles.selectorArrow}>▼</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Subject *</Text>
        <TextInput
          style={styles.input}
          value={subject}
          onChangeText={setSubject}
          placeholder="Brief subject of your RTI application"
          placeholderTextColor="#999"
        />
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Information Sought *</Text>
        <TextInput
          style={[styles.input, styles.textArea]}
          value={informationSought}
          onChangeText={setInformationSought}
          placeholder="Detailed description of information you are seeking..."
          placeholderTextColor="#999"
          multiline
          numberOfLines={6}
          textAlignVertical="top"
        />
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Applicant Details</Text>
        <View style={styles.applicantDetails}>
          <TextInput
            style={[styles.input, styles.applicantInput]}
            value={applicantName}
            onChangeText={setApplicantName}
            placeholder="Full Name *"
            placeholderTextColor="#999"
          />
          <TextInput
            style={[styles.input, styles.applicantInput]}
            value={applicantEmail}
            onChangeText={setApplicantEmail}
            placeholder="Email Address"
            placeholderTextColor="#999"
            keyboardType="email-address"
          />
          <TextInput
            style={[styles.input, styles.applicantInput]}
            value={applicantPhone}
            onChangeText={setApplicantPhone}
            placeholder="Phone Number"
            placeholderTextColor="#999"
            keyboardType="phone-pad"
          />
        </View>
        <TextInput
          style={[styles.input, styles.textArea]}
          value={applicantAddress}
          onChangeText={setApplicantAddress}
          placeholder="Complete Address *"
          placeholderTextColor="#999"
          multiline
          numberOfLines={3}
          textAlignVertical="top"
        />
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Priority</Text>
        <View style={styles.priorityContainer}>
          {(['normal', 'urgent', 'emergency'] as const).map(level => (
            <TouchableOpacity
              key={level}
              style={[
                styles.priorityOption,
                priority === level && styles.selectedPriorityOption,
              ]}
              onPress={() => setPriority(level)}
            >
              <Text style={styles.priorityText}>
                {level.charAt(0).toUpperCase() + level.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Category</Text>
        <View style={styles.categoryContainer}>
          {categories.map(cat => (
            <TouchableOpacity
              key={cat}
              style={[
                styles.categoryOption,
                category === cat && styles.selectedCategoryOption,
              ]}
              onPress={() => setCategory(cat)}
            >
              <Text style={styles.categoryText}>{cat}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.formSection}>
        <Text style={styles.label}>Additional Context (Optional)</Text>
        <TextInput
          style={[styles.input, styles.textArea]}
          value={context}
          onChangeText={setContext}
          placeholder="Any additional context or background information..."
          placeholderTextColor="#999"
          multiline
          numberOfLines={4}
          textAlignVertical="top"
        />
      </View>

      <TouchableOpacity
        style={[
          styles.generateButton,
          (!selectedAuthority || !subject.trim() || !informationSought.trim() || !applicantName.trim() || !applicantAddress.trim() || isGenerating) && styles.generateButtonDisabled,
        ]}
        onPress={handleGenerateRTI}
        disabled={!selectedAuthority || !subject.trim() || !informationSought.trim() || !applicantName.trim() || !applicantAddress.trim() || isGenerating}
      >
        {isGenerating ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Text style={styles.generateButtonText}>Generate RTI Application</Text>
        )}
      </TouchableOpacity>
    </ScrollView>
  );

  return (
    <View style={[styles.container, style]}>
      {showPreview && currentRTI ? renderRTIPreview() : renderForm()}

      <Modal
        visible={showAuthorityModal}
        animationType="slide"
        presentationStyle="pageSheet"
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Public Authority</Text>
            <TouchableOpacity onPress={() => setShowAuthorityModal(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>
          
          <FlatList
            data={publicAuthorities}
            keyExtractor={item => item.id}
            renderItem={({ item }) => (
              <TouchableOpacity
                style={styles.authorityItem}
                onPress={() => {
                  setSelectedAuthority(item);
                  setShowAuthorityModal(false);
                }}
              >
                <View style={styles.authorityItemContent}>
                  <Text style={styles.authorityItemName}>{item.name}</Text>
                  <Text style={styles.authorityItemDescription}>{item.description}</Text>
                  <Text style={styles.authorityItemCategory}>{item.category}</Text>
                </View>
              </TouchableOpacity>
            )}
          />
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
    marginBottom: 8,
    textAlign: 'center',
  },
  formSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 24,
    textAlign: 'center',
  },
  formSection: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  authoritySelector: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  placeholderSelector: {
    backgroundColor: '#f9f9f9',
  },
  authorityText: {
    fontSize: 16,
    color: '#333',
    flex: 1,
  },
  placeholderText: {
    color: '#999',
  },
  selectorArrow: {
    fontSize: 14,
    color: '#666',
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
    height: 100,
    textAlignVertical: 'top',
  },
  applicantDetails: {
    marginBottom: 8,
  },
  applicantInput: {
    marginBottom: 8,
  },
  priorityContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  priorityOption: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginHorizontal: 4,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  selectedPriorityOption: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  priorityText: {
    fontSize: 14,
    color: '#333',
  },
  categoryContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -2,
  },
  categoryOption: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 8,
    marginHorizontal: 2,
    marginBottom: 4,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  selectedCategoryOption: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  categoryText: {
    fontSize: 12,
    color: '#333',
  },
  generateButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  generateButtonDisabled: {
    backgroundColor: '#ccc',
  },
  generateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  previewContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  previewHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderBottomWidth: 1,
    borderBottomColor: '#dee2e6',
  },
  previewTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  previewActions: {
    flexDirection: 'row',
  },
  previewButton: {
    backgroundColor: '#007AFF',
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginLeft: 8,
  },
  previewButtonText: {
    color: '#fff',
    fontSize: 12,
  },
  previewContent: {
    padding: 16,
  },
  previewSection: {
    marginBottom: 20,
  },
  previewLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  previewValue: {
    fontSize: 16,
    color: '#333',
    marginBottom: 2,
  },
  previewSubValue: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  previewApplication: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#007AFF',
  },
  previewSuggestion: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  previewChecklist: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  previewSection: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  previewFooter: {
    marginTop: 20,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: '#dee2e6',
  },
  previewConfidence: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  previewProcessingTime: {
    fontSize: 12,
    color: '#666',
  },
  previewActionsBottom: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderTopWidth: 1,
    borderTopColor: '#dee2e6',
  },
  editButton: {
    flex: 1,
    backgroundColor: '#6c757d',
    borderRadius: 8,
    padding: 12,
    marginRight: 8,
    alignItems: 'center',
  },
  editButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  newButton: {
    flex: 1,
    backgroundColor: '#007AFF',
    borderRadius: 8,
    padding: 12,
    marginLeft: 8,
    alignItems: 'center',
  },
  newButtonText: {
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
  authorityItem: {
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  authorityItemContent: {
    padding: 16,
  },
  authorityItemName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  authorityItemDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  authorityItemCategory: {
    fontSize: 12,
    color: '#999',
    textTransform: 'uppercase',
  },
});