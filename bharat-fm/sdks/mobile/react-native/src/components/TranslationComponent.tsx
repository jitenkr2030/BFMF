/**
 * Translation Component for Bharat AI SDK
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Picker,
  Alert,
  ScrollView,
} from 'react-native';

import { BharatAIClient } from '../client/BharatAIClient';
import { LanguageAIClient, TranslationRequest, TranslationResponse } from '../domains/LanguageAIClient';
import { BharatAIError } from '../errors/BharatAIError';

interface LanguageOption {
  code: string;
  name: string;
  nativeName: string;
}

interface TranslationComponentProps {
  /** Bharat AI client instance */
  client: BharatAIClient;
  /** Supported languages */
  supportedLanguages?: LanguageOption[];
  /** Placeholder text */
  placeholder?: string;
  ** Enable auto-detection */
  enableAutoDetection?: boolean;
  ** Show confidence score */
  showConfidence?: boolean;
  ** On translation complete callback */
  onTranslationComplete?: (result: TranslationResponse) => void;
  ** On error callback */
  onError?: (error: BharatAIError) => void;
  ** Custom styles */
  style?: any;
}

export const TranslationComponent: React.FC<TranslationComponentProps> = ({
  client,
  supportedLanguages = [
    { code: 'en', name: 'English', nativeName: 'English' },
    { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी' },
    { code: 'bn', name: 'Bengali', nativeName: 'বাংলা' },
    { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்' },
    { code: 'te', name: 'Telugu', nativeName: 'తెలుగు' },
    { code: 'mr', name: 'Marathi', nativeName: 'मराठी' },
    { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી' },
    { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ' },
    { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം' },
    { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ' },
  ],
  placeholder = 'Enter text to translate...',
  enableAutoDetection = true,
  showConfidence = true,
  onTranslationComplete,
  onError,
  style,
}) => {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLanguage, setSourceLanguage] = useState('auto');
  const [targetLanguage, setTargetLanguage] = useState('hi');
  const [isTranslating, setIsTranslating] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const [alternatives, setAlternatives] = useState<string[]>([]);

  const languageClient = new LanguageAIClient(client);

  useEffect(() => {
    // Initialize client if not already initialized
    if (!client.isClientInitialized()) {
      client.initialize().catch(error => {
        console.error('Failed to initialize client:', error);
        onError?.(error);
      });
    }
  }, [client]);

  const handleTranslate = async () => {
    if (!inputText.trim() || isTranslating) return;

    setIsTranslating(true);
    setTranslatedText('');
    setConfidence(null);
    setDetectedLanguage(null);
    setAlternatives([]);

    try {
      const request: TranslationRequest = {
        text: inputText.trim(),
        sourceLanguage: sourceLanguage === 'auto' ? 'en' : sourceLanguage, // Default to English if auto
        targetLanguage: targetLanguage,
        includeConfidence: true,
      };

      const response = await languageClient.translate(request);

      setTranslatedText(response.translatedText);
      setConfidence(response.confidence);
      setDetectedLanguage(response.detectedSourceLanguage || sourceLanguage);
      setAlternatives(response.alternatives || []);

      onTranslationComplete?.(response);
    } catch (error) {
      console.error('Translation error:', error);
      const aiError = error instanceof BharatAIError ? error : new BharatAIError('TRANSLATION_ERROR', 'Failed to translate text', error);
      onError?.(aiError);
      
      Alert.alert('Translation Error', 'Failed to translate text. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  const swapLanguages = () => {
    if (sourceLanguage !== 'auto') {
      const newSource = targetLanguage;
      const newTarget = sourceLanguage;
      setSourceLanguage(newSource);
      setTargetLanguage(newTarget);
      
      // Swap the texts if they exist
      if (translatedText) {
        setInputText(translatedText);
        setTranslatedText(inputText);
      }
    }
  };

  const clearText = () => {
    setInputText('');
    setTranslatedText('');
    setConfidence(null);
    setDetectedLanguage(null);
    setAlternatives([]);
  };

  const copyToClipboard = (text: string) => {
    // In a real implementation, you would use Clipboard API
    Alert.alert('Copied', 'Text copied to clipboard');
  };

  return (
    <View style={[styles.container, style]}>
      <View style={styles.languageSelectors}>
        <View style={styles.languageSelector}>
          <Text style={styles.selectorLabel}>From:</Text>
          <View style={styles.pickerContainer}>
            {enableAutoDetection && (
              <TouchableOpacity
                style={[
                  styles.languageOption,
                  sourceLanguage === 'auto' && styles.selectedLanguage,
                ]}
                onPress={() => setSourceLanguage('auto')}
              >
                <Text style={styles.languageText}>Auto Detect</Text>
              </TouchableOpacity>
            )}
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {supportedLanguages.map(lang => (
                <TouchableOpacity
                  key={`source-${lang.code}`}
                  style={[
                    styles.languageOption,
                    sourceLanguage === lang.code && styles.selectedLanguage,
                  ]}
                  onPress={() => setSourceLanguage(lang.code)}
                >
                  <Text style={styles.languageText}>{lang.name}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>

        <TouchableOpacity style={styles.swapButton} onPress={swapLanguages}>
          <Text style={styles.swapButtonText}>⇄</Text>
        </TouchableOpacity>

        <View style={styles.languageSelector}>
          <Text style={styles.selectorLabel}>To:</Text>
          <View style={styles.pickerContainer}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {supportedLanguages.map(lang => (
                <TouchableOpacity
                  key={`target-${lang.code}`}
                  style={[
                    styles.languageOption,
                    targetLanguage === lang.code && styles.selectedLanguage,
                  ]}
                  onPress={() => setTargetLanguage(lang.code)}
                >
                  <Text style={styles.languageText}>{lang.name}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
      </View>

      <View style={styles.textContainers}>
        <View style={styles.textContainer}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder={placeholder}
            placeholderTextColor="#999"
            multiline
            textAlignVertical="top"
            editable={!isTranslating}
          />
          {inputText.length > 0 && (
            <TouchableOpacity style={styles.clearButton} onPress={clearText}>
              <Text style={styles.clearButtonText}>✕</Text>
            </TouchableOpacity>
          )}
        </View>

        <TouchableOpacity
          style={[
            styles.translateButton,
            (!inputText.trim() || isTranslating) && styles.translateButtonDisabled,
          ]}
          onPress={handleTranslate}
          disabled={!inputText.trim() || isTranslating}
        >
          {isTranslating ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.translateButtonText}>Translate</Text>
          )}
        </TouchableOpacity>

        <View style={styles.textContainer}>
          <TextInput
            style={[styles.textInput, styles.translatedText]}
            value={translatedText}
            placeholder="Translation will appear here..."
            placeholderTextColor="#999"
            multiline
            textAlignVertical="top"
            editable={false}
          />
          {translatedText.length > 0 && (
            <TouchableOpacity
              style={styles.copyButton}
              onPress={() => copyToClipboard(translatedText)}
            >
              <Text style={styles.copyButtonText}>📋</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      {(showConfidence && confidence !== null) && (
        <View style={styles.infoContainer}>
          <Text style={styles.infoText}>
            Confidence: {(confidence * 100).toFixed(1)}%
          </Text>
          {detectedLanguage && (
            <Text style={styles.infoText}>
              Detected: {supportedLanguages.find(l => l.code === detectedLanguage)?.name || detectedLanguage}
            </Text>
          )}
        </View>
      )}

      {alternatives.length > 0 && (
        <View style={styles.alternativesContainer}>
          <Text style={styles.alternativesTitle}>Alternative Translations:</Text>
          {alternatives.map((alt, index) => (
            <TouchableOpacity
              key={index}
              style={styles.alternativeItem}
              onPress={() => setTranslatedText(alt)}
            >
              <Text style={styles.alternativeText}>{alt}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  languageSelectors: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  languageSelector: {
    flex: 1,
  },
  selectorLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
    marginBottom: 8,
  },
  pickerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  languageOption: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginRight: 8,
    borderRadius: 16,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  selectedLanguage: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  languageText: {
    fontSize: 14,
    color: '#333',
  },
  swapButton: {
    padding: 8,
    marginHorizontal: 8,
  },
  swapButtonText: {
    fontSize: 20,
    color: '#666',
  },
  textContainers: {
    flex: 1,
  },
  textContainer: {
    position: 'relative',
    marginBottom: 16,
  },
  textInput: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    fontSize: 16,
    minHeight: 120,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  translatedText: {
    backgroundColor: '#f0f8ff',
  },
  clearButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
  },
  clearButtonText: {
    fontSize: 14,
    color: '#666',
  },
  copyButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  copyButtonText: {
    fontSize: 14,
  },
  translateButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    padding: 12,
    marginVertical: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  translateButtonDisabled: {
    backgroundColor: '#ccc',
  },
  translateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  infoContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 8,
    marginBottom: 16,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
  },
  alternativesContainer: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
  },
  alternativesTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  alternativeItem: {
    padding: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  alternativeText: {
    fontSize: 14,
    color: '#666',
  },
});