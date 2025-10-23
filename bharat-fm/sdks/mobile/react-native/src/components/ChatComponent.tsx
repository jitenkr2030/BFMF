/**
 * Chat Component for Bharat AI SDK
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';

import { BharatAIClient } from '../client/BharatAIClient';
import { GenerationRequest, GenerationResponse, StreamingResponse } from '../types/ApiTypes';
import { BharatAIError } from '../errors/BharatAIError';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  isStreaming?: boolean;
}

interface ChatComponentProps {
  /** Bharat AI client instance */
  client: BharatAIClient;
  /** Initial messages */
  initialMessages?: Message[];
  /** Placeholder text */
  placeholder?: string;
  /** Maximum message history */
  maxHistory?: number;
  /** Enable streaming */
  enableStreaming?: boolean;
  ** Chat title */
  title?: string;
  ** On message send callback */
  onMessageSend?: (message: string) => void;
  ** On message receive callback */
  onMessageReceive?: (message: string) => void;
  ** On error callback */
  onError?: (error: BharatAIError) => void;
  ** Custom styles */
  style?: any;
}

export const ChatComponent: React.FC<ChatComponentProps> = ({
  client,
  initialMessages = [],
  placeholder = 'Type a message...',
  maxHistory = 50,
  enableStreaming = true,
  title = 'Bharat AI Chat',
  onMessageSend,
  onMessageReceive,
  onError,
  style,
}) => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  useEffect(() => {
    // Initialize client if not already initialized
    if (!client.isClientInitialized()) {
      client.initialize().catch(error => {
        console.error('Failed to initialize client:', error);
        onError?.(error);
      });
    }
  }, [client]);

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      text: inputText.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    onMessageSend?.(userMessage.text);

    try {
      const aiMessage: Message = {
        id: `ai_${Date.now()}`,
        text: '',
        sender: 'ai',
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages(prev => [...prev, aiMessage]);

      if (enableStreaming) {
        await handleStreamingResponse(aiMessage, inputText.trim());
      } else {
        await handleNormalResponse(aiMessage, inputText.trim());
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const aiError = error instanceof BharatAIError ? error : new BharatAIError('CHAT_ERROR', 'Failed to send message', error);
      onError?.(aiError);
      
      // Show error message
      Alert.alert('Error', 'Failed to send message. Please try again.');
      
      // Remove the streaming message
      setMessages(prev => prev.filter(msg => msg.id !== `ai_${Date.now()}`));
    } finally {
      setIsLoading(false);
    }
  };

  const handleNormalResponse = async (aiMessage: Message, userText: string) => {
    const request: GenerationRequest = {
      prompt: userText,
      maxTokens: 1000,
      temperature: 0.7,
      metadata: {
        operation: 'chat',
        messageHistory: messages.slice(-maxHistory),
      },
    };

    const response = await client.generateText(request);

    setMessages(prev =>
      prev.map(msg =>
        msg.id === aiMessage.id
          ? { ...msg, text: response.text, isStreaming: false }
          : msg
      )
    );

    onMessageReceive?.(response.text);
  };

  const handleStreamingResponse = async (aiMessage: Message, userText: string) => {
    const request: GenerationRequest = {
      prompt: userText,
      maxTokens: 1000,
      temperature: 0.7,
      stream: true,
      metadata: {
        operation: 'chat_stream',
        messageHistory: messages.slice(-maxHistory),
      },
    };

    let accumulatedText = '';

    await client.generateTextStream(
      request,
      (chunk: StreamingResponse) => {
        accumulatedText += chunk.text;
        
        setMessages(prev =>
          prev.map(msg =>
            msg.id === aiMessage.id
              ? { ...msg, text: accumulatedText }
              : msg
          )
        );
      }
    );

    // Mark streaming as complete
    setMessages(prev =>
      prev.map(msg =>
        msg.id === aiMessage.id
          ? { ...msg, isStreaming: false }
          : msg
      )
    );

    onMessageReceive?.(accumulatedText);
  };

  const renderMessage = ({ item }: { item: Message }) => (
    <View
      style={[
        styles.messageContainer,
        item.sender === 'user' ? styles.userMessage : styles.aiMessage,
      ]}
    >
      <Text
        style={[
          styles.messageText,
          item.sender === 'user' ? styles.userMessageText : styles.aiMessageText,
        ]}
      >
        {item.text}
        {item.isStreaming && (
          <Text style={styles.streamingText}>...</Text>
        )}
      </Text>
      <Text style={styles.timestampText}>
        {item.timestamp.toLocaleTimeString()}
      </Text>
    </View>
  );

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <KeyboardAvoidingView
      style={[styles.container, style]}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <TouchableOpacity onPress={clearChat} style={styles.clearButton}>
          <Text style={styles.clearButtonText}>Clear</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.messagesList}
        onContentSizeChange={() => {
          flatListRef.current?.scrollToEnd({ animated: true });
        }}
      />

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={inputText}
          onChangeText={setInputText}
          placeholder={placeholder}
          placeholderTextColor="#999"
          multiline
          maxLength={1000}
          editable={!isLoading}
        />
        
        <TouchableOpacity
          style={[
            styles.sendButton,
            (!inputText.trim() || isLoading) && styles.sendButtonDisabled,
          ]}
          onPress={handleSendMessage}
          disabled={!inputText.trim() || isLoading}
        >
          {isLoading ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.sendButtonText}>Send</Text>
          )}
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  clearButton: {
    padding: 8,
  },
  clearButtonText: {
    color: '#666',
    fontSize: 14,
  },
  messagesList: {
    padding: 16,
  },
  messageContainer: {
    maxWidth: '80%',
    marginVertical: 4,
    padding: 12,
    borderRadius: 16,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
  },
  aiMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 20,
  },
  userMessageText: {
    color: '#fff',
  },
  aiMessageText: {
    color: '#333',
  },
  streamingText: {
    color: '#999',
  },
  timestampText: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
    alignSelf: 'flex-end',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  textInput: {
    flex: 1,
    backgroundColor: '#f0f0f0',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    maxHeight: 100,
    marginRight: 8,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    borderRadius: 20,
    paddingHorizontal: 20,
    paddingVertical: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});