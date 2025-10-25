"""
Bharat-FM Phase 2 Demo

This demo showcases the advanced features implemented in Phase 2:
1. Advanced Model Registry & Versioning
2. Multi-Modal Processing Capabilities

Run this demo to see Phase 2 features in action.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bharat_fm.registry.model_registry import (
    ModelRegistry, ModelMetadata, ModelMetrics, ModelType, ModelStatus, create_model_registry
)
from bharat_fm.multimodal.multimodal_processor import (
    MultiModalProcessor, MultiModalInput, TextContent, ImageContent, AudioContent,
    ProcessingTask, ModalityType, create_multimodal_processor
)


async def demo_model_registry():
    """Demonstrate Model Registry & Versioning capabilities"""
    print("=" * 60)
    print("BHARAT-FM PHASE 2: MODEL REGISTRY & VERSIONING DEMO")
    print("=" * 60)
    
    # Create model registry
    registry = await create_model_registry({
        "storage_path": "./demo_model_registry"
    })
    
    try:
        print("\n1. Registering new models...")
        
        # Register a language model
        lang_metadata = ModelMetadata(
            name="Bharat-Lite-v2",
            description="Enhanced multilingual language model for Indian languages",
            version="2.0.0",
            model_type=ModelType.LANGUAGE,
            framework="pytorch",
            architecture="transformer",
            parameters=1_300_000_000,
            file_size=2_500_000_000,
            created_by="Bharat AI Team",
            created_at=None,  # Will be set automatically
            tags=["multilingual", "indian-languages", "lightweight"],
            dependencies=["torch", "transformers", "tokenizers"],
            config={"max_length": 512, "temperature": 0.7},
            input_format="text",
            output_format="text",
            supported_languages=["hi", "en", "bn", "ta", "te", "mr", "gu"],
            domain="general"
        )
        
        lang_model_id = await registry.register_model(
            lang_metadata, 
            "./demo_models/bharat_lite_v2.pt"  # Simulated file path
        )
        print(f"✓ Registered language model: {lang_model_id}")
        
        # Register a multimodal model
        multimodal_metadata = ModelMetadata(
            name="Bharat-Multi-v1",
            description="Multi-modal model supporting text, image, and audio",
            version="1.0.0",
            model_type=ModelType.MULTIMODAL,
            framework="pytorch",
            architecture="multimodal_transformer",
            parameters=7_000_000_000,
            file_size=13_000_000_000,
            created_by="Bharat AI Team",
            created_at=None,
            tags=["multimodal", "vision", "audio", "text"],
            dependencies=["torch", "transformers", "PIL", "librosa"],
            config={"image_size": 224, "audio_sample_rate": 16000},
            input_format="multimodal",
            output_format="text",
            supported_languages=["hi", "en"],
            domain="general"
        )
        
        multimodal_model_id = await registry.register_model(
            multimodal_metadata,
            "./demo_models/bharat_multi_v1.pt"
        )
        print(f"✓ Registered multimodal model: {multimodal_model_id}")
        
        print("\n2. Adding new versions...")
        
        # Add a new version to the language model
        lang_v2_metadata = ModelMetadata(
            name="Bharat-Lite-v2",
            description="Improved version with better performance",
            version="2.1.0",
            model_type=ModelType.LANGUAGE,
            framework="pytorch",
            architecture="transformer",
            parameters=1_300_000_000,
            file_size=2_600_000_000,
            created_by="Bharat AI Team",
            created_at=None,
            tags=["multilingual", "indian-languages", "lightweight", "improved"],
            dependencies=["torch", "transformers", "tokenizers"],
            config={"max_length": 512, "temperature": 0.7},
            input_format="text",
            output_format="text",
            supported_languages=["hi", "en", "bn", "ta", "te", "mr", "gu", "kn"],
            domain="general"
        )
        
        lang_v2_version_id = await registry.add_version(
            lang_model_id,
            lang_v2_metadata,
            "./demo_models/bharat_lite_v2_1.pt",
            ModelMetrics(
                accuracy=0.92,
                latency=45.0,
                throughput=120.0,
                memory_usage=2800.0,
                cost_per_request=0.0012
            )
        )
        print(f"✓ Added new version: {lang_v2_version_id}")
        
        print("\n3. Updating metrics and status...")
        
        # Update metrics for the first version
        await registry.update_metrics(
            list(await registry.get_model_versions(lang_model_id))[0].version_id,
            ModelMetrics(
                accuracy=0.89,
                latency=52.0,
                throughput=100.0,
                memory_usage=2700.0,
                cost_per_request=0.0015
            )
        )
        
        # Set version status
        await registry.set_version_status(lang_v2_version_id, ModelStatus.PRODUCTION)
        await registry.set_default_version(lang_model_id, lang_v2_version_id)
        print("✓ Updated metrics and set production status")
        
        print("\n4. Deploying models...")
        
        # Deploy the language model
        deployment_id = await registry.deploy_model(
            lang_model_id,
            lang_v2_version_id,
            "http://localhost:8000/api/v1/language",
            "production",
            {"replicas": 3, "gpu": "1"}
        )
        print(f"✓ Deployed model: {deployment_id}")
        
        print("\n5. Querying the registry...")
        
        # List all models
        models = await registry.list_models()
        print(f"Total models in registry: {len(models)}")
        
        for model in models:
            print(f"  - {model['metadata']['name']} v{model['metadata']['version']}")
            print(f"    Type: {model['metadata']['model_type']}")
            print(f"    Versions: {model['versions_count']}")
            if model['default_version']:
                print(f"    Default: {model['default_version']['version_id']}")
            print()
        
        # Search models
        search_results = await registry.search_models("multilingual", limit=5)
        print(f"Search results for 'multilingual': {len(search_results)}")
        for result in search_results:
            print(f"  - {result['metadata']['name']}")
        
        print("\n6. Registry statistics...")
        
        stats = await registry.get_registry_stats()
        print("Registry Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
    finally:
        await registry.stop()
    
    print("\n✅ Model Registry Demo Completed!")


async def demo_multimodal_processing():
    """Demonstrate Multi-Modal Processing capabilities"""
    print("\n" + "=" * 60)
    print("BHARAT-FM PHASE 2: MULTI-MODAL PROCESSING DEMO")
    print("=" * 60)
    
    # Create multi-modal processor
    processor = await create_multimodal_processor({
        "storage_path": "./demo_multimodal_cache"
    })
    
    try:
        print("\n1. Supported capabilities...")
        
        modalities = await processor.get_supported_modalities()
        tasks = await processor.get_supported_tasks()
        
        print(f"Supported modalities: {', '.join(modalities)}")
        print(f"Supported tasks: {', '.join(tasks)}")
        
        print("\n2. Text processing demo...")
        
        # Create text content
        text_content = TextContent(
            content_type=ModalityType.TEXT,
            data="नमस्ते! मैं भारत का AI सहायक हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
            text="नमस्ते! मैं भारत का AI सहायक हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
            language="hi"
        )
        
        # Process text for different tasks
        text_input = MultiModalInput()
        text_input.add_content(text_content)
        
        # Text generation
        text_input.task = ProcessingTask.GENERATION
        result = await processor.process(text_input)
        print(f"Text Generation: {result.result}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        # Text classification
        text_input.task = ProcessingTask.CLASSIFICATION
        result = await processor.process(text_input)
        print(f"Text Classification: {result.result}")
        
        # Text translation
        text_input.task = ProcessingTask.TRANSLATION
        text_input.parameters = {"target_language": "en"}
        result = await processor.process(text_input)
        print(f"Text Translation: {result.result}")
        
        print("\n3. Image processing demo...")
        
        # Create simulated image content
        # In a real scenario, this would be actual image data
        image_data = b"simulated_image_data_for_demo_purposes"
        image_content = ImageContent(
            content_type=ModalityType.IMAGE,
            data=image_data,
            image_bytes=image_data,
            format="png",
            width=800,
            height=600
        )
        
        # Process image
        image_input = MultiModalInput()
        image_input.add_content(image_content)
        image_input.task = ProcessingTask.CLASSIFICATION
        
        result = await processor.process(image_input)
        print(f"Image Classification: {result.result}")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Image feature extraction
        image_input.task = ProcessingTask.EXTRACTION
        result = await processor.process(image_input)
        print(f"Image Features: {json.dumps(result.result, indent=2)}")
        
        print("\n4. Audio processing demo...")
        
        # Create simulated audio content
        audio_data = b"simulated_audio_data_for_demo_purposes"
        audio_content = AudioContent(
            content_type=ModalityType.AUDIO,
            data=audio_data,
            audio_bytes=audio_data,
            format="wav",
            duration=5.2,
            sample_rate=16000,
            channels=1
        )
        
        # Process audio
        audio_input = MultiModalInput()
        audio_input.add_content(audio_content)
        audio_input.task = ProcessingTask.EXTRACTION
        
        result = await processor.process(audio_input)
        print(f"Speech-to-Text: {result.result}")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Audio classification
        audio_input.task = ProcessingTask.CLASSIFICATION
        result = await processor.process(audio_input)
        print(f"Audio Classification: {result.result}")
        
        print("\n5. Multi-modal fusion demo...")
        
        # Create multi-modal input with text and image
        multimodal_input = MultiModalInput()
        multimodal_input.add_content(text_content)
        multimodal_input.add_content(image_content)
        multimodal_input.task = ProcessingTask.FUSION
        
        result = await processor.process(multimodal_input)
        print(f"Multi-modal Fusion Result:")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Modalities processed: {result.metadata.get('modalities_processed', 0)}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        
        print("\n6. Cache statistics...")
        
        cache_stats = await processor.get_cache_stats()
        print("Cache Statistics:")
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
        
    finally:
        await processor.stop()
    
    print("\n✅ Multi-Modal Processing Demo Completed!")


async def demo_integration():
    """Demonstrate integration between Phase 1 and Phase 2 components"""
    print("\n" + "=" * 60)
    print("BHARAT-FM PHASE 2: INTEGRATION DEMO")
    print("=" * 60)
    
    try:
        # Import Phase 1 components
        from bharat_fm.core.inference_engine import create_inference_engine
        from bharat_fm.core.chat_engine import create_chat_engine
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        
        print("\n1. Setting up Phase 1 components...")
        
        # Create inference engine with optimization
        inference_config = {
            "optimization_enabled": True,
            "caching_enabled": True,
            "batching_enabled": True,
            "cost_monitoring_enabled": True
        }
        
        inference_engine = await create_inference_engine(inference_config)
        print("✓ Inference engine created with optimization")
        
        # Create chat engine with memory
        chat_config = {
            "inference": inference_config,
            "memory": {"max_history_length": 100}
        }
        
        chat_engine = await create_chat_engine(chat_config)
        print("✓ Chat engine created with memory")
        
        # Create memory manager
        memory_manager = ConversationMemoryManager()
        await memory_manager.start()
        print("✓ Memory manager started")
        
        print("\n2. Setting up Phase 2 components...")
        
        # Create model registry
        registry = await create_model_registry({
            "storage_path": "./demo_integration_registry"
        })
        print("✓ Model registry created")
        
        # Create multi-modal processor
        processor = await create_multimodal_processor({
            "storage_path": "./demo_integration_cache"
        })
        print("✓ Multi-modal processor created")
        
        print("\n3. Integrated workflow demo...")
        
        # Simulate a user interaction with multi-modal input
        user_id = "demo_user_123"
        
        # Create multi-modal input (text + image)
        text_content = TextContent(
            content_type=ModalityType.TEXT,
            data="Can you help me understand this document?",
            text="Can you help me understand this document?",
            language="en"
        )
        
        image_content = ImageContent(
            content_type=ModalityType.IMAGE,
            data=b"simulated_document_image",
            image_bytes=b"simulated_document_image",
            format="png",
            width=1200,
            height=1600
        )
        
        multimodal_input = MultiModalInput()
        multimodal_input.add_content(text_content)
        multimodal_input.add_content(image_content)
        multimodal_input.task = ProcessingTask.FUSION
        
        # Process with multi-modal processor
        mm_result = await processor.process(multimodal_input)
        print(f"Multi-modal processing completed: {mm_result.success}")
        
        # Use the result in conversation
        if mm_result.success:
            # Start chat session
            session = await chat_engine.start_session(user_id)
            session_id = session["session_id"]
            
            # Generate response based on multi-modal analysis
            response = await chat_engine.generate_response(
                user_id,
                session_id,
                f"I have a document with text and image content. {mm_result.result['fusion_type']} analysis shows multiple modalities."
            )
            
            print(f"Chat response: {response['response']['generated_text']}")
            
            # Store in conversation memory
            await memory_manager.store_exchange(
                user_id,
                session_id,
                "Can you help me understand this document?",
                response['response']['generated_text']
            )
            
            print("✓ Conversation stored in memory")
        
        print("\n4. Performance metrics...")
        
        # Get registry stats
        registry_stats = await registry.get_registry_stats()
        print(f"Registry models: {registry_stats['total_models']}")
        
        # Get cache stats
        cache_stats = await processor.get_cache_stats()
        print(f"Cache entries: {cache_stats['cache_size']}")
        
        # Get inference metrics (simulated)
        print("Inference optimization active")
        print("Conversation memory tracking enabled")
        
    finally:
        # Clean up
        try:
            await registry.stop()
            await processor.stop()
            await memory_manager.stop()
        except:
            pass
    
    print("\n✅ Integration Demo Completed!")


async def main():
    """Main demo function"""
    print("🇮🇳 BHARAT FOUNDATION MODEL FRAMEWORK - PHASE 2 DEMO")
    print("=" * 60)
    print("This demo showcases the advanced features of Phase 2:")
    print("1. Advanced Model Registry & Versioning")
    print("2. Multi-Modal Processing Capabilities")
    print("3. Integration with Phase 1 components")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_model_registry()
        await demo_multimodal_processing()
        await demo_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL PHASE 2 DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nPhase 2 Features Demonstrated:")
        print("✅ Advanced Model Registry & Versioning")
        print("✅ Multi-Modal Processing (Text, Image, Audio)")
        print("✅ Cross-Modal Integration")
        print("✅ Performance Optimization")
        print("✅ Cache Management")
        print("✅ Seamless Phase 1 Integration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())