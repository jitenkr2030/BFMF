"""
Bharat-FM Phase 3 Demo

This demo showcases the advanced features implemented in Phase 3:
1. Knowledge Graph Integration with semantic reasoning and fact verification
2. AI Development Assistant with code analysis and generation capabilities

Run this demo to see Phase 3 features in action.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bharat_fm.knowledge.knowledge_graph import (
    create_knowledge_graph, NodeType, RelationType, VerificationStatus
)
from bharat_fm.assistant.code_assistant import (
    create_code_assistant, ProgrammingLanguage, CodeIssueType
)


async def demo_knowledge_graph():
    """Demonstrate Knowledge Graph capabilities"""
    print("=" * 60)
    print("BHARAT-FM PHASE 3: KNOWLEDGE GRAPH INTEGRATION DEMO")
    print("=" * 60)
    
    # Create knowledge graph
    kg = await create_knowledge_graph({
        "storage_path": "./demo_knowledge_graph"
    })
    
    try:
        print("\n1. Knowledge graph initialization...")
        
        # Get initial stats
        stats = kg.get_knowledge_stats()
        print(f"Initial knowledge graph stats: {json.dumps(stats, indent=2)}")
        
        print("\n2. Adding knowledge entities...")
        
        # Add AI-related entities
        ai_research_node = kg.nodes["concept_ai"]
        
        ml_node = kg.add_node(kg.nodes["concept_ai"].__class__(
            node_id="concept_ml",
            node_type=NodeType.CONCEPT,
            label="Machine Learning",
            description="Subset of AI that enables systems to learn from data",
            properties={"field": "Computer Science", "emerged": "1950s"},
            confidence=0.9,
            sources=["knowledge_base"]
        ))
        
        dl_node = kg.add_node(kg.nodes["concept_ai"].__class__(
            node_id="concept_dl",
            node_type=NodeType.CONCEPT,
            label="Deep Learning",
            description="Subset of ML using neural networks with multiple layers",
            properties={"field": "Computer Science", "emerged": "2010s"},
            confidence=0.9,
            sources=["knowledge_base"]
        ))
        
        nlp_node = kg.add_node(kg.nodes["concept_ai"].__class__(
            node_id="concept_nlp",
            node_type=NodeType.CONCEPT,
            label="Natural Language Processing",
            description="AI field focused on interaction between computers and human language",
            properties={"field": "Computer Science", "applications": ["translation", "sentiment"]},
            confidence=0.85,
            sources=["knowledge_base"]
        ))
        
        print(f"✓ Added {len(kg.nodes)} knowledge entities")
        
        print("\n3. Creating relationships...")
        
        # Add relationships
        ml_relation = kg.add_edge(kg.edges["edge_india_ai"].__class__(
            edge_id="edge_ai_ml",
            source_id="concept_ai",
            target_id="concept_ml",
            relation_type=RelationType.INCLUDES,
            weight=0.9,
            confidence=0.95,
            sources=["knowledge_base"]
        ))
        
        dl_relation = kg.add_edge(kg.edges["edge_india_ai"].__class__(
            edge_id="edge_ml_dl",
            source_id="concept_ml",
            target_id="concept_dl",
            relation_type=RelationType.INCLUDES,
            weight=0.8,
            confidence=0.9,
            sources=["knowledge_base"]
        ))
        
        nlp_relation = kg.add_edge(kg.edges["edge_india_ai"].__class__(
            edge_id="edge_ai_nlp",
            source_id="concept_ai",
            target_id="concept_nlp",
            relation_type=RelationType.INCLUDES,
            weight=0.8,
            confidence=0.85,
            sources=["knowledge_base"]
        ))
        
        print(f"✓ Created {len(kg.edges)} relationships")
        
        print("\n4. Knowledge extraction from text...")
        
        # Extract knowledge from sample text
        sample_text = """
        Artificial Intelligence is transforming India's digital landscape. 
        Machine Learning, a subset of AI, enables systems to learn from data. 
        Deep Learning uses neural networks for complex pattern recognition. 
        Natural Language Processing helps computers understand human language.
        """
        
        extracted = kg.extract_knowledge_from_text(sample_text)
        print(f"✓ Extracted {len(extracted)} knowledge items from text")
        
        for item in extracted:
            if item["type"] == "node":
                print(f"  - Entity: {item['data']['label']}")
            elif item["type"] == "edge":
                print(f"  - Relation: {item['data']['relation_type']}")
        
        print("\n5. Fact verification...")
        
        # Create and verify facts
        from bharat_fm.knowledge.knowledge_graph import Fact
        
        fact1 = Fact(
            fact_id="fact_ai_ml",
            subject="Artificial Intelligence",
            predicate="includes",
            object="Machine Learning",
            confidence=0.8,
            sources=["text_extraction"]
        )
        
        fact2 = Fact(
            fact_id="fact_india_ai",
            subject="India",
            predicate="develops",
            object="AI Technology",
            confidence=0.7,
            sources=["text_extraction"]
        )
        
        # Verify facts
        verified_fact1 = kg.verify_fact(fact1)
        verified_fact2 = kg.verify_fact(fact2)
        
        print(f"Fact 1: {verified_fact1.subject} {verified_fact1.predicate} {verified_fact1.object}")
        print(f"  Status: {verified_fact1.verification_status.value}")
        print(f"  Confidence: {verified_fact1.confidence:.2f}")
        print(f"  Evidence: {len(verified_fact1.evidence)} items")
        
        print(f"Fact 2: {verified_fact2.subject} {verified_fact2.predicate} {verified_fact2.object}")
        print(f"  Status: {verified_fact2.verification_status.value}")
        print(f"  Confidence: {verified_fact2.confidence:.2f}")
        print(f"  Evidence: {len(verified_fact2.evidence)} items")
        
        print("\n6. Semantic reasoning...")
        
        # Perform semantic reasoning
        reasoning_results = kg.semantic_reasoning("What is the relationship between AI and Deep Learning?")
        
        print("Reasoning results:")
        for result in reasoning_results:
            print(f"  - Path: {' -> '.join(result['path'])}")
            print(f"    Confidence: {result['confidence']:.2f}")
            print(f"    Explanation: {result['explanation']}")
        
        print("\n7. Fact checking statements...")
        
        # Fact check statements
        statements = [
            "AI includes Machine Learning",
            "Deep Learning is a subset of AI",
            "India is located in Asia",
            "Python is a programming language"
        ]
        
        for statement in statements:
            result = kg.fact_check_statement(statement)
            print(f"Statement: {statement}")
            print(f"  Verification: {result['overall_verification']}")
            print(f"  Confidence: {result['overall_confidence']:.2f}")
            print(f"  Fact checks: {len(result['fact_checks'])}")
            print()
        
        print("\n8. Knowledge graph queries...")
        
        # Query the knowledge graph
        queries = [
            ("AI", "entity_search"),
            ("What is the relationship between AI and ML?", "semantic"),
            ("AI includes Machine Learning", "fact_check")
        ]
        
        for query, query_type in queries:
            results = kg.query_knowledge(query, query_type)
            print(f"Query ({query_type}): {query}")
            print(f"  Results: {len(results)}")
            if results:
                print(f"  First result: {str(results[0])[:100]}...")
            print()
        
        print("\n9. Final knowledge graph statistics...")
        
        final_stats = kg.get_knowledge_stats()
        print("Final statistics:")
        for key, value in final_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
    finally:
        await kg.stop()
    
    print("\n✅ Knowledge Graph Demo Completed!")


async def demo_code_assistant():
    """Demonstrate AI Development Assistant capabilities"""
    print("\n" + "=" * 60)
    print("BHARAT-FM PHASE 3: AI DEVELOPMENT ASSISTANT DEMO")
    print("=" * 60)
    
    # Create code assistant
    assistant = await create_code_assistant({
        "storage_path": "./demo_code_assistant"
    })
    
    try:
        print("\n1. Code analysis capabilities...")
        
        # Sample Python code with issues
        python_code = """
import os, sys, math  # Wildcard import issue

def calculate_factorial(n):  # Missing type hints
    # Calculate factorial of a number
    if n < 0:
        return None  # Should raise ValueError
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):  # Inefficient for large n
            result *= i
        return result

def fibonacci(n):  # Missing docstring
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)  # Inefficient recursive approach

# Unused variable
unused_var = "This is never used"

# Bare except clause
try:
    x = 1 / 0
except:
    print("Error occurred")
"""
        
        print("Analyzing Python code...")
        analysis = assistant.analyze_code(python_code, "sample.py")
        
        print(f"Language: {analysis['language']}")
        print(f"Overall quality: {analysis['overall_quality']['level']} ({analysis['overall_quality']['score']:.2f})")
        print(f"Issues found: {len(analysis['issues'])}")
        print(f"Suggestions: {len(analysis['suggestions'])}")
        
        print("\nCode metrics:")
        metrics = analysis['metrics']
        print(f"  Lines of code: {metrics['lines_of_code']}")
        print(f"  Comment lines: {metrics['comment_lines']}")
        print(f"  Complexity score: {metrics['complexity_score']:.1f}")
        print(f"  Maintainability index: {metrics['maintainability_index']:.2f}")
        print(f"  Quality score: {metrics['quality_score']:.2f}")
        
        print("\nDetected issues:")
        for issue in analysis['issues'][:5]:  # Show first 5 issues
            print(f"  - {issue['issue_type']} ({issue['severity']}): {issue['message']}")
            print(f"    Line {issue['line_number']}: {issue['code_snippet']}")
            print(f"    Suggestion: {issue['suggestion']}")
            print()
        
        print("\nCode suggestions:")
        for suggestion in analysis['suggestions'][:3]:  # Show first 3 suggestions
            print(f"  - {suggestion['suggestion_type']}: {suggestion['title']}")
            print(f"    {suggestion['description']}")
            print(f"    Benefits: {', '.join(suggestion['benefits'])}")
            print()
        
        print("\n2. JavaScript code analysis...")
        
        # Sample JavaScript code
        js_code = """
var x = 10;  // Should use let or const
var y = 20;

function addNumbers(a, b) {
    return a + b;
}

function calculateTotal(items) {
    var total = 0;
    for (var i = 0; i < items.length; i++) {
        total += items[i].price;
    }
    console.log("Total calculated:", total);  // Console.log in production
    return total;
}

if (x == y) {  // Should use ===
    console.log("x equals y");
}
"""
        
        print("Analyzing JavaScript code...")
        js_analysis = assistant.analyze_code(js_code, "sample.js")
        
        print(f"Language: {js_analysis['language']}")
        print(f"Overall quality: {js_analysis['overall_quality']['level']} ({js_analysis['overall_quality']['score']:.2f})")
        print(f"Issues found: {len(js_analysis['issues'])}")
        
        print("\nJavaScript issues:")
        for issue in js_analysis['issues']:
            print(f"  - {issue['issue_type']} ({issue['severity']}): {issue['message']}")
            print(f"    Line {issue['line_number']}: {issue['code_snippet']}")
        
        print("\n3. Code generation capabilities...")
        
        # Generate code from prompts
        prompts = [
            ("Create a Python function that adds two numbers", ProgrammingLanguage.PYTHON),
            ("Create a JavaScript function that calculates factorial", ProgrammingLanguage.JAVASCRIPT),
            ("Create a Python class for a Bank Account", ProgrammingLanguage.PYTHON)
        ]
        
        for prompt, language in prompts:
            print(f"\nGenerating {language.value} code for: {prompt}")
            result = assistant.generate_code(prompt, language)
            
            print("Generated code:")
            print(result['generated_code'])
            
            print(f"Analysis confidence: {result['confidence']:.2f}")
            print(f"Generated code quality: {result['analysis']['overall_quality']['level']}")
            
            if result['documentation']:
                print("Generated documentation:")
                for doc in result['documentation'][:2]:  # Show first 2 docs
                    print(f"  - {doc.title}: {doc.content[:100]}...")
        
        print("\n4. Documentation generation...")
        
        # Generate documentation for complex code
        complex_code = """
class DataProcessor:
    def __init__(self, data_source):
        self.data_source = data_source
        self.processed_data = None
    
    def load_data(self):
        \"\"\"Load data from the source.\"\"\"
        # Implementation would go here
        pass
    
    def process_data(self, transformations):
        \"\"\"Process data using provided transformations.\"\"\"
        # Implementation would go here
        pass
    
    def save_results(self, output_path):
        \"\"\"Save processed results to file.\"\"\"
        # Implementation would go here
        pass

def analyze_results(data):
    \"\"\"Analyze processed data and return insights.\"\"\"
    # Implementation would go here
    pass
"""
        
        print("Generating documentation for complex code...")
        docs = assistant.generate_documentation(complex_code, ProgrammingLanguage.PYTHON)
        
        print(f"Generated {len(docs)} documentation items:")
        for doc in docs:
            print(f"  - {doc.title}")
            print(f"    Type: {doc.doc_type}")
            print(f"    Content: {doc.content[:100]}...")
            print()
        
        print("\n5. Code assistant statistics...")
        
        stats = assistant.get_assistant_stats()
        print("Code Assistant Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
    finally:
        await assistant.stop()
    
    print("\n✅ Code Assistant Demo Completed!")


async def demo_integration():
    """Demonstrate integration between Phase 3 and previous phases"""
    print("\n" + "=" * 60)
    print("BHARAT-FM PHASE 3: INTEGRATION DEMO")
    print("=" * 60)
    
    try:
        # Import components from all phases
        from bharat_fm.core.chat_engine import create_chat_engine
        from bharat_fm.multimodal.multimodal_processor import create_multimodal_processor, TextContent
        from bharat_fm.knowledge.knowledge_graph import create_knowledge_graph
        from bharat_fm.assistant.code_assistant import create_code_assistant
        
        print("\n1. Setting up integrated system...")
        
        # Create Phase 1 components
        chat_engine = await create_chat_engine({
            "inference": {"optimization_enabled": True},
            "memory": {"max_history_length": 100}
        })
        print("✓ Chat engine created")
        
        # Create Phase 2 components
        processor = await create_multimodal_processor({
            "storage_path": "./demo_integration_cache"
        })
        print("✓ Multi-modal processor created")
        
        # Create Phase 3 components
        kg = await create_knowledge_graph({
            "storage_path": "./demo_integration_kg"
        })
        print("✓ Knowledge graph created")
        
        assistant = await create_code_assistant({
            "storage_path": "./demo_integration_assistant"
        })
        print("✓ Code assistant created")
        
        print("\n2. Integrated workflow demo...")
        
        user_id = "demo_developer_123"
        
        # Start chat session
        session = await chat_engine.start_session(user_id)
        session_id = session["session_id"]
        
        # Simulate a developer asking for help with AI development
        user_query = "I need help creating an AI model for text classification in Python"
        
        # Process with multi-modal processor
        text_content = TextContent(
            content_type="text",
            data=user_query,
            text=user_query,
            language="en"
        )
        
        from bharat_fm.multimodal.multimodal_processor import MultiModalInput, ProcessingTask
        text_input = MultiModalInput()
        text_input.add_content(text_content)
        text_input.task = ProcessingTask.ANALYSIS
        
        mm_result = await processor.process(text_input)
        
        # Use knowledge graph for semantic understanding
        kg_results = kg.query_knowledge(user_query, "semantic")
        
        # Generate code using code assistant
        code_result = assistant.generate_code(
            "Create a Python class for text classification using machine learning",
            ProgrammingLanguage.PYTHON,
            context=user_query
        )
        
        # Generate comprehensive response
        response_parts = []
        
        if kg_results:
            response_parts.append("Based on my knowledge graph analysis:")
            for result in kg_results[:2]:
                if "explanation" in result:
                    response_parts.append(f"- {result['explanation']}")
        
        response_parts.append(f"\nHere's a Python class for text classification:")
        response_parts.append(f"```python\n{code_result['generated_code']}\n```")
        
        if code_result['analysis']['issues']:
            response_parts.append(f"\nI found {len(code_result['analysis']['issues'])} potential issues in the generated code:")
            for issue in code_result['analysis']['issues'][:2]:
                response_parts.append(f"- {issue['message']} (line {issue['line_number']})")
        
        if code_result['analysis']['suggestions']:
            response_parts.append(f"\nSuggestions for improvement:")
            for suggestion in code_result['analysis']['suggestions'][:2]:
                response_parts.append(f"- {suggestion['title']}: {suggestion['description']}")
        
        comprehensive_response = "\n".join(response_parts)
        
        # Use chat engine to deliver the response
        chat_response = await chat_engine.generate_response(
            user_id, session_id, comprehensive_response
        )
        
        print("Integrated system response:")
        print("=" * 40)
        print(chat_response['response']['generated_text'])
        print("=" * 40)
        
        print("\n3. Advanced fact-checking and code analysis...")
        
        # Create a statement about code quality
        code_statement = "Python code with high complexity is harder to maintain"
        
        # Fact-check the statement
        fact_check = kg.fact_check_statement(code_statement)
        
        # Analyze code quality
        sample_code = """
def complex_function(data):
    result = []
    for item in data:
        if item is not None:
            if isinstance(item, str):
                if len(item) > 0:
                    if item[0].isupper():
                        result.append(item.upper())
                    else:
                        result.append(item.lower())
                else:
                    result.append("")
            else:
                result.append(str(item))
    return result
"""
        
        code_analysis = assistant.analyze_code(sample_code)
        
        print(f"Fact check: '{code_statement}'")
        print(f"  Verification: {fact_check['overall_verification']}")
        print(f"  Confidence: {fact_check['overall_confidence']:.2f}")
        
        print(f"\nCode analysis results:")
        print(f"  Quality: {code_analysis['overall_quality']['level']} ({code_analysis['overall_quality']['score']:.2f})")
        print(f"  Complexity: {code_analysis['metrics']['complexity_score']:.1f}")
        print(f"  Issues: {len(code_analysis['issues'])}")
        
        print("\n4. Performance metrics across all components...")
        
        # Get statistics from all components
        kg_stats = kg.get_knowledge_stats()
        assistant_stats = assistant.get_assistant_stats()
        
        print("Knowledge Graph Stats:")
        print(f"  Nodes: {kg_stats['nodes']}")
        print(f"  Edges: {kg_stats['edges']}")
        print(f"  Facts: {kg_stats['facts']}")
        
        print("\nCode Assistant Stats:")
        print(f"  Total analyses: {assistant_stats['total_analyses']}")
        print(f"  Total issues: {assistant_stats['total_issues']}")
        print(f"  Total suggestions: {assistant_stats['total_suggestions']}")
        
        print("\n✅ Integrated system is working correctly!")
        
    finally:
        # Clean up
        try:
            await kg.stop()
            await assistant.stop()
            await processor.stop()
        except:
            pass
    
    print("\n✅ Integration Demo Completed!")


async def main():
    """Main demo function"""
    print("🇮🇳 BHARAT FOUNDATION MODEL FRAMEWORK - PHASE 3 DEMO")
    print("=" * 60)
    print("This demo showcases the advanced features of Phase 3:")
    print("1. Knowledge Graph Integration with semantic reasoning")
    print("2. AI Development Assistant with code analysis")
    print("3. Integration with Phase 1 & 2 components")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_knowledge_graph()
        await demo_code_assistant()
        await demo_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL PHASE 3 DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nPhase 3 Features Demonstrated:")
        print("✅ Knowledge Graph Integration")
        print("✅ Semantic Reasoning & Fact Verification")
        print("✅ AI Development Assistant")
        print("✅ Code Analysis & Generation")
        print("✅ Documentation Generation")
        print("✅ Multi-Phase Integration")
        print("✅ Advanced AI Capabilities")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())