"""
AI Development Assistant for Bharat-FM Phase 3

This module provides comprehensive code analysis and generation capabilities including:
- Code analysis and quality assessment
- Code generation and completion
- Bug detection and fixing
- Code optimization suggestions
- Documentation generation
- Multi-language support
"""

import asyncio
import json
import re
import ast
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import pickle
from pathlib import Path
import tempfile
import subprocess
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPLUSPLUS = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"


class CodeIssueType(Enum):
    """Types of code issues"""
    SYNTAX_ERROR = "syntax_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_ISSUE = "security_issue"
    STYLE_VIOLATION = "style_violation"
    DOCUMENTATION_ISSUE = "documentation_issue"
    DEPENDENCY_ISSUE = "dependency_issue"
    TESTING_ISSUE = "testing_issue"


class CodeQuality(Enum):
    """Code quality levels"""
    EXCELLENT = 0.9
    GOOD = 0.7
    FAIR = 0.5
    POOR = 0.3
    CRITICAL = 0.1


@dataclass
class CodeIssue:
    """Code issue detected during analysis"""
    issue_id: str
    issue_type: CodeIssueType
    severity: str  # low, medium, high, critical
    message: str
    line_number: int
    column_number: int = 0
    code_snippet: str = ""
    suggestion: str = ""
    confidence: float = 0.5
    file_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['issue_type'] = self.issue_type.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeIssue':
        """Create from dictionary"""
        data = data.copy()
        data['issue_type'] = CodeIssueType(data['issue_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class CodeMetrics:
    """Code metrics and statistics"""
    file_path: str
    language: ProgrammingLanguage
    lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    test_coverage: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['language'] = self.language.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeMetrics':
        """Create from dictionary"""
        data = data.copy()
        data['language'] = ProgrammingLanguage(data['language'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class CodeSuggestion:
    """Code improvement suggestion"""
    suggestion_id: str
    suggestion_type: str  # optimization, refactoring, documentation, etc.
    title: str
    description: str
    original_code: str
    suggested_code: str
    confidence: float = 0.5
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    file_path: str = ""
    line_number: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSuggestion':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class Documentation:
    """Generated documentation"""
    doc_id: str
    doc_type: str  # function_doc, class_doc, module_doc, etc.
    title: str
    content: str
    code_element: str
    file_path: str = ""
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['language'] = self.language.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Documentation':
        """Create from dictionary"""
        data = data.copy()
        data['language'] = ProgrammingLanguage(data['language'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class CodeAssistant:
    """AI Development Assistant with code analysis and generation capabilities"""
    
    def __init__(self, storage_path: str = "./code_assistant"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Analysis results storage
        self.issues: Dict[str, CodeIssue] = {}
        self.metrics: Dict[str, CodeMetrics] = {}
        self.suggestions: Dict[str, CodeSuggestion] = {}
        self.documentation: Dict[str, Documentation] = {}
        
        # Storage files
        self.issues_file = self.storage_path / "issues.pkl"
        self.metrics_file = self.storage_path / "metrics.pkl"
        self.suggestions_file = self.storage_path / "suggestions.pkl"
        self.documentation_file = self.storage_path / "documentation.pkl"
        
        # Load existing data
        self._load_data()
        
        # Code patterns and rules
        self.code_patterns = self._initialize_code_patterns()
        
        logger.info(f"CodeAssistant initialized with storage path: {storage_path}")
    
    async def start(self):
        """Start the code assistant"""
        logger.info("CodeAssistant started")
    
    async def stop(self):
        """Stop the code assistant and save data"""
        self._save_data()
        logger.info("CodeAssistant stopped")
    
    def _load_data(self):
        """Load data from disk"""
        try:
            if self.issues_file.exists():
                with open(self.issues_file, 'rb') as f:
                    self.issues = pickle.load(f)
            
            if self.metrics_file.exists():
                with open(self.metrics_file, 'rb') as f:
                    self.metrics = pickle.load(f)
            
            if self.suggestions_file.exists():
                with open(self.suggestions_file, 'rb') as f:
                    self.suggestions = pickle.load(f)
            
            if self.documentation_file.exists():
                with open(self.documentation_file, 'rb') as f:
                    self.documentation = pickle.load(f)
            
            logger.info(f"Loaded {len(self.issues)} issues, {len(self.metrics)} metrics, {len(self.suggestions)} suggestions, {len(self.documentation)} documentation")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to disk"""
        try:
            with open(self.issues_file, 'wb') as f:
                pickle.dump(self.issues, f)
            
            with open(self.metrics_file, 'wb') as f:
                pickle.dump(self.metrics, f)
            
            with open(self.suggestions_file, 'wb') as f:
                pickle.dump(self.suggestions, f)
            
            with open(self.documentation_file, 'wb') as f:
                pickle.dump(self.documentation, f)
                
            logger.info("Code assistant data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = int(time.time())
        random_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_hash}"
    
    def _initialize_code_patterns(self):
        """Initialize code patterns and rules for analysis"""
        return {
            "python": {
                "anti_patterns": [
                    {
                        "pattern": r"import\s+\*",
                        "type": "style_violation",
                        "severity": "medium",
                        "message": "Avoid using wildcard imports",
                        "suggestion": "Import specific modules instead"
                    },
                    {
                        "pattern": r"except\s*:",
                        "type": "logic_error",
                        "severity": "high",
                        "message": "Bare except clause catches all exceptions",
                        "suggestion": "Specify the exception type"
                    },
                    {
                        "pattern": r"len\(range\(",
                        "type": "performance_issue",
                        "severity": "low",
                        "message": "Using len(range()) is inefficient",
                        "suggestion": "Use range() directly in loops"
                    }
                ],
                "best_practices": [
                    {
                        "pattern": r"def\s+\w+\s*\([^)]*:\s*str[^)]*\)",
                        "type": "style_violation",
                        "severity": "low",
                        "message": "Type hints should use proper syntax",
                        "suggestion": "Use '-> str' for return type hints"
                    }
                ]
            },
            "javascript": {
                "anti_patterns": [
                    {
                        "pattern": r"var\s+",
                        "type": "style_violation",
                        "severity": "medium",
                        "message": "Use 'let' or 'const' instead of 'var'",
                        "suggestion": "Prefer 'const' for constants and 'let' for variables"
                    },
                    {
                        "pattern": r"==\s*[^=]",
                        "type": "logic_error",
                        "severity": "high",
                        "message": "Use strict equality (===) instead of loose equality (==)",
                        "suggestion": "Use '===' for strict equality comparison"
                    }
                ]
            }
        }
    
    def detect_language(self, code: str, file_path: str = "") -> ProgrammingLanguage:
        """Detect programming language from code or file extension"""
        if file_path:
            ext = Path(file_path).suffix.lower()
            ext_map = {
                ".py": ProgrammingLanguage.PYTHON,
                ".js": ProgrammingLanguage.JAVASCRIPT,
                ".ts": ProgrammingLanguage.TYPESCRIPT,
                ".java": ProgrammingLanguage.JAVA,
                ".cpp": ProgrammingLanguage.CPLUSPLUS,
                ".cs": ProgrammingLanguage.CSHARP,
                ".go": ProgrammingLanguage.GO,
                ".rs": ProgrammingLanguage.RUST,
                ".html": ProgrammingLanguage.HTML,
                ".css": ProgrammingLanguage.CSS,
                ".sql": ProgrammingLanguage.SQL,
                ".sh": ProgrammingLanguage.BASH
            }
            
            if ext in ext_map:
                return ext_map[ext]
        
        # Try to detect from code content
        if "import " in code and "def " in code:
            return ProgrammingLanguage.PYTHON
        elif "function " in code and "var " in code:
            return ProgrammingLanguage.JAVASCRIPT
        elif "public class " in code:
            return ProgrammingLanguage.JAVA
        elif "#include " in code:
            return ProgrammingLanguage.CPLUSPLUS
        elif "package main" in code:
            return ProgrammingLanguage.GO
        elif "fn main()" in code:
            return ProgrammingLanguage.RUST
        
        return ProgrammingLanguage.PYTHON  # Default
    
    def analyze_code(self, code: str, file_path: str = "") -> Dict[str, Any]:
        """Analyze code for issues, metrics, and suggestions"""
        language = self.detect_language(code, file_path)
        
        # Calculate metrics
        metrics = self._calculate_metrics(code, language, file_path)
        
        # Detect issues
        issues = self._detect_issues(code, language, file_path)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(code, language, metrics, issues, file_path)
        
        # Store results
        metrics_id = self._generate_id("metrics")
        self.metrics[metrics_id] = metrics
        
        for issue in issues:
            issue_id = self._generate_id("issue")
            self.issues[issue_id] = issue
        
        for suggestion in suggestions:
            suggestion_id = self._generate_id("suggestion")
            self.suggestions[suggestion_id] = suggestion
        
        self._save_data()
        
        return {
            "language": language.value,
            "metrics": metrics.to_dict(),
            "issues": [issue.to_dict() for issue in issues],
            "suggestions": [suggestion.to_dict() for suggestion in suggestions],
            "overall_quality": self._calculate_overall_quality(metrics, issues),
            "analysis_id": metrics_id
        }
    
    def _calculate_metrics(self, code: str, language: ProgrammingLanguage, file_path: str) -> CodeMetrics:
        """Calculate code metrics"""
        lines = code.split('\n')
        
        # Basic line counts
        lines_of_code = len(lines)
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                comment_lines += 1
        
        # Calculate complexity (simplified)
        complexity_score = self._calculate_complexity(code, language)
        
        # Calculate maintainability (simplified)
        maintainability_index = self._calculate_maintainability(code, comment_lines, lines_of_code)
        
        # Extract functions and classes
        functions, classes = self._extract_code_elements(code, language)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(complexity_score, maintainability_index)
        
        return CodeMetrics(
            file_path=file_path,
            language=language,
            lines_of_code=lines_of_code,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            complexity_score=complexity_score,
            maintainability_index=maintainability_index,
            quality_score=quality_score,
            functions=functions,
            classes=classes
        )
    
    def _calculate_complexity(self, code: str, language: ProgrammingLanguage) -> float:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = {
            ProgrammingLanguage.PYTHON: ["if", "elif", "for", "while", "and", "or", "except"],
            ProgrammingLanguage.JAVASCRIPT: ["if", "else if", "for", "while", "&&", "||", "catch"],
            ProgrammingLanguage.JAVA: ["if", "else if", "for", "while", "&&", "||", "catch"],
            ProgrammingLanguage.CPLUSPLUS: ["if", "else if", "for", "while", "&&", "||", "catch"],
            ProgrammingLanguage.CSHARP: ["if", "else if", "for", "while", "&&", "||", "catch"],
            ProgrammingLanguage.GO: ["if", "else if", "for", "while", "&&", "||"],
            ProgrammingLanguage.RUST: ["if", "else if", "for", "while", "&&", "||"]
        }
        
        keywords = decision_keywords.get(language, ["if", "for", "while"])
        
        for keyword in keywords:
            complexity += len(re.findall(r'\b' + keyword + r'\b', code))
        
        return min(complexity, 50)  # Cap at 50
    
    def _calculate_maintainability(self, code: str, comment_lines: int, total_lines: int) -> float:
        """Calculate maintainability index (simplified)"""
        if total_lines == 0:
            return 0.0
        
        # Comment ratio
        comment_ratio = comment_lines / total_lines
        
        # Average line length
        lines = code.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        
        # Maintainability factors
        comment_factor = min(comment_ratio * 2, 1.0)  # Max 1.0 for 50% comments
        length_factor = max(0, 1 - (avg_line_length - 50) / 100)  # Penalize long lines
        
        return (comment_factor + length_factor) / 2
    
    def _extract_code_elements(self, code: str, language: ProgrammingLanguage) -> Tuple[List[str], List[str]]:
        """Extract function and class names"""
        functions = []
        classes = []
        
        if language == ProgrammingLanguage.PYTHON:
            # Extract function definitions
            func_matches = re.findall(r'def\s+(\w+)\s*\(', code)
            functions.extend(func_matches)
            
            # Extract class definitions
            class_matches = re.findall(r'class\s+(\w+)', code)
            classes.extend(class_matches)
        
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            # Extract function definitions
            func_matches = re.findall(r'function\s+(\w+)\s*\(', code)
            functions.extend(func_matches)
            
            # Extract class definitions
            class_matches = re.findall(r'class\s+(\w+)', code)
            classes.extend(class_matches)
        
        elif language == ProgrammingLanguage.JAVA:
            # Extract method definitions
            method_matches = re.findall(r'public\s+\w+\s+(\w+)\s*\(', code)
            functions.extend(method_matches)
            
            # Extract class definitions
            class_matches = re.findall(r'public\s+class\s+(\w+)', code)
            classes.extend(class_matches)
        
        return functions, classes
    
    def _calculate_quality_score(self, complexity: float, maintainability: float) -> float:
        """Calculate overall quality score"""
        # Complexity factor (lower is better)
        complexity_factor = max(0, 1 - complexity / 20)
        
        # Maintainability factor (higher is better)
        maintainability_factor = maintainability
        
        return (complexity_factor + maintainability_factor) / 2
    
    def _detect_issues(self, code: str, language: ProgrammingLanguage, file_path: str) -> List[CodeIssue]:
        """Detect code issues"""
        issues = []
        lines = code.split('\n')
        
        # Get patterns for the language
        patterns = self.code_patterns.get(language.value, {})
        anti_patterns = patterns.get("anti_patterns", [])
        
        for pattern_info in anti_patterns:
            pattern = pattern_info["pattern"]
            
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issue = CodeIssue(
                        issue_id=self._generate_id("issue"),
                        issue_type=CodeIssueType(pattern_info["type"]),
                        severity=pattern_info["severity"],
                        message=pattern_info["message"],
                        line_number=line_num,
                        code_snippet=line.strip(),
                        suggestion=pattern_info["suggestion"],
                        confidence=0.8,
                        file_path=file_path
                    )
                    issues.append(issue)
        
        # Language-specific issue detection
        if language == ProgrammingLanguage.PYTHON:
            issues.extend(self._detect_python_issues(code, file_path))
        elif language == ProgrammingLanguage.JAVASCRIPT:
            issues.extend(self._detect_javascript_issues(code, file_path))
        
        return issues
    
    def _detect_python_issues(self, code: str, file_path: str) -> List[CodeIssue]:
        """Detect Python-specific issues"""
        issues = []
        lines = code.split('\n')
        
        try:
            # Try to parse the code
            tree = ast.parse(code)
            
            # Check for unused imports
            imported_names = set()
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
            
            unused_imports = imported_names - used_names
            for imp in unused_imports:
                # Find the line number of the import
                for line_num, line in enumerate(lines, 1):
                    if f"import {imp}" in line or f"from {imp}" in line:
                        issue = CodeIssue(
                            issue_id=self._generate_id("issue"),
                            issue_type=CodeIssueType.STYLE_VIOLATION,
                            severity="low",
                            message=f"Unused import: {imp}",
                            line_number=line_num,
                            code_snippet=line.strip(),
                            suggestion="Remove the unused import",
                            confidence=0.9,
                            file_path=file_path
                        )
                        issues.append(issue)
                        break
        
        except SyntaxError as e:
            issue = CodeIssue(
                issue_id=self._generate_id("issue"),
                issue_type=CodeIssueType.SYNTAX_ERROR,
                severity="critical",
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                code_snippet=lines[e.lineno - 1].strip() if e.lineno <= len(lines) else "",
                suggestion="Fix the syntax error",
                confidence=1.0,
                file_path=file_path
            )
            issues.append(issue)
        
        return issues
    
    def _detect_javascript_issues(self, code: str, file_path: str) -> List[CodeIssue]:
        """Detect JavaScript-specific issues"""
        issues = []
        lines = code.split('\n')
        
        # Check for console.log statements (should be removed in production)
        for line_num, line in enumerate(lines, 1):
            if "console.log" in line:
                issue = CodeIssue(
                    issue_id=self._generate_id("issue"),
                    issue_type=CodeIssueType.STYLE_VIOLATION,
                    severity="low",
                    message="Console.log statement found",
                    line_number=line_num,
                    code_snippet=line.strip(),
                    suggestion="Remove console.log statements in production code",
                    confidence=0.8,
                    file_path=file_path
                )
                issues.append(issue)
        
        return issues
    
    def _generate_suggestions(self, code: str, language: ProgrammingLanguage, 
                            metrics: CodeMetrics, issues: List[CodeIssue], file_path: str) -> List[CodeSuggestion]:
        """Generate code improvement suggestions"""
        suggestions = []
        
        # Generate suggestions based on metrics
        if metrics.complexity_score > 10:
            suggestion = CodeSuggestion(
                suggestion_id=self._generate_id("suggestion"),
                suggestion_type="refactoring",
                title="Reduce complexity",
                description=f"Function complexity is {metrics.complexity_score:.1f}, consider breaking it down",
                original_code="",
                suggested_code="",
                confidence=0.7,
                benefits=["Improved readability", "Easier testing", "Better maintainability"],
                risks=["May require additional testing"],
                file_path=file_path
            )
            suggestions.append(suggestion)
        
        if metrics.comment_lines / metrics.lines_of_code < 0.1:
            suggestion = CodeSuggestion(
                suggestion_id=self._generate_id("suggestion"),
                suggestion_type="documentation",
                title="Add comments",
                description="Code has low comment ratio, consider adding more documentation",
                original_code="",
                suggested_code="",
                confidence=0.8,
                benefits=["Better code understanding", "Easier maintenance"],
                risks=["None"],
                file_path=file_path
            )
            suggestions.append(suggestion)
        
        # Generate suggestions based on issues
        for issue in issues:
            if issue.issue_type == CodeIssueType.PERFORMANCE_ISSUE:
                suggestion = CodeSuggestion(
                    suggestion_id=self._generate_id("suggestion"),
                    suggestion_type="optimization",
                    title=f"Optimize {issue.message}",
                    description=issue.suggestion,
                    original_code=issue.code_snippet,
                    suggested_code=self._generate_optimized_code(issue.code_snippet, language),
                    confidence=issue.confidence,
                    benefits=["Improved performance"],
                    risks=["May change behavior"],
                    file_path=file_path,
                    line_number=issue.line_number
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_optimized_code(self, original_code: str, language: ProgrammingLanguage) -> str:
        """Generate optimized version of code (simplified)"""
        if language == ProgrammingLanguage.PYTHON:
            # Simple optimizations
            if "len(range(" in original_code:
                return original_code.replace("len(range(", "range(")
            elif "range(len(" in original_code:
                return original_code.replace("range(len(", "enumerate(")
        
        return original_code  # Default: return original
    
    def _calculate_overall_quality(self, metrics: CodeMetrics, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Calculate overall quality assessment"""
        # Count issues by severity
        severity_counts = defaultdict(int)
        for issue in issues:
            severity_counts[issue.severity] += 1
        
        # Calculate overall score
        base_score = metrics.quality_score
        
        # Deduct for issues
        issue_penalty = 0
        issue_penalty += severity_counts["critical"] * 0.3
        issue_penalty += severity_counts["high"] * 0.2
        issue_penalty += severity_counts["medium"] * 0.1
        issue_penalty += severity_counts["low"] * 0.05
        
        final_score = max(0, base_score - issue_penalty)
        
        # Determine quality level
        if final_score >= 0.9:
            quality_level = "excellent"
        elif final_score >= 0.7:
            quality_level = "good"
        elif final_score >= 0.5:
            quality_level = "fair"
        elif final_score >= 0.3:
            quality_level = "poor"
        else:
            quality_level = "critical"
        
        return {
            "score": final_score,
            "level": quality_level,
            "issues_count": len(issues),
            "severity_breakdown": dict(severity_counts),
            "metrics_score": metrics.quality_score
        }
    
    def generate_code(self, prompt: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
                     context: str = "") -> Dict[str, Any]:
        """Generate code based on prompt and context"""
        # Simple code generation (in real implementation, use LLM)
        generated_code = self._generate_code_from_prompt(prompt, language, context)
        
        # Analyze the generated code
        analysis = self.analyze_code(generated_code)
        
        # Generate documentation
        docs = self.generate_documentation(generated_code, language)
        
        return {
            "generated_code": generated_code,
            "analysis": analysis,
            "documentation": docs,
            "confidence": 0.7,
            "language": language.value
        }
    
    def _generate_code_from_prompt(self, prompt: str, language: ProgrammingLanguage, context: str) -> str:
        """Generate code from prompt (simplified)"""
        prompt_lower = prompt.lower()
        
        if language == ProgrammingLanguage.PYTHON:
            if "function" in prompt_lower and "add" in prompt_lower:
                return """def add_numbers(a, b):
    \"\"\"Add two numbers and return the result.
    
    Args:
        a (int/float): First number
        b (int/float): Second number
        
    Returns:
        int/float: Sum of the two numbers
    \"\"\"
    return a + b"""
            
            elif "class" in prompt_lower and "person" in prompt_lower:
                return """class Person:
    \"\"\"Represents a person with name and age.\"\"\"
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        \"\"\"Return a greeting message.\"\"\"
        return f\"Hello, my name is {self.name} and I am {self.age} years old.\"\""
            
            elif "loop" in prompt_lower and "print" in prompt_lower:
                return """# Print numbers from 1 to 10
for i in range(1, 11):
    print(f"Number: {i}")"""
        
        elif language == ProgrammingLanguage.JAVASCRIPT:
            if "function" in prompt_lower and "add" in prompt_lower:
                return """function addNumbers(a, b) {
    // Add two numbers and return the result
    return a + b;
}"""
        
        # Default response
        return f"# Generated code for: {prompt}\n# Language: {language.value}\n# TODO: Implement the logic"
    
    def generate_documentation(self, code: str, language: ProgrammingLanguage) -> List[Documentation]:
        """Generate documentation for code"""
        docs = []
        
        if language == ProgrammingLanguage.PYTHON:
            docs.extend(self._generate_python_docs(code))
        elif language == ProgrammingLanguage.JAVASCRIPT:
            docs.extend(self._generate_javascript_docs(code))
        
        # Store documentation
        for doc in docs:
            doc_id = self._generate_id("doc")
            self.documentation[doc_id] = doc
        
        self._save_data()
        
        return docs
    
    def _generate_python_docs(self, code: str) -> List[Documentation]:
        """Generate Python documentation"""
        docs = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    doc = Documentation(
                        doc_id=self._generate_id("doc"),
                        doc_type="function_doc",
                        title=f"Function: {node.name}",
                        content=self._generate_function_doc(node),
                        code_element=node.name,
                        language=ProgrammingLanguage.PYTHON,
                        confidence=0.8
                    )
                    docs.append(doc)
                
                elif isinstance(node, ast.ClassDef):
                    doc = Documentation(
                        doc_id=self._generate_id("doc"),
                        doc_type="class_doc",
                        title=f"Class: {node.name}",
                        content=self._generate_class_doc(node),
                        code_element=node.name,
                        language=ProgrammingLanguage.PYTHON,
                        confidence=0.8
                    )
                    docs.append(doc)
        
        except SyntaxError:
            pass
        
        return docs
    
    def _generate_function_doc(self, node: ast.FunctionDef) -> str:
        """Generate function documentation"""
        doc = f"Function `{node.name}`"
        
        # Extract arguments
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        if args:
            doc += f"\n\nArguments: {', '.join(args)}"
        
        # Check for existing docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            existing_doc = node.body[0].value.s
            doc += f"\n\nExisting documentation:\n{existing_doc}"
        else:
            doc += "\n\nNo existing documentation found."
        
        return doc
    
    def _generate_class_doc(self, node: ast.ClassDef) -> str:
        """Generate class documentation"""
        doc = f"Class `{node.name}`"
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
        
        if methods:
            doc += f"\n\nMethods: {', '.join(methods)}"
        
        return doc
    
    def _generate_javascript_docs(self, code: str) -> List[Documentation]:
        """Generate JavaScript documentation"""
        docs = []
        
        # Simple function extraction
        func_matches = re.findall(r'function\s+(\w+)\s*\([^)]*\)', code)
        
        for func_name in func_matches:
            doc = Documentation(
                doc_id=self._generate_id("doc"),
                doc_type="function_doc",
                title=f"Function: {func_name}",
                content=f"JavaScript function `{func_name}`",
                code_element=func_name,
                language=ProgrammingLanguage.JAVASCRIPT,
                confidence=0.7
            )
            docs.append(doc)
        
        return docs
    
    def get_assistant_stats(self) -> Dict[str, Any]:
        """Get code assistant statistics"""
        return {
            "total_analyses": len(self.metrics),
            "total_issues": len(self.issues),
            "total_suggestions": len(self.suggestions),
            "total_documentation": len(self.documentation),
            "issues_by_type": self._count_issues_by_type(),
            "suggestions_by_type": self._count_suggestions_by_type(),
            "languages_analyzed": self._count_languages_analyzed()
        }
    
    def _count_issues_by_type(self) -> Dict[str, int]:
        """Count issues by type"""
        counts = defaultdict(int)
        for issue in self.issues.values():
            counts[issue.issue_type.value] += 1
        return dict(counts)
    
    def _count_suggestions_by_type(self) -> Dict[str, int]:
        """Count suggestions by type"""
        counts = defaultdict(int)
        for suggestion in self.suggestions.values():
            counts[suggestion.suggestion_type] += 1
        return dict(counts)
    
    def _count_languages_analyzed(self) -> Dict[str, int]:
        """Count analyses by language"""
        counts = defaultdict(int)
        for metrics in self.metrics.values():
            counts[metrics.language.value] += 1
        return dict(counts)


# Factory function for creating code assistant
async def create_code_assistant(config: Dict[str, Any] = None) -> CodeAssistant:
    """Create and initialize code assistant"""
    config = config or {}
    storage_path = config.get("storage_path", "./code_assistant")
    
    assistant = CodeAssistant(storage_path)
    await assistant.start()
    
    return assistant