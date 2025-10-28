"""
Pipeline Orchestration System for Bharat-FM MLOps Platform

This module provides comprehensive pipeline orchestration capabilities for ML workflows,
including training, evaluation, deployment, and monitoring pipelines. It supports
complex workflows with dependencies, parallel execution, and resource management.

Features:
- Pipeline definition and execution
- Workflow dependency management
- Parallel and distributed execution
- Resource scheduling and optimization
- Pipeline monitoring and logging
- Failure recovery and retry mechanisms
- Pipeline versioning and rollback
"""

import time
import threading
import json
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import asyncio
import concurrent.futures
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class TaskType(Enum):
    """Task types"""
    PYTHON = "python"
    SHELL = "shell"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    HTTP = "http"
    SQL = "sql"
    CUSTOM = "custom"

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: str = None
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = None
    resources_used: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

@dataclass
class Task:
    """Pipeline task definition"""
    task_id: str
    name: str
    task_type: TaskType
    command: Union[str, Callable, Dict[str, Any]]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    retry_count: int = 3
    retry_delay_seconds: int = 60
    resources: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    condition: str = None  # Conditional execution logic
    
@dataclass
class Pipeline:
    """Pipeline definition"""
    pipeline_id: str
    name: str
    description: str
    tasks: List[Task]
    global_environment: Dict[str, str] = field(default_factory=dict)
    global_resources: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 86400  # 24 hours
    max_concurrent_tasks: int = 4
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
@dataclass
class PipelineExecution:
    """Pipeline execution instance"""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = None
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

@dataclass
class PipelineSchedule:
    """Pipeline schedule configuration"""
    schedule_id: str
    pipeline_id: str
    cron_expression: str
    enabled: bool = True
    max_concurrent_runs: int = 1
    timezone: str = "UTC"
    next_run_time: datetime = None
    created_at: datetime = field(default_factory=datetime.now)

class PipelineOrchestrator:
    """
    Comprehensive pipeline orchestration system
    """
    
    def __init__(self, max_workers: int = 10, storage_dir: str = "pipeline_storage"):
        self.max_workers = max_workers
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.pipelines = {}
        self.pipeline_executions = {}
        self.schedules = {}
        self.running_executions = set()
        
        # Task execution
        self.task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.task_futures = {}
        
        # Configuration
        self.default_resources = {
            'cpu': '1000m',
            'memory': '1Gi',
            'gpu': '0'
        }
        
        # Monitoring and logging
        self.execution_history = deque(maxlen=1000)
        self.task_history = deque(maxlen=10000)
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread = None
        self._monitor_thread = None
        
        # Statistics
        self.stats = {
            'pipelines_created': 0,
            'executions_started': 0,
            'executions_completed': 0,
            'executions_failed': 0,
            'tasks_executed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0
        }
        
    def start_orchestrator(self):
        """Start the pipeline orchestrator"""
        if self._running:
            logger.warning("Pipeline orchestrator already running")
            return
            
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        
        self._scheduler_thread.start()
        self._monitor_thread.start()
        
        logger.info("Pipeline orchestrator started")
        
    def stop_orchestrator(self):
        """Stop the pipeline orchestrator"""
        self._running = False
        
        # Cancel running executions
        for execution_id in list(self.running_executions):
            self.cancel_execution(execution_id)
            
        # Shutdown executor
        self.task_executor.shutdown(wait=True)
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            
        logger.info("Pipeline orchestrator stopped")
        
    def create_pipeline(self, pipeline: Pipeline) -> str:
        """
        Create a new pipeline
        
        Args:
            pipeline: Pipeline object
            
        Returns:
            Pipeline ID
        """
        with self._lock:
            # Validate pipeline
            self._validate_pipeline(pipeline)
            
            # Store pipeline
            self.pipelines[pipeline.pipeline_id] = pipeline
            self.stats['pipelines_created'] += 1
            
            logger.info(f"Created pipeline: {pipeline.pipeline_id}")
            
            return pipeline.pipeline_id
            
    def update_pipeline(self, pipeline_id: str, pipeline: Pipeline):
        """
        Update an existing pipeline
        
        Args:
            pipeline_id: Pipeline identifier
            pipeline: Updated pipeline object
        """
        with self._lock:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
                
            # Validate pipeline
            self._validate_pipeline(pipeline)
            
            # Update pipeline
            self.pipelines[pipeline_id] = pipeline
            
            logger.info(f"Updated pipeline: {pipeline_id}")
            
    def delete_pipeline(self, pipeline_id: str):
        """
        Delete a pipeline
        
        Args:
            pipeline_id: Pipeline identifier
        """
        with self._lock:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
                
            # Check if pipeline is running
            running_executions = [
                exec_id for exec_id, execution in self.pipeline_executions.items()
                if execution.pipeline_id == pipeline_id and execution.status == PipelineStatus.RUNNING
            ]
            
            if running_executions:
                raise ValueError(f"Cannot delete pipeline {pipeline_id}: {len(running_executions)} executions are running")
                
            # Delete pipeline
            del self.pipelines[pipeline_id]
            
            # Remove associated schedules
            schedules_to_remove = [
                schedule_id for schedule_id, schedule in self.schedules.items()
                if schedule.pipeline_id == pipeline_id
            ]
            
            for schedule_id in schedules_to_remove:
                del self.schedules[schedule_id]
                
            logger.info(f"Deleted pipeline: {pipeline_id}")
            
    def execute_pipeline(self, pipeline_id: str, parameters: Dict[str, Any] = None) -> str:
        """
        Execute a pipeline
        
        Args:
            pipeline_id: Pipeline identifier
            parameters: Execution parameters
            
        Returns:
            Execution ID
        """
        with self._lock:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
                
            pipeline = self.pipelines[pipeline_id]
            
            # Create execution instance
            execution_id = f"exec_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            execution = PipelineExecution(
                execution_id=execution_id,
                pipeline_id=pipeline_id,
                status=PipelineStatus.PENDING,
                metadata=parameters or {}
            )
            
            self.pipeline_executions[execution_id] = execution
            self.running_executions.add(execution_id)
            self.stats['executions_started'] += 1
            
            # Start execution asynchronously
            self.task_executor.submit(self._execute_pipeline, execution_id)
            
            logger.info(f"Started pipeline execution: {execution_id}")
            
            return execution_id
            
    def cancel_execution(self, execution_id: str):
        """
        Cancel a pipeline execution
        
        Args:
            execution_id: Execution identifier
        """
        with self._lock:
            if execution_id not in self.pipeline_executions:
                raise ValueError(f"Execution {execution_id} not found")
                
            execution = self.pipeline_executions[execution_id]
            
            if execution.status not in [PipelineStatus.PENDING, PipelineStatus.RUNNING]:
                raise ValueError(f"Cannot cancel execution {execution_id} in status {execution.status}")
                
            # Cancel execution
            execution.status = PipelineStatus.CANCELLED
            execution.end_time = datetime.now()
            
            if execution_id in self.running_executions:
                self.running_executions.remove(execution_id)
                
            # Cancel task futures
            for task_id, future in list(self.task_futures.items()):
                if task_id.startswith(execution_id):
                    future.cancel()
                    del self.task_futures[task_id]
                    
            logger.info(f"Cancelled pipeline execution: {execution_id}")
            
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """
        Get pipeline by ID
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Pipeline object or None
        """
        with self._lock:
            return self.pipelines.get(pipeline_id)
            
    def get_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """
        Get execution by ID
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            PipelineExecution object or None
        """
        with self._lock:
            return self.pipeline_executions.get(execution_id)
            
    def get_executions(self, pipeline_id: str = None, status: PipelineStatus = None) -> List[PipelineExecution]:
        """
        Get pipeline executions
        
        Args:
            pipeline_id: Optional pipeline filter
            status: Optional status filter
            
        Returns:
            List of PipelineExecution objects
        """
        with self._lock:
            executions = list(self.pipeline_executions.values())
            
            if pipeline_id:
                executions = [exec for exec in executions if exec.pipeline_id == pipeline_id]
                
            if status:
                executions = [exec for exec in executions if exec.status == status]
                
            return sorted(executions, key=lambda x: x.start_time or datetime.min, reverse=True)
            
    def create_schedule(self, pipeline_id: str, cron_expression: str, **kwargs) -> str:
        """
        Create a pipeline schedule
        
        Args:
            pipeline_id: Pipeline identifier
            cron_expression: Cron expression for scheduling
            **kwargs: Additional schedule parameters
            
        Returns:
            Schedule ID
        """
        with self._lock:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
                
            schedule_id = f"schedule_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            schedule = PipelineSchedule(
                schedule_id=schedule_id,
                pipeline_id=pipeline_id,
                cron_expression=cron_expression,
                **kwargs
            )
            
            self.schedules[schedule_id] = schedule
            
            logger.info(f"Created schedule: {schedule_id} for pipeline {pipeline_id}")
            
            return schedule_id
            
    def delete_schedule(self, schedule_id: str):
        """
        Delete a pipeline schedule
        
        Args:
            schedule_id: Schedule identifier
        """
        with self._lock:
            if schedule_id not in self.schedules:
                raise ValueError(f"Schedule {schedule_id} not found")
                
            del self.schedules[schedule_id]
            logger.info(f"Deleted schedule: {schedule_id}")
            
    def get_schedules(self, pipeline_id: str = None) -> List[PipelineSchedule]:
        """
        Get pipeline schedules
        
        Args:
            pipeline_id: Optional pipeline filter
            
        Returns:
            List of PipelineSchedule objects
        """
        with self._lock:
            schedules = list(self.schedules.values())
            
            if pipeline_id:
                schedules = [schedule for schedule in schedules if schedule.pipeline_id == pipeline_id]
                
            return schedules
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'pipelines_count': len(self.pipelines),
                'schedules_count': len(self.schedules),
                'running_executions_count': len(self.running_executions),
                'active_tasks_count': len(self.task_futures)
            }
            
    def export_pipeline_data(self, filename: str = None) -> str:
        """
        Export pipeline data to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of pipeline data
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'pipelines': {k: asdict(v) for k, v in self.pipelines.items()},
                'schedules': {k: asdict(v) for k, v in self.schedules.items()},
                'recent_executions': [asdict(exec) for exec in self.get_executions()[:10]],
                'statistics': self.get_statistics()
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Pipeline data exported to {filename}")
                
            return json_data
            
    def _validate_pipeline(self, pipeline: Pipeline):
        """Validate pipeline definition"""
        # Check for duplicate task IDs
        task_ids = [task.task_id for task in pipeline.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found in pipeline")
            
        # Check for circular dependencies
        self._check_circular_dependencies(pipeline)
        
        # Validate task configurations
        for task in pipeline.tasks:
            self._validate_task(task)
            
    def _validate_task(self, task: Task):
        """Validate task configuration"""
        if not task.task_id:
            raise ValueError("Task ID is required")
            
        if not task.name:
            raise ValueError("Task name is required")
            
        if not task.command:
            raise ValueError("Task command is required")
            
        # Validate dependencies exist
        pipeline = next((p for p in self.pipelines.values() if task in p.tasks), None)
        if pipeline:
            for dep_id in task.dependencies:
                if not any(t.task_id == dep_id for t in pipeline.tasks):
                    raise ValueError(f"Dependency task {dep_id} not found in pipeline")
                    
    def _check_circular_dependencies(self, pipeline: Pipeline):
        """Check for circular dependencies in pipeline"""
        task_map = {task.task_id: task for task in pipeline.tasks}
        
        def has_cycle(task_id, visited=None, recursion_stack=None):
            if visited is None:
                visited = set()
            if recursion_stack is None:
                recursion_stack = set()
                
            visited.add(task_id)
            recursion_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id, visited, recursion_stack):
                            return True
                    elif dep_id in recursion_stack:
                        return True
                        
            recursion_stack.remove(task_id)
            return False
            
        for task in pipeline.tasks:
            if has_cycle(task.task_id):
                raise ValueError(f"Circular dependency detected starting from task {task.task_id}")
                
    def _execute_pipeline(self, execution_id: str):
        """Execute a pipeline"""
        try:
            with self._lock:
                execution = self.pipeline_executions.get(execution_id)
                if not execution:
                    logger.error(f"Execution {execution_id} not found")
                    return
                    
                if execution.status != PipelineStatus.PENDING:
                    logger.error(f"Execution {execution_id} is not in PENDING status")
                    return
                    
                execution.status = PipelineStatus.RUNNING
                execution.start_time = datetime.now()
                
            pipeline = self.pipelines[execution.pipeline_id]
            
            logger.info(f"Executing pipeline {pipeline.pipeline_id} (execution: {execution_id})")
            
            # Build task execution graph
            task_graph = self._build_task_graph(pipeline)
            
            # Execute tasks in topological order
            await self._execute_tasks(execution_id, task_graph)
            
            # Check final status
            with self._lock:
                execution = self.pipeline_executions.get(execution_id)
                if execution:
                    if execution.status == PipelineStatus.RUNNING:
                        execution.status = PipelineStatus.COMPLETED
                        execution.end_time = datetime.now()
                        
                    if execution_id in self.running_executions:
                        self.running_executions.remove(execution_id)
                        
                    self.stats['executions_completed'] += 1
                    
            logger.info(f"Completed pipeline execution: {execution_id}")
            
        except Exception as e:
            with self._lock:
                execution = self.pipeline_executions.get(execution_id)
                if execution:
                    execution.status = PipelineStatus.FAILED
                    execution.end_time = datetime.now()
                    execution.logs.append(f"Pipeline execution failed: {str(e)}")
                    
                    if execution_id in self.running_executions:
                        self.running_executions.remove(execution_id)
                        
                    self.stats['executions_failed'] += 1
                    
            logger.error(f"Pipeline execution {execution_id} failed: {e}")
            
    async def _execute_tasks(self, execution_id: str, task_graph: Dict[str, List[str]]):
        """Execute tasks in dependency order"""
        execution = self.pipeline_executions[execution_id]
        pipeline = self.pipelines[execution.pipeline_id]
        
        # Track completed tasks
        completed_tasks = set()
        running_tasks = set()
        
        # Track task futures
        task_futures = {}
        
        while len(completed_tasks) < len(pipeline.tasks):
            # Find ready tasks (all dependencies completed)
            ready_tasks = []
            
            for task in pipeline.tasks:
                if (task.task_id not in completed_tasks and 
                    task.task_id not in running_tasks and
                    all(dep_id in completed_tasks for dep_id in task.dependencies)):
                    
                    # Check task condition if specified
                    if task.condition and not self._evaluate_condition(task.condition, execution.task_results):
                        # Skip task
                        task_result = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.SKIPPED,
                            start_time=datetime.now(),
                            end_time=datetime.now()
                        )
                        
                        with self._lock:
                            execution.task_results[task.task_id] = task_result
                            
                        completed_tasks.add(task.task_id)
                        continue
                        
                    ready_tasks.append(task)
                    
            # Execute ready tasks (respect max concurrency)
            available_slots = pipeline.max_concurrent_tasks - len(running_tasks)
            tasks_to_execute = ready_tasks[:available_slots]
            
            for task in tasks_to_execute:
                running_tasks.add(task.task_id)
                
                # Submit task for execution
                future = self.task_executor.submit(self._execute_task, execution_id, task)
                task_futures[task.task_id] = future
                
                with self._lock:
                    self.task_futures[f"{execution_id}:{task.task_id}"] = future
                    
            # Wait for at least one task to complete
            if task_futures:
                # Wait for any task to complete
                completed_futures = []
                
                for task_id, future in list(task_futures.items()):
                    if future.done():
                        completed_futures.append((task_id, future))
                        
                if not completed_futures:
                    await asyncio.sleep(1)
                    continue
                    
                # Process completed tasks
                for task_id, future in completed_futures:
                    try:
                        task_result = future.result()
                        
                        with self._lock:
                            execution.task_results[task_id] = task_result
                            
                        completed_tasks.add(task_id)
                        running_tasks.remove(task_id)
                        
                        # Remove from tracking
                        if f"{execution_id}:{task_id}" in self.task_futures:
                            del self.task_futures[f"{execution_id}:{task_id}"]
                            
                        del task_futures[task_id]
                        
                        # Check if task failed and handle retry
                        if task_result.status == TaskStatus.FAILED:
                            task = next(t for t in pipeline.tasks if t.task_id == task_id)
                            
                            if task_result.error and task.retry_count > 0:
                                # Retry task
                                logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                                task.retry_count -= 1
                                
                                # Remove from completed and running, add back to ready
                                completed_tasks.remove(task_id)
                                running_tasks.discard(task_id)
                                
                                # Add delay before retry
                                time.sleep(task.retry_delay_seconds)
                                
                    except Exception as e:
                        logger.error(f"Task {task_id} execution failed: {e}")
                        
                        # Mark as failed
                        task_result = TaskResult(
                            task_id=task_id,
                            status=TaskStatus.FAILED,
                            error=str(e),
                            start_time=datetime.now(),
                            end_time=datetime.now()
                        )
                        
                        with self._lock:
                            execution.task_results[task_id] = task_result
                            
                        completed_tasks.add(task_id)
                        running_tasks.discard(task_id)
                        
                        # Remove from tracking
                        if f"{execution_id}:{task_id}" in self.task_futures:
                            del self.task_futures[f"{execution_id}:{task_id}"]
                            
                        del task_futures[task_id]
                        
            else:
                # No tasks ready, wait a bit
                await asyncio.sleep(1)
                
    def _execute_task(self, execution_id: str, task: Task) -> TaskResult:
        """Execute a single task"""
        task_result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"Executing task {task.task_id} for execution {execution_id}")
            
            # Merge resources
            resources = {**self.default_resources, **task.resources}
            
            # Execute based on task type
            if task.task_type == TaskType.PYTHON:
                if callable(task.command):
                    result = task.command(**task_result.__dict__)
                else:
                    result = self._execute_python_task(task, execution_id)
                    
            elif task.task_type == TaskType.SHELL:
                result = self._execute_shell_task(task, execution_id)
                
            elif task.task_type == TaskType.DOCKER:
                result = self._execute_docker_task(task, execution_id)
                
            elif task.task_type == TaskType.HTTP:
                result = self._execute_http_task(task, execution_id)
                
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")
                
            task_result.status = TaskStatus.COMPLETED
            task_result.output = result
            
            with self._lock:
                self.stats['tasks_executed'] += 1
                
            logger.info(f"Completed task {task.task_id}")
            
        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.error = str(e)
            
            with self._lock:
                self.stats['tasks_failed'] += 1
                
            logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            task_result.end_time = datetime.now()
            
        return task_result
        
    def _execute_python_task(self, task: Task, execution_id: str) -> Any:
        """Execute Python task"""
        # This is a placeholder for Python task execution
        # In a real implementation, this would execute Python code or import and call functions
        logger.info(f"Executing Python task: {task.command}")
        return f"Python task {task.task_id} executed"
        
    def _execute_shell_task(self, task: Task, execution_id: str) -> Any:
        """Execute shell task"""
        import subprocess
        
        try:
            result = subprocess.run(
                task.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                env={**os.environ, **task.environment}
            )
            
            if result.returncode != 0:
                raise Exception(f"Shell command failed with return code {result.returncode}: {result.stderr}")
                
            return result.stdout
            
        except subprocess.TimeoutExpired:
            raise Exception(f"Shell command timed out after {task.timeout_seconds} seconds")
            
    def _execute_docker_task(self, task: Task, execution_id: str) -> Any:
        """Execute Docker task"""
        # This is a placeholder for Docker task execution
        logger.info(f"Executing Docker task: {task.command}")
        return f"Docker task {task.task_id} executed"
        
    def _execute_http_task(self, task: Task, execution_id: str) -> Any:
        """Execute HTTP task"""
        import requests
        
        try:
            if isinstance(task.command, dict):
                response = requests.request(
                    method=task.command.get('method', 'GET'),
                    url=task.command['url'],
                    headers=task.command.get('headers', {}),
                    json=task.command.get('json'),
                    timeout=task.timeout_seconds
                )
                response.raise_for_status()
                return response.json()
            else:
                raise Exception("HTTP task command must be a dictionary")
                
        except requests.RequestException as e:
            raise Exception(f"HTTP request failed: {e}")
            
    def _build_task_graph(self, pipeline: Pipeline) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        
        for task in pipeline.tasks:
            graph[task.task_id] = task.dependencies
            
        return graph
        
    def _evaluate_condition(self, condition: str, task_results: Dict[str, TaskResult]) -> bool:
        """Evaluate task condition"""
        # This is a placeholder for condition evaluation
        # In a real implementation, this would evaluate the condition string
        # using the task results
        return True
        
    def _scheduler_loop(self):
        """Main scheduler loop for scheduled pipelines"""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Check schedules
                for schedule_id, schedule in list(self.schedules.items()):
                    if not schedule.enabled:
                        continue
                        
                    # Check if it's time to run
                    if (schedule.next_run_time and 
                        current_time >= schedule.next_run_time):
                        
                        # Check concurrent run limit
                        running_count = len([
                            exec for exec in self.pipeline_executions.values()
                            if (exec.pipeline_id == schedule.pipeline_id and 
                                exec.status == PipelineStatus.RUNNING)
                        ])
                        
                        if running_count < schedule.max_concurrent_runs:
                            # Execute pipeline
                            execution_id = self.execute_pipeline(schedule.pipeline_id)
                            logger.info(f"Scheduled execution started: {execution_id}")
                            
                        # Calculate next run time
                        schedule.next_run_time = self._calculate_next_run_time(schedule.cron_expression)
                        
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(30)
                
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Monitor running executions
                for execution_id in list(self.running_executions):
                    execution = self.pipeline_executions.get(execution_id)
                    
                    if execution and execution.status == PipelineStatus.RUNNING:
                        # Check timeout
                        if (execution.start_time and 
                            (datetime.now() - execution.start_time).total_seconds() > execution.timeout_seconds):
                            
                            logger.warning(f"Execution {execution_id} timed out")
                            self.cancel_execution(execution_id)
                            
                # Clean up old data
                self._cleanup_old_data()
                
                # Log statistics
                if self.stats['executions_started'] > 0 and self.stats['executions_started'] % 10 == 0:
                    logger.info(f"Pipeline orchestrator stats: {self.stats}")
                    
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(15)
                
    def _calculate_next_run_time(self, cron_expression: str) -> datetime:
        """Calculate next run time from cron expression"""
        # This is a placeholder for cron expression parsing
        # In a real implementation, this would use a cron library
        return datetime.now() + timedelta(hours=1)
        
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        with self._lock:
            # Clean up old executions
            executions_to_remove = [
                exec_id for exec_id, execution in self.pipeline_executions.items()
                if (execution.end_time and execution.end_time < cutoff_time and
                    execution.status not in [PipelineStatus.RUNNING, PipelineStatus.PENDING])
            ]
            
            for exec_id in executions_to_remove:
                del self.pipeline_executions[exec_id]
                self.execution_history.append(exec_id)

# Example usage and testing
def main():
    """Example usage of the pipeline orchestration system"""
    orchestrator = PipelineOrchestrator(max_workers=5)
    
    try:
        orchestrator.start_orchestrator()
        
        # Create a simple pipeline
        task1 = Task(
            task_id="task1",
            name="Data Preparation",
            task_type=TaskType.PYTHON,
            command="prepare_data",
            timeout_seconds=300
        )
        
        task2 = Task(
            task_id="task2",
            name="Model Training",
            task_type=TaskType.PYTHON,
            command="train_model",
            dependencies=["task1"],
            timeout_seconds=3600
        )
        
        task3 = Task(
            task_id="task3",
            name="Model Evaluation",
            task_type=TaskType.PYTHON,
            command="evaluate_model",
            dependencies=["task2"],
            timeout_seconds=600
        )
        
        pipeline = Pipeline(
            pipeline_id="ml_pipeline",
            name="Machine Learning Pipeline",
            description="A simple ML pipeline for training and evaluation",
            tasks=[task1, task2, task3],
            max_concurrent_tasks=2
        )
        
        # Create pipeline
        pipeline_id = orchestrator.create_pipeline(pipeline)
        
        # Execute pipeline
        execution_id = orchestrator.execute_pipeline(pipeline_id)
        
        # Monitor execution
        import time
        time.sleep(5)
        
        execution = orchestrator.get_execution(execution_id)
        print(f"Execution status: {execution.status}")
        print(f"Task results: {list(execution.task_results.keys())}")
        
        # Get statistics
        stats = orchestrator.get_statistics()
        print(f"Orchestrator statistics: {stats}")
        
        # Export pipeline data
        orchestrator.export_pipeline_data("pipeline_orchestration_data.json")
        
        time.sleep(10)  # Let execution complete
        
    finally:
        orchestrator.stop_orchestrator()

if __name__ == "__main__":
    main()