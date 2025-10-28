"""
Main CLI entry point for BharatFM
"""

import typer
from typing import Optional
import sys
import logging
from pathlib import Path

from .config import CLIConfig
from .commands import (
    TrainCommand, EvalCommand, DeployCommand, 
    LanguageCommand, GovernanceCommand, EducationCommand, FinanceCommand
)

# Create main Typer app
app = typer.Typer(
    name="bharat",
    help="Bharat Foundation Model Framework CLI",
    add_completion=False
)

# Global configuration
config = CLIConfig()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def version():
    """Show BharatFM version"""
    print("BharatFM v0.1.0")
    print("India's Open-Source Ecosystem for Building, Training, and Deploying Foundation Models")


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name"),
    path: str = typer.Option(".", "--path", help="Project path"),
    template: str = typer.Option("default", "--template", help="Project template")
):
    """Initialize a new BharatFM project"""
    from .init import init_project
    
    project_path = Path(path) / project_name
    
    try:
        init_project(project_name, project_path, template)
        typer.echo(f"✅ Project '{project_name}' initialized successfully at {project_path}")
    except Exception as e:
        typer.echo(f"❌ Error initializing project: {e}", err=True)
        sys.exit(1)


@app.command()
def train(
    config_path: str = typer.Option("config.json", "--config", help="Configuration file path"),
    model: str = typer.Option("bharat-base", "--model", help="Model to train"),
    dataset: str = typer.Option("indic_mix", "--dataset", help="Dataset to use"),
    steps: int = typer.Option(50000, "--steps", help="Number of training steps"),
    output_dir: str = typer.Option("./outputs", "--output-dir", help="Output directory"),
    distributed: bool = typer.Option(False, "--distributed", help="Enable distributed training"),
    deepspeed: bool = typer.Option(False, "--deepspeed", help="Enable DeepSpeed"),
    lora: bool = typer.Option(False, "--lora", help="Enable LoRA fine-tuning"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate", help="Learning rate"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    epochs: int = typer.Option(10, "--epochs", help="Number of epochs")
):
    """Train a BharatFM model"""
    try:
        train_cmd = TrainCommand(config)
        
        # Build training configuration
        train_config = {
            "model_name": model,
            "dataset_path": dataset,
            "max_steps": steps,
            "output_dir": output_dir,
            "distributed": distributed,
            "use_deepspeed": deepspeed,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": epochs,
            "use_lora": lora
        }
        
        # Load additional config from file if exists
        config_file = Path(config_path)
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                train_config.update(file_config)
        
        # Run training
        result = train_cmd.execute(train_config)
        typer.echo(f"✅ Training completed successfully")
        typer.echo(f"   Model saved to: {result.get('model_path', 'N/A')}")
        typer.echo(f"   Final loss: {result.get('final_loss', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@app.command()
def finetune(
    model: str = typer.Option("bharat-base", "--model", help="Base model to fine-tune"),
    dataset: str = typer.Option("govt_data", "--dataset", help="Fine-tuning dataset"),
    lora: bool = typer.Option(True, "--lora", help="Use LoRA fine-tuning"),
    lora_rank: int = typer.Option(16, "--lora-rank", help="LoRA rank"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", help="Learning rate"),
    epochs: int = typer.Option(3, "--epochs", help="Number of epochs"),
    output_dir: str = typer.Option("./finetuned", "--output-dir", help="Output directory")
):
    """Fine-tune a BharatFM model"""
    try:
        from .finetune import fine_tune_model
        
        # Build fine-tuning configuration
        finetune_config = {
            "base_model": model,
            "dataset_path": dataset,
            "use_lora": lora,
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
            "num_epochs": epochs,
            "output_dir": output_dir
        }
        
        # Run fine-tuning
        result = fine_tune_model(finetune_config)
        typer.echo(f"✅ Fine-tuning completed successfully")
        typer.echo(f"   Model saved to: {result.get('model_path', 'N/A')}")
        typer.echo(f"   Final accuracy: {result.get('final_accuracy', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Fine-tuning failed: {e}", err=True)
        sys.exit(1)


@app.command()
def eval(
    model: str = typer.Option(..., "--model", help="Model path to evaluate"),
    benchmarks: str = typer.Option("perplexity,generation_quality,multilingual_accuracy", "--benchmarks", help="Comma-separated list of benchmarks"),
    languages: str = typer.Option("hi,en,bn,ta,te,mr,gu,kn,ml,pa", "--languages", help="Comma-separated list of languages"),
    output_dir: str = typer.Option("./evaluation", "--output-dir", help="Output directory"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size"),
    save_predictions: bool = typer.Option(True, "--save-predictions", help="Save predictions")
):
    """Evaluate a BharatFM model"""
    try:
        eval_cmd = EvalCommand(config)
        
        # Parse benchmarks and languages
        benchmark_list = [b.strip() for b in benchmarks.split(",")]
        language_list = [l.strip() for l in languages.split(",")]
        
        # Build evaluation configuration
        eval_config = {
            "model_path": model,
            "benchmarks": benchmark_list,
            "languages": language_list,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "save_predictions": save_predictions
        }
        
        # Run evaluation
        result = eval_cmd.execute(eval_config)
        typer.echo(f"✅ Evaluation completed successfully")
        typer.echo(f"   Results saved to: {result.get('report_path', 'N/A')}")
        typer.echo(f"   Overall score: {result.get('overall_score', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        sys.exit(1)


@app.command()
def deploy(
    model: str = typer.Option(..., "--model", help="Model path to deploy"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host address"),
    port: int = typer.Option(8000, "--port", help="Port number"),
    engine: str = typer.Option("vllm", "--engine", help="Inference engine (vllm, native)"),
    workers: int = typer.Option(1, "--workers", help="Number of workers"),
    api_key: str = typer.Option(None, "--api-key", help="API key for authentication")
):
    """Deploy a BharatFM model as an API"""
    try:
        deploy_cmd = DeployCommand(config)
        
        # Build deployment configuration
        deploy_config = {
            "model_path": model,
            "host": host,
            "port": port,
            "engine": engine,
            "workers": workers,
            "api_key": api_key
        }
        
        # Run deployment
        deploy_cmd.execute(deploy_config)
        
    except KeyboardInterrupt:
        typer.echo("\n🛑 Deployment stopped by user")
    except Exception as e:
        typer.echo(f"❌ Deployment failed: {e}", err=True)
        sys.exit(1)


@app.command()
def serve(
    model: str = typer.Option(..., "--model", help="Model path to serve"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host address"),
    port: int = typer.Option(8001, "--port", help="Port number"),
    engine: str = typer.Option("vllm", "--engine", help="Inference engine (vllm, native)"),
    max_batch_size: int = typer.Option(32, "--max-batch-size", help="Maximum batch size")
):
    """Start inference server for a BharatFM model"""
    try:
        from .serve import start_inference_server
        
        # Build server configuration
        server_config = {
            "model_path": model,
            "host": host,
            "port": port,
            "engine": engine,
            "max_batch_size": max_batch_size
        }
        
        # Start server
        start_inference_server(server_config)
        
    except KeyboardInterrupt:
        typer.echo("\n🛑 Server stopped by user")
    except Exception as e:
        typer.echo(f"❌ Server failed: {e}", err=True)
        sys.exit(1)


@app.command()
def convert(
    model: str = typer.Option(..., "--model", help="Input model path"),
    output: str = typer.Option(..., "--output", help="Output model path"),
    format: str = typer.Option("pytorch", "--format", help="Output format (pytorch, onnx, tensorrt)"),
    quantization: str = typer.Option(None, "--quantization", help="Quantization method (int8, fp16, bf16)")
):
    """Convert a BharatFM model to different formats"""
    try:
        from .convert import convert_model
        
        # Build conversion configuration
        convert_config = {
            "input_model": model,
            "output_path": output,
            "format": format,
            "quantization": quantization
        }
        
        # Run conversion
        result = convert_model(convert_config)
        typer.echo(f"✅ Model conversion completed successfully")
        typer.echo(f"   Converted model saved to: {result.get('output_path', 'N/A')}")
        typer.echo(f"   Format: {result.get('format', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Model conversion failed: {e}", err=True)
        sys.exit(1)


@app.command()
def benchmark(
    model: str = typer.Option(..., "--model", help="Model path to benchmark"),
    dataset: str = typer.Option("benchmark_data", "--dataset", help="Benchmark dataset"),
    iterations: int = typer.Option(100, "--iterations", help="Number of iterations"),
    warmup: int = typer.Option(10, "--warmup", help="Number of warmup iterations"),
    output: str = typer.Option("benchmark_results.json", "--output", help="Output file")
):
    """Benchmark a BharatFM model"""
    try:
        from .benchmark import benchmark_model
        
        # Build benchmark configuration
        benchmark_config = {
            "model_path": model,
            "dataset_path": dataset,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "output_file": output
        }
        
        # Run benchmark
        result = benchmark_model(benchmark_config)
        typer.echo(f"✅ Benchmarking completed successfully")
        typer.echo(f"   Results saved to: {result.get('output_file', 'N/A')}")
        typer.echo(f"   Average latency: {result.get('avg_latency', 'N/A')} ms")
        typer.echo(f"   Throughput: {result.get('throughput', 'N/A')} tokens/sec")
        
    except Exception as e:
        typer.echo(f"❌ Benchmarking failed: {e}", err=True)
        sys.exit(1)


@app.command()
def list(
    resource: str = typer.Argument(..., help="Resource type (models, datasets, experiments)"),
    format: str = typer.Option("table", "--format", help="Output format (table, json)")
):
    """List available resources"""
    try:
        from .list import list_resources
        
        # Build list configuration
        list_config = {
            "resource_type": resource,
            "output_format": format
        }
        
        # Run list command
        result = list_resources(list_config)
        
        if format == "json":
            import json
            typer.echo(json.dumps(result, indent=2))
        else:
            # Format as table
            if resource == "models":
                typer.echo("Available Models:")
                typer.echo("-" * 50)
                for model in result:
                    typer.echo(f"  {model['name']:<20} {model['type']:<10} {model['size']:<10}")
            elif resource == "datasets":
                typer.echo("Available Datasets:")
                typer.echo("-" * 50)
                for dataset in result:
                    typer.echo(f"  {dataset['name']:<20} {dataset['languages']:<15} {dataset['size']:<10}")
            elif resource == "experiments":
                typer.echo("Available Experiments:")
                typer.echo("-" * 50)
                for exp in result:
                    typer.echo(f"  {exp['name']:<20} {exp['status']:<10} {exp['created']:<20}")
        
    except Exception as e:
        typer.echo(f"❌ Listing resources failed: {e}", err=True)
        sys.exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_key: str = typer.Option(None, "--set", help="Set configuration key (format: key=value)"),
    reset: bool = typer.Option(False, "--reset", help="Reset configuration to defaults")
):
    """Manage CLI configuration"""
    try:
        if show:
            config.show()
        elif set_key:
            if "=" not in set_key:
                typer.echo("❌ Invalid format. Use: key=value", err=True)
                sys.exit(1)
            
            key, value = set_key.split("=", 1)
            config.set(key, value)
            typer.echo(f"✅ Set {key} = {value}")
        elif reset:
            config.reset()
            typer.echo("✅ Configuration reset to defaults")
        else:
            typer.echo("Use --show to display configuration or --set key=value to set a value")
        
    except Exception as e:
        typer.echo(f"❌ Configuration management failed: {e}", err=True)
        sys.exit(1)


# Domain-specific command groups
language_app = typer.Typer(help="Language AI commands for multilingual applications")
governance_app = typer.Typer(help="Governance AI commands for policy and public services")
education_app = typer.Typer(help="Education AI commands for learning and tutoring")
finance_app = typer.Typer(help="Finance AI commands for financial analysis and audit")

# Register domain-specific apps
app.add_typer(language_app, name="language")
app.add_typer(governance_app, name="governance")
app.add_typer(education_app, name="education")
app.add_typer(finance_app, name="finance")


@language_app.command("train-chatbot")
def language_train_chatbot(
    languages: str = typer.Option("hi,en,bn", "--languages", help="Comma-separated list of languages"),
    dataset: str = typer.Option("government_schemes", "--dataset", help="Dataset to use"),
    model_name: str = typer.Option("bharat-lang-chatbot", "--model-name", help="Model name"),
    output_dir: str = typer.Option("./outputs/language_chatbot", "--output-dir", help="Output directory"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate", help="Learning rate"),
    epochs: int = typer.Option(5, "--epochs", help="Number of epochs")
):
    """Train a multilingual chatbot model"""
    try:
        lang_cmd = LanguageCommand(config)
        
        # Parse languages
        language_list = [l.strip() for l in languages.split(",")]
        
        # Build configuration
        lang_config = {
            "subcommand": "train-chatbot",
            "languages": language_list,
            "dataset": dataset,
            "model_name": model_name,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": epochs
        }
        
        # Run command
        result = lang_cmd.execute(lang_config)
        typer.echo(f"✅ Multilingual chatbot training completed")
        typer.echo(f"   Model saved to: {result.get('model_path', 'N/A')}")
        typer.echo(f"   Supported languages: {', '.join(result.get('supported_languages', []))}")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@language_app.command("deploy-translator")
def language_deploy_translator(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address"),
    port: int = typer.Option(8002, "--port", help="Port number"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Deploy translation engine application"""
    try:
        lang_cmd = LanguageCommand(config)
        
        # Build configuration
        lang_config = {
            "subcommand": "deploy-translator",
            "host": host,
            "port": port,
            "debug": debug
        }
        
        # Run command
        result = lang_cmd.execute(lang_config)
        typer.echo(f"✅ Translation engine deployed successfully")
        typer.echo(f"   Running on: {result.get('host', 'N/A')}:{result.get('port', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Deployment failed: {e}", err=True)
        sys.exit(1)


@governance_app.command("train-policy")
def governance_train_policy(
    dataset: str = typer.Option("government_policies", "--dataset", help="Dataset to use"),
    model_name: str = typer.Option("bharat-gov-policy", "--model-name", help="Model name"),
    output_dir: str = typer.Option("./outputs/governance_policy", "--output-dir", help="Output directory"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate", help="Learning rate"),
    epochs: int = typer.Option(3, "--epochs", help="Number of epochs")
):
    """Train policy analysis model"""
    try:
        gov_cmd = GovernanceCommand(config)
        
        # Build configuration
        gov_config = {
            "subcommand": "train-policy",
            "dataset": dataset,
            "model_name": model_name,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": epochs
        }
        
        # Run command
        result = gov_cmd.execute(gov_config)
        typer.echo(f"✅ Policy model training completed")
        typer.echo(f"   Model saved to: {result.get('model_path', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@governance_app.command("deploy-rti")
def governance_deploy_rti(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address"),
    port: int = typer.Option(8001, "--port", help="Port number"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Deploy RTI assistant application"""
    try:
        gov_cmd = GovernanceCommand(config)
        
        # Build configuration
        gov_config = {
            "subcommand": "deploy-rti",
            "host": host,
            "port": port,
            "debug": debug
        }
        
        # Run command
        result = gov_cmd.execute(gov_config)
        typer.echo(f"✅ RTI assistant deployed successfully")
        typer.echo(f"   Running on: {result.get('host', 'N/A')}:{result.get('port', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Deployment failed: {e}", err=True)
        sys.exit(1)


@education_app.command("train-tutor")
def education_train_tutor(
    dataset: str = typer.Option("ncert_content", "--dataset", help="Dataset to use"),
    model_name: str = typer.Option("bharat-edu-tutor", "--model-name", help="Model name"),
    subjects: str = typer.Option("mathematics,science,social_studies", "--subjects", help="Comma-separated list of subjects"),
    output_dir: str = typer.Option("./outputs/education_tutor", "--output-dir", help="Output directory"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate", help="Learning rate"),
    epochs: int = typer.Option(5, "--epochs", help="Number of epochs")
):
    """Train AI tutor model"""
    try:
        edu_cmd = EducationCommand(config)
        
        # Parse subjects
        subject_list = [s.strip() for s in subjects.split(",")]
        
        # Build configuration
        edu_config = {
            "subcommand": "train-tutor",
            "dataset": dataset,
            "model_name": model_name,
            "subjects": subject_list,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": epochs
        }
        
        # Run command
        result = edu_cmd.execute(edu_config)
        typer.echo(f"✅ AI tutor training completed")
        typer.echo(f"   Model saved to: {result.get('model_path', 'N/A')}")
        typer.echo(f"   Supported subjects: {', '.join(result.get('subjects', []))}")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@education_app.command("generate-content")
def education_generate_content(
    model: str = typer.Option(..., "--model", help="Model path"),
    topic: str = typer.Option(..., "--topic", help="Content topic"),
    subject: str = typer.Option(..., "--subject", help="Subject area"),
    grade_level: str = typer.Option("10", "--grade-level", help="Grade level"),
    content_type: str = typer.Option("explanation", "--content-type", help="Content type (explanation, example, exercise)"),
    output_file: str = typer.Option(None, "--output-file", help="Output file path")
):
    """Generate educational content"""
    try:
        edu_cmd = EducationCommand(config)
        
        # Build configuration
        edu_config = {
            "subcommand": "generate-content",
            "model_path": model,
            "topic": topic,
            "subject": subject,
            "grade_level": grade_level,
            "content_type": content_type,
            "output_file": output_file
        }
        
        # Run command
        result = edu_cmd.execute(edu_config)
        typer.echo(f"✅ Educational content generated")
        typer.echo(f"   Topic: {result.get('topic', 'N/A')}")
        typer.echo(f"   Subject: {result.get('subject', 'N/A')}")
        typer.echo(f"   Content length: {result.get('content_length', 'N/A')} characters")
        if result.get('output_file'):
            typer.echo(f"   Saved to: {result.get('output_file')}")
        
    except Exception as e:
        typer.echo(f"❌ Content generation failed: {e}", err=True)
        sys.exit(1)


@finance_app.command("train-analyst")
def finance_train_analyst(
    dataset: str = typer.Option("financial_statements", "--dataset", help="Dataset to use"),
    model_name: str = typer.Option("bharat-fin-analyst", "--model-name", help="Model name"),
    output_dir: str = typer.Option("./outputs/finance_analyst", "--output-dir", help="Output directory"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate", help="Learning rate"),
    epochs: int = typer.Option(4, "--epochs", help="Number of epochs")
):
    """Train financial analyst model"""
    try:
        fin_cmd = FinanceCommand(config)
        
        # Build configuration
        fin_config = {
            "subcommand": "train-analyst",
            "dataset": dataset,
            "model_name": model_name,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": epochs
        }
        
        # Run command
        result = fin_cmd.execute(fin_config)
        typer.echo(f"✅ Financial analyst training completed")
        typer.echo(f"   Model saved to: {result.get('model_path', 'N/A')}")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@finance_app.command("audit-transactions")
def finance_audit_transactions(
    model: str = typer.Option(..., "--model", help="Model path"),
    transactions_file: str = typer.Option(..., "--transactions-file", help="Transactions JSON file"),
    output_file: str = typer.Option("./audit_report.json", "--output-file", help="Output report file")
):
    """Audit financial transactions for anomalies"""
    try:
        fin_cmd = FinanceCommand(config)
        
        # Build configuration
        fin_config = {
            "subcommand": "audit-transactions",
            "model_path": model,
            "transactions_file": transactions_file,
            "output_file": output_file
        }
        
        # Run command
        result = fin_cmd.execute(fin_config)
        typer.echo(f"✅ Transaction audit completed")
        typer.echo(f"   Transactions analyzed: {result.get('transactions_analyzed', 'N/A')}")
        typer.echo(f"   Anomalies detected: {result.get('anomalies_detected', 'N/A')}")
        typer.echo(f"   Risk level: {result.get('risk_level', 'N/A')}")
        if result.get('output_file'):
            typer.echo(f"   Report saved to: {result.get('output_file')}")
        
    except Exception as e:
        typer.echo(f"❌ Audit failed: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()