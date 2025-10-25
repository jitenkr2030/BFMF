# Bharat Foundation Model Framework (BFMF)

India's Open-Source Ecosystem for Building, Training, and Deploying Foundation Models

## 🧭 Vision

To empower India's AI independence by providing an open, modular, and scalable foundation model framework — enabling developers, researchers, and institutions to build **Indian-language, domain-specific, and locally governed AI systems**.

## ⚙️ Core Philosophy

* 🇮🇳 **Built for Bharat** – Native support for Indian languages, datasets, and regional diversity.
* 🧠 **Foundation First** – Focused on base model pretraining, fine-tuning, and adaptation.
* 🔒 **Sovereign AI** – 100% self-hostable, privacy-first, and data-residency compliant.
* 🌍 **Open Collaboration** – Interoperable with Hugging Face, OpenAI-style APIs, and ExoStack.
* 🧩 **Modular Stack** – Each layer can work independently or as part of a complete pipeline.

## 🎯 Use Cases & Capabilities

BFMF supports **10 major use case categories** with comprehensive, production-ready implementations:

### 🗣️ 1. National & Regional Language AI
**Multilingual models for 22+ Indian languages with code-switching support**

- **AI Chatbots** for citizens to access government schemes in their language
- **Multilingual translation engine** for official documents
- **Voice-to-text transcription** for rural call centers, police stations, hospitals
- **AI news summarization** across Indian states

**Quick Start:**
```bash
# Train multilingual chatbot
bharat language train-chatbot --languages hi,en,bn --dataset government_schemes

# Deploy translation engine
bharat language deploy-translator --host 0.0.0.0 --port 8002
```

### 🏛️ 2. Digital Governance & Public Policy AI
**AI-powered policy analysis and public service automation**

- **Policy document drafting** and analysis
- **RTI response generation** and management
- **Citizen grievance redressal** systems
- **Compliance auditing** and risk assessment

**Quick Start:**
```bash
# Train policy analysis model
bharat governance train-policy --dataset government_policies

# Deploy RTI assistant
bharat governance deploy-rti --host 0.0.0.0 --port 8001
```

### 🎓 3. Education & Skill Development AI
**Personalized learning and educational content generation**

- **AI Teachers (VidyaYantra)** for personalized tutoring
- **Automated question generation** from NCERT/university content
- **Educational content generation** for multiple subjects and grades
- **Digital classroom** deployment with progress tracking

**Quick Start:**
```bash
# Train AI tutor
bharat education train-tutor --dataset ncert_content --subjects mathematics,science

# Generate educational content
bharat education generate-content --model ./models/tutor --topic "Photosynthesis" --subject "Biology"
```

### 💰 4. Finance, Accounting & Audit AI
**Financial analysis and audit automation**

- **Financial statement analysis** and forecasting
- **Transaction auditing** and anomaly detection
- **Tax compliance checking** using ICAI standards
- **Audit report generation** and risk assessment

**Quick Start:**
```bash
# Train financial analyst
bharat finance train-analyst --dataset financial_statements

# Audit transactions
bharat finance audit-transactions --model ./models/analyst --transactions-file transactions.json
```

### 🏥 5. Healthcare & Public Health AI
**Medical AI for rural telemedicine and public health**

- **AI symptom checkers** for rural telemedicine platforms
- **Medical record summarization** in multiple languages
- **Drug interaction analysis** and health recommendations
- **Public health monitoring** and outbreak prediction

### 🌾 6. Agriculture & Rural Development AI
**AI-driven insights for farmers and rural development**

- **Voice-based advisory bots** on weather, soil, and crop prices
- **AI translation** of agricultural research
- **Yield prediction** using satellite and sensor data
- **Scheme recommendation** for PM-KISAN, PMFBY, etc.

### 📰 7. Media, Journalism & Research AI
**Content generation and fact verification for media**

- **AI news summarization** in regional languages
- **Bias detection** and fact verification systems
- **Academic research assistants** for universities
- **Content generation** for regional media

### 🧩 8. Open-Source AI Research & Academia
**Sandbox for LLM research and experimentation**

- **Custom fine-tuning** experiments via ExoStack
- **Dataset creation** for local languages and culture
- **Student competitions** for building Bharat-model variants
- **AI ethics** and safety research

### 🔐 9. Sovereign AI Cloud Infrastructure
**Self-reliant AI hosting and compute**

- **VillageCloud / DesiCompute** node deployment
- **State-level AI clusters** (JharkhandAI, KeralaAI, etc.)
- **ExoStack Federation** across universities and startups
- **National AI compute registry**

### 🧠 10. Enterprise & Startup Use
**Custom AI applications for businesses**

- **Domain-specific fine-tuning** (legal, education, retail)
- **Plug-and-play LLM APIs** (Indian alternative to OpenAI)
- **Private AI deployment** with no foreign data transfer
- **LLM-as-a-Service** (LLaaS) for SMBs

## 🏗️ High-Level Architecture

```
 ┌───────────────────────────────────────────────┐
 │                BharatFM Stack                 │
 ├───────────────────────────────────────────────┤
 │ 7. Governance & Registry (MLflow, Audit, ACL) │
 │ 6. Serving & Deployment (vLLM, ExoStack)      │
 │ 5. Evaluation & Benchmark (HELM, lm-eval)     │
 │ 4. Fine-tuning Interface (Axolotl, LoRA)      │
 │ 3. Training Engine (Deepspeed, Megatron)      │
 │ 2. Model Architectures (GLM, LLaMA, Mistral)  │
 │ 1. Data Layer (Indic Corpora, RedPajama)      │
 └───────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm

# Install in development mode
pip install -e ".[dev]"

# Or install with specific extras
pip install -e ".[train,eval,deploy]"
```

### Basic Usage

#### 1. Train Bharat-Lite (1.3B) for Hindi-English Chat

```bash
bharat train --model bharat-lite --dataset indic_mix --steps 50000
```

#### 2. Domain-Specific Training

```bash
# Language AI
bharat language train-chatbot --languages hi,en,bn --dataset government_schemes

# Governance AI
bharat governance train-policy --dataset government_policies

# Education AI
bharat education train-tutor --dataset ncert_content --subjects mathematics,science

# Finance AI
bharat finance train-analyst --dataset financial_statements
```

#### 3. Deploy Applications

```bash
# Deploy multilingual chatbot
bharat language deploy-translator --host 0.0.0.0 --port 8002

# Deploy RTI assistant
bharat governance deploy-rti --host 0.0.0.0 --port 8001

# Deploy standard model API
bharat deploy --model ./models/bharat-base --host 0.0.0.0 --port 8000
```

#### 4. Evaluate Models

```bash
# Standard evaluation
bharat eval --model ./models/bharat-base --benchmarks perplexity,generation_quality --languages hi,en

# Domain-specific evaluation
bharat language evaluate-models --model ./models/bharat-lang --languages hi,en,bn
```

## 📁 Project Structure

```
bharat-fm/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── bharat_data/           # Data preparation and preprocessing
│   ├── datasets/
│   ├── tokenizers/
│   └── preprocess.py
│
├── bharat_model/          # Model architectures
│   ├── config/
│   ├── modeling_glm.py
│   ├── modeling_llama.py
│   └── modeling_moe.py
│
├── bharat_train/          # Training and fine-tuning
│   ├── trainer.py
│   ├── finetune.py
│   └── deepspeed_config.json
│
├── bharat_eval/           # Evaluation and benchmarking
│   ├── benchmarks/
│   └── evaluator.py
│
├── bharat_deploy/         # Serving and deployment
│   ├── api.py
│   └── inference_server.py
│
├── bharat_registry/       # Model registry and versioning
│   └── mlflow_utils.py
│
├── bharat_cli/            # Command-line interface
│   └── main.py
│
├── bharat_domains/        # Domain-specific modules
│   ├── language/          # Language AI
│   ├── governance/        # Governance AI
│   ├── education/         # Education AI
│   ├── finance/           # Finance AI
│   ├── healthcare/        # Healthcare AI
│   ├── agriculture/       # Agriculture AI
│   ├── media/             # Media AI
│   ├── research/          # Research AI
│   ├── infrastructure/    # Infrastructure AI
│   └── enterprise/        # Enterprise AI
│
├── bharat_apps/           # Ready-to-deploy applications
│   ├── language_apps/     # Language applications
│   ├── governance_apps/   # Governance applications
│   ├── education_apps/    # Education applications
│   ├── finance_apps/      # Finance applications
│   └── ...
│
├── docs/                  # Documentation
│   ├── architecture.png
│   └── user_guide.md
│
├── examples/              # Example scripts and configs
│   └── config_examples/
│
└── tests/                 # Test suite
    ├── test_data.py
    ├── test_model.py
    └── test_train.py
```

## 🧩 Modular Components

| Module | Description | Tools / Dependencies |
| --- | --- | --- |
| **bharat_data** | Prepares multilingual datasets, tokenizers, and cleaning pipelines | Hugging Face Datasets, IndicNLP, Dolma |
| **bharat_model** | Defines base model architectures (decoder-only, encoder-decoder, mixture-of-experts) | PyTorch, Transformers, Megatron-LM |
| **bharat_train** | Distributed pretraining and fine-tuning pipeline | Deepspeed, Axolotl, FSDP |
| **bharat_eval** | Evaluation and benchmarking suite | HELM, lm-eval-harness, OpenCompass |
| **bharat_deploy** | Serving layer using vLLM or ExoStack | FastAPI, Triton, vLLM |
| **bharat_registry** | Model registry, versioning, and experiment tracking | MLflow, Hugging Face Hub |
| **bharat_cli** | CLI toolkit for job scheduling, config management | Typer, ExoCLI |
| **bharat_domains** | Domain-specific models and datasets for each use case | Domain-specific libraries |
| **bharat_apps** | Ready-to-deploy applications for each use case | FastAPI, Streamlit, Gradio |

## 🧠 Supported Model Families

| Model | Type | Size | Purpose |
| --- | --- | --- | --- |
| **Bharat-Base** | Decoder-only | 1.3B / 7B | General-purpose pre-trained model |
| **Bharat-Lite** | 1.3B | On-device / low-resource | |
| **Bharat-MoE** | Mixture of Experts | 12×7B | Scalable modular architecture |
| **Bharat-Gov** | Finetuned | Governance, policy, public data | |
| **Bharat-Edu** | Finetuned | Education, tutoring, content generation | |
| **Bharat-Lang** | Finetuned | Multilingual & translation tasks | |
| **Bharat-Fin** | Finetuned | Financial analysis & audit | |

## 🌐 Integration with Other Frameworks

| Layer | Integrated With |
| --- | --- |
| Compute | ExoStack |
| Training | Axolotl, Deepspeed |
| Inference | vLLM, ExoServe |
| Registry | MLflow, Hugging Face Hub |
| Dataset | IndicNLP, RedPajama, Dolma |

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone and install
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
ruff check bharat_/
black bharat_/
```

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🏁 License

This project is licensed under the **Apache 2.0 / Bharat Open AI License (BOAL)**.

- ✅ **Commercial usage allowed**
- ✅ **Academic usage allowed** 
- ✅ **Modification and distribution allowed**
- 📝 **India-first attribution required**

See [LICENSE](LICENSE) for the full license text.

## 🤝 Community & Support

- **GitHub Discussions**: [Join the conversation](https://github.com/bharat-ai/bharat-fm/discussions)
- **Discord Community**: [Join our Discord](https://discord.gg/bharat-ai)
- **Documentation**: [Read the docs](docs/)
- **Issues**: [Report bugs or request features](https://github.com/bharat-ai/bharat-fm/issues)

## 🙏 Acknowledgments

- **Government of India** - For supporting sovereign AI initiatives
- **Indian AI Research Community** - For technical guidance and expertise
- **Open Source Community** - For building the foundation we build upon
- **Hugging Face** - For the amazing transformers ecosystem
- **ExoStack** - For compute infrastructure integration

## 📈 Citation

If you use BFMF in your research, please cite:

```bibtex
@software{bharat_fmf_2025,
  title={Bharat Foundation Model Framework: India's Open-Source Ecosystem for Sovereign AI},
  author={Bharat AI Team},
  year={2025},
  url={https://github.com/bharat-ai/bharat-fm},
  license={Apache 2.0 / BOAL}
}
```

---

## 🪔 Summary: Why BFMF Matters

| Impact Area | Benefit |
| --- | --- |
| **Language Inclusion** | 1 AI framework for 22+ languages |
| **Sovereign Data** | Built & hosted entirely within India |
| **Education & Skill** | AI literacy through BharatEdu models |
| **Innovation** | Open playground for students & startups |
| **Economic Growth** | Enables AI-first Bharat economy |

---

<div align="center">

**🇮🇳 Made with ❤️ for Bharat's AI Independence**

[![Back to top](https://img.shields.io/badge/Back%20to%20top-↑-blue.svg)](#readme)

</div>