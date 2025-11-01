# Strategic Partnership Program for India's AI Sovereignty

## Empowering India's AI Future Through Collaboration

The Bharat Foundation Model Framework (BFMF) represents a landmark initiative in India's journey towards AI independence. As a strategic, community-driven open-source project, we are developing sovereign foundation models that are uniquely Indian - trained on our data, built with cutting-edge open technologies, and deployed on domestic infrastructure.

To accelerate this national mission, we invite leading institutions, technology companies, research organizations, and visionary individuals to join us as strategic partners in building India's AI future.

## Partnership Opportunities

| Category | Strategic Value | Partner Examples |
|----------|----------------|------------------|
| **Computing Infrastructure** | High-performance GPU/TPU clusters for model training and optimization | ExoStack, Cloud Providers, HPC Centers |
| **Data Partnerships** | Curated datasets spanning Indian languages, domains, and use-cases | Research Institutions, Government Agencies |
| **Research & Development** | Advanced model architectures and domain-specific innovations | AI Institutes, Universities, R&D Labs |
| **Strategic Investment** | Sustainable funding for compute, talent, and infrastructure | Corporations, Investment Firms, Grants |
| **Technical Ecosystem** | Engineering expertise, tools, and platform capabilities | Technology Companies, Developer Communities |

## Partnership Benefits

### Strategic Advantages
- **Priority Access**: Early preview and integration rights for new Bharat models
- **Co-Innovation**: Joint R&D opportunities and technology collaboration
- **Market Leadership**: First-mover advantage in India's emerging AI ecosystem

### Institutional Benefits
- **Brand Recognition**: Featured placement on BFMF's partner ecosystem
- **Technical Support**: Dedicated integration and optimization assistance
- **Research Publications**: Co-authored papers and technical documentation

## Current Strategic Requirements (Q4 2025)

| Resource | Scope | Strategic Impact |
|----------|--------|------------------|
| Computing Infrastructure | 16-32 NVIDIA A100/H100 GPUs | Enable Bharat-Base (7B) model training |
| Storage Infrastructure | 50TB+ NVMe/Cloud Storage | Support dataset and model hosting |
| Financial Resources | ₹10-20L Investment | Scale compute and research operations |
| Research Partnerships | 5-10 Technical Partners | Drive model optimization and evaluation |

## Become a Strategic Partner

Join us in shaping India's AI future. Connect with our partnership team:

- **Email**: partnerships@bharat-ai.org
- **Strategic Discussions**: [Partner Inquiry Portal](https://github.com/bharat-ai/partnerships)
- **Partnership Portal**: https://bharat-ai.org/partners (launching soon)

Our team will provide detailed information about partnership opportunities, technical requirements, and engagement frameworks.

---

> "Every partnership strengthens India's path to AI sovereignty. Together, we're not just adopting AI - we're defining its future with Indian innovation and values."

Join us in building India's sovereign AI capabilities. The future of AI is being written in Bharat.

# 🇮🇳 Bharat Foundation Model Framework (BFMF)

<div align="center">

**India's Open-Source Ecosystem for Building, Training, and Deploying Foundation Models**

[![License](https://img.shields.io/badge/License-Apache%202.0%20%2F%20BOAL-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Docs](https://img.shields.io/badge/Docs-Coming%20Soon-green.svg)](docs/)
[![Community](https://img.shields.io/badge/Community-Join%20Us-orange.svg)](https://github.com/bharat-ai/bharat-fm/discussions)

</div>

---

## 🧭 Vision

> To empower India's AI independence by providing an open, modular, and scalable foundation model framework — enabling developers, researchers, and institutions to build **Indian-language, domain-specific, and locally governed AI systems**.

---

## ⚙️ Core Philosophy

* 🇮🇳 **Built for Bharat** – Native support for Indian languages, datasets, and regional diversity.
* 🧠 **Foundation First** – Focused on base model pretraining, fine-tuning, and adaptation.
* 🔒 **Sovereign AI** – 100% self-hostable, privacy-first, and data-residency compliant.
* 🌍 **Open Collaboration** – Interoperable with Hugging Face, OpenAI-style APIs, and ExoStack.
* 🧩 **Modular Stack** – Each layer can work independently or as part of a complete pipeline.

---

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

![BFMF Architecture](docs/architecture.png)

---

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

---

## 🧠 Supported Model Families

| Model | Type | Size | Purpose |
| --- | --- | --- | --- |
| **Bharat-Base** | Decoder-only | 1.3B / 7B | General-purpose pre-trained model |
| **Bharat-Lite** | 1.3B | On-device / low-resource | |
| **Bharat-MoE** | Mixture of Experts | 12×7B | Scalable modular architecture |
| **Bharat-Gov** | Finetuned | Governance, policy, public data | |
| **Bharat-Edu** | Finetuned | Education, tutoring, content generation | |
| **Bharat-Lang** | Finetuned | Multilingual & translation tasks | |

---

## 🌐 Integration with Other Frameworks

| Layer | Integrated With |
| --- | --- |
| Compute | ExoStack |
| Training | Axolotl, Deepspeed |
| Inference | vLLM, ExoServe |
| Registry | MLflow, Hugging Face Hub |
| Dataset | IndicNLP, RedPajama, Dolma |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (for 1.3B model training)

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
bharat train --model glm --dataset indic_mix --steps 50000
```

#### 2. Fine-tune Bharat-Gov for Policy AI

```bash
bharat finetune --model bharat-base --lora --dataset govt_data
```

#### 3. Deploy via ExoStack

```bash
bharat deploy --model bharat-gov --infra exostack --replicas 3
```

#### 4. Evaluate Model Performance

```bash
bharat eval --model bharat-base --benchmark helm --languages hi,en
```

---

## 📁 Project Structure

```
bharat-fm/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── bharat_data/
│   ├── datasets/
│   ├── tokenizers/
│   └── preprocess.py
│
├── bharat_model/
│   ├── config/
│   ├── modeling_glm.py
│   ├── modeling_llama.py
│   └── modeling_moe.py
│
├── bharat_train/
│   ├── trainer.py
│   ├── finetune.py
│   └── deepspeed_config.json
│
├── bharat_eval/
│   ├── benchmarks/
│   └── evaluator.py
│
├── bharat_deploy/
│   ├── api.py
│   └── inference_server.py
│
├── bharat_registry/
│   └── mlflow_utils.py
│
├── bharat_cli/
│   └── main.py
│
├── docs/
│   ├── architecture.png
│   └── user_guide.md
│
├── examples/
│   ├── hello_bharat.py
│   └── config_examples/
│
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_train.py
```

---

## 📘 Example Use Cases

### 1. Multilingual Chatbot

```python
from bharat_model import BharatLite
from bharat_deploy import InferenceServer

# Load pre-trained model
model = BharatLite.from_pretrained("bharat-ai/bharat-lite-1.3b")

# Create inference server
server = InferenceServer(model, host="0.0.0.0", port=8000)
server.start()
```

### 2. Custom Fine-tuning

```python
from bharat_train import FineTuner
from bharat_data import IndicDataset

# Load dataset
dataset = IndicDataset("hindi_english_pairs")

# Initialize fine-tuner
finetuner = FineTuner(
    base_model="bharat-base",
    lora_rank=16,
    learning_rate=2e-5
)

# Fine-tune
finetuner.train(dataset, epochs=3)
```

### 3. Model Evaluation

```python
from bharat_eval import Evaluator
from bharat_model import BharatBase

# Load model
model = BharatBase.from_pretrained("bharat-base")

# Initialize evaluator
evaluator = Evaluator(model)

# Run benchmarks
results = evaluator.evaluate(
    benchmarks=["helm", "lm-eval"],
    languages=["hi", "en", "bn"]
)

print(results)
```

---

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

---

## 🚀 Roadmap (Phase-wise)

| Phase | Goal | Timeline |
| --- | --- | --- |
| Phase 1 | Base repo setup + data pipeline + model configs | ✅ Nov 2025 |
| Phase 2 | Training engine + Axolotl integration | Dec 2025 |
| Phase 3 | Inference + deployment (ExoStack integration) | Jan 2026 |
| Phase 4 | Launch Bharat-Lite (1.3B multilingual) | Mar 2026 |
| Phase 5 | Community datasets + fine-tuned variants | Mid 2026 |

---

## 🏛️ Governance

### Steering Committee
- **Technical Lead**: AI Research Institute, India
- **Community Lead**: Open Source India Foundation
- **Industry Lead**: Bharat AI Consortium

### Decision Making Process
- [RFC Process](docs/rfcs/)
- [Community Meetings](community/meetings/)
- [Technical Advisory Board](governance/advisory-board.md)

---

## 📄 License

This project is licensed under the **Apache 2.0 / Bharat Open AI License (BOAL)**.

- ✅ **Commercial usage allowed**
- ✅ **Academic usage allowed** 
- ✅ **Modification and distribution allowed**
- 📝 **India-first attribution required**

See [LICENSE](LICENSE) for the full license text.

---

## 🤝 Community & Support

- **GitHub Discussions**: [Join the conversation](https://github.com/bharat-ai/bharat-fm/discussions)
- **Discord Community**: [Join our Discord](https://discord.gg/bharat-ai)
- **Documentation**: [Read the docs](docs/)
- **Issues**: [Report bugs or request features](https://github.com/bharat-ai/bharat-fm/issues)

---

## 🙏 Acknowledgments

- **Government of India** - For supporting sovereign AI initiatives
- **Indian AI Research Community** - For technical guidance and expertise
- **Open Source Community** - For building the foundation we build upon
- **Hugging Face** - For the amazing transformers ecosystem
- **ExoStack** - For compute infrastructure integration

---

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

<div align="center">

**🇮🇳 Made with ❤️ for Bharat's AI Independence**

[![Back to top](https://img.shields.io/badge/Back%20to%20top-↑-blue.svg)](#readme)

</div>
