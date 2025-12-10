# 🚀 TechFutures 2025 — End-to-End MLOps on Databricks

<p align="center">
  <img src="https://i0.wp.com/techfutures.com/wp-content/uploads/Techfutures_site-Version-6_logo-scaled.webp?fit=2560%2C1537&ssl=1"
       alt="TechFutures 2025 Logo"
       width="42%"
       style="vertical-align: middle; margin-right: 3%;">
  <img src="images/session-info-fixed.jpg"
       alt="Debu Sinha Session Info"
       width="42%"
       style="vertical-align: middle;">
</p>

Welcome to the **TechFutures 2025** hands-on workshop:  
**"End-to-End MLOps Pipelines on Databricks: From Model Training to Production."**

Led by **Debu Sinha — Lead Applied AI/ML Engineer at Databricks** and author of  
*Practical Machine Learning on Databricks*.

---

## 🧠 Workshop Overview

In this workshop, you'll learn how to build and scale a **complete MLOps lifecycle** on Databricks:

- Train and track models using **PyTorch + MLflow**
- Register and govern models with **Unity Catalog**
- Deploy and serve models using **Model Serving / AI Gateway**
- Automate experiments with **Optuna and MLflow nested runs**
- Extend the same workflow to **LLMs via Mosaic AI**

---

## 📚 Notebook Index

| # | Notebook | Description |
|:--:|:----------|:-------------|
| **1** | [`01_MLOps_Pipeline_on_Databricks.py`](./01_MLOps_Pipeline_on_Databricks.py) | Hands-on notebook showing an end-to-end pipeline — train, track, register and deploy a PyTorch model with MLflow. |
| **2** | [`02_Advanced_MLflow_PyTorch_Tutorial.py`](./02_Advanced_MLflow_PyTorch_Tutorial.py) | Advanced workflow covering Optuna tuning, Unity Catalog governance, and Spark batch inference. |
| *(Optional)* **3** | [`03_LLMOps_with_MosaicAI_and_MLflow.py` *(coming soon)*](./03_LLMOps_with_MosaicAI_and_MLflow.py) | Extends the same lifecycle to LLMs with prompt tracking and Mosaic AI Serving. |

---

## 🏗️ MLOps Lifecycle on Databricks

<img src="images/1.jpg" alt="Databricks MLOps Lifecycle" width="90%">

**Pipeline Stages**

1. Data Preparation (Delta Tables)  
2. Model Training (PyTorch + MLflow)  
3. Experiment Tracking (MLflow UI)  
4. Model Registry (Unity Catalog)  
5. Model Serving (REST API / AI Gateway)  
6. Monitoring (Lakehouse Monitoring + Alerts)

---

## 🤖 LLMOps Lifecycle — Unifying Generative AI Workflows

<img src="images/2.jpg" alt="Databricks LLMOps Lifecycle" width="90%">

**Lifecycle Flow**

1. Data (Prompts & Responses)  
2. Fine-Tuning (Mosaic AI Training)  
3. Tracking (MLflow Metrics + Prompts)  
4. Registry (Unity Catalog Models)  
5. Serving (Databricks Model Serving)  
6. Evaluation (LLM-as-a-Judge, Metrics, Drift)

> 🧩 *MLOps → LLMOps → AgentOps — a unified Databricks AI platform.*

### 🔬 Hands-On LLM Fine-Tuning Demos

To bridge the gap between traditional MLOps (Notebooks 1-2) and the upcoming LLMOps notebook, explore these **interactive fine-tuning examples** that demonstrate the same lifecycle principles applied to Large Language Models:

**[📖 Databricks LLM Fine-Tuning Demos](https://notebooks.databricks.com/demos/llm-fine-tuning/index.html)**

These demos showcase:
- **Classification Fine-Tuning** — Specialize models for customer support scenarios using the Chat API
- **RAG Chatbot Fine-Tuning** — Optimize conversational AI with retrieval augmentation and conversation history
- **Entity Extraction Fine-Tuning** — Train models for specialized extraction tasks (e.g., medical/drug entities)
- **MLflow Evaluation** — Measure fine-tuning improvements with built-in evaluation frameworks

**Why These Demos Matter:**
- Apply the **same MLflow + Unity Catalog patterns** from Notebooks 1-2 to LLM workflows
- See how model tracking, versioning, and serving extend to generative AI
- Practice evaluation techniques specific to LLMs (including LLM-as-a-Judge)
- Bridge traditional MLOps foundations with modern LLMOps practices

These demos complement Notebook 3 *(coming soon)* and provide immediate, hands-on practice with production LLM workflows on Databricks.

---

## ⚙️ Setup & Requirements

- Databricks workspace with **MLflow & Unity Catalog enabled**  
- Runtime: **Databricks Runtime ML 15.x or above**  
- Python 3.10 +  
- Optional: **Mosaic AI Model Serving** access

---

## 🚦 Run the Notebooks

Clone this repo or import directly into Databricks:

```bash
git clone https://github.com/debu-sinha/techfutures-2025-mlops-databricks.git
```

---

## 📖 Additional Resources

- **[Practical Machine Learning on Databricks](https://www.amazon.com/Practical-Data-Science-Databricks-end/dp/1801812039)** — Debu's comprehensive guide
- **[Databricks MLOps Workflows](https://docs.databricks.com/machine-learning/mlops/mlops-workflow.html)** — Official documentation
- **[The Big Book of MLOps](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)** — Free comprehensive guide

---

**Questions?** Reach out during the workshop or connect with Debu on [LinkedIn](https://www.linkedin.com/in/debusinha/)

---

*Made for TechFutures 2025 attendees*
