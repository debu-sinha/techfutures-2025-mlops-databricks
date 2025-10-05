# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction
# MAGIC
# MAGIC Welcome to **TechFutures 2025**!  This workshop is prepared and led by
# MAGIC **Debu Sinha**, Lead Applied AI/ML Engineer at Databricks.  In this
# MAGIC session we will build an end‑to‑end MLOps pipeline using Databricks.
# MAGIC By the end of the workshop you will:
# MAGIC
# MAGIC * Understand the core stages of a machine‑learning pipeline from data
# MAGIC   ingestion to production deployment.
# MAGIC * Use **PyTorch** to define and train a model on the Fashion‑MNIST
# MAGIC   dataset.
# MAGIC * Track experiments, parameters, and metrics using **MLflow**.
# MAGIC * Register a trained model in the **MLflow Model Registry**.
# MAGIC * Deploy the model for inference and test it on new data.
# MAGIC
# MAGIC Throughout the workshop we will insert interactive widgets so you
# MAGIC can adjust hyperparameters such as learning rate, batch size and
# MAGIC number of epochs.  Feel free to experiment and see how these
# MAGIC changes affect training results and logged metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Environment setup
# MAGIC
# MAGIC First we install and import the required packages.  Databricks
# MAGIC notebooks run on clusters with many libraries preinstalled, but
# MAGIC for reproducibility we explicitly install MLflow, PyTorch and
# MAGIC associated packages.  We also configure a few Databricks widgets
# MAGIC to capture hyperparameters from the user.

# COMMAND ----------

# MAGIC %pip install -q mlflow torch torchvision torchmetrics torchinfo
# MAGIC %restart_python

# COMMAND ----------

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
print(current_user)

# COMMAND ----------

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

# Databricks widgets for interactive hyperparameter tuning
dbutils.widgets.text("experiment_name", f"/Users/{current_user}/techfutures-mlops", "MLflow Experiment Name")
dbutils.widgets.text("learning_rate", "0.001", "Learning Rate")
dbutils.widgets.text("batch_size", "64", "Batch Size")
dbutils.widgets.text("epochs", "3", "Epochs")

experiment_name = dbutils.widgets.get("experiment_name")
learning_rate = float(dbutils.widgets.get("learning_rate"))
batch_size = int(dbutils.widgets.get("batch_size"))
epochs = int(dbutils.widgets.get("epochs"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data loading and exploration
# MAGIC
# MAGIC In this section we download the Fashion‑MNIST dataset, wrap it in
# MAGIC PyTorch `DataLoader` objects, and visualise a few samples.  Feel
# MAGIC free to substitute your own dataset here.  When you run the
# MAGIC notebook on Databricks, the dataset will be cached in the
# MAGIC cluster’s local storage for faster access on subsequent runs.

# COMMAND ----------

# Load Fashion‑MNIST data
training_data = datasets.FashionMNIST(
    root="/tmp/data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="/tmp/data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print(f"Training set size: {len(training_data)} samples")
print(f"Test set size: {len(test_data)} samples")

# Visualise a few examples
examples = [training_data[i][0] for i in range(6)]
class_names = training_data.classes

fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for idx, (img, ax) in enumerate(zip(examples, axes)):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(class_names[training_data[idx][1]])
    ax.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define the model
# MAGIC
# MAGIC We build a simple convolutional neural network (CNN) using PyTorch.
# MAGIC This architecture is intentionally compact so that we can train it
# MAGIC quickly during the workshop.  In your own projects you can
# MAGIC replace this with a deeper network or a completely different
# MAGIC architecture.  We also define training and evaluation functions.

# COMMAND ----------

class ImageClassifier(nn.Module):
    """Simple CNN for Fashion‑MNIST classification."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_epoch(dataloader, model, loss_fn, metrics_fn, optimizer, device, epoch):
    """Train the model for one epoch and log metrics to MLflow."""
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        acc = metrics_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            step = epoch * len(dataloader) + batch
            mlflow.log_metric("train_loss", loss.item(), step=step)
            mlflow.log_metric("train_accuracy", acc.item(), step=step)
            print(f"Epoch {epoch}, batch {batch}: loss={loss.item():.4f}, acc={acc.item():.4f}")


def evaluate(dataloader, model, loss_fn, metrics_fn, device, epoch):
    """Evaluate the model and log validation metrics."""
    model.eval()
    num_batches = 0
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            total_acc += metrics_fn(pred, y).item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    mlflow.log_metric("val_loss", avg_loss, step=epoch)
    mlflow.log_metric("val_accuracy", avg_acc, step=epoch)
    print(f"Validation — epoch {epoch}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Configure MLflow tracking
# MAGIC
# MAGIC MLflow allows us to track experiments, parameters, metrics and
# MAGIC artifacts.  In this cell we connect to the MLflow tracking server in
# MAGIC Databricks, set the experiment, and start a run.  When running on
# MAGIC Databricks the tracking server is automatically configured.  If
# MAGIC you are running this notebook locally, you can call `mlflow.login()`
# MAGIC to authenticate using your Databricks host and personal access
# MAGIC token (see docs for details).  The experiment name is pulled
# MAGIC from the widget defined in the setup cell.

# COMMAND ----------

# Optionally log in if running outside of Databricks
# mlflow.login()

# Configure MLflow to use Databricks as the tracking backend
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)

model = ImageClassifier().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    # Log a model summary as an artifact
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model, input_size=(batch_size, 1, 28, 28))))
    mlflow.log_artifact("model_summary.txt")

    # Training loop
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        train_epoch(train_dataloader, model, loss_fn, metric_fn, optimizer, device, epoch)
        evaluate(test_dataloader, model, loss_fn, metric_fn, device, epoch)

    # Capture a batch of inputs from the test dataloader for signature inference
    input_example, _ = next(iter(test_dataloader))  # shape [batch_size, 1, 28, 28]
    input_example = input_example.to(device)

    with torch.no_grad():
        example_output = model(input_example).cpu()

    # Convert to numpy for MLflow
    input_example_np = input_example.cpu().numpy()
    example_output_np = example_output.numpy()

    # Infer the model signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(input_example_np, example_output_np)

    # Log the trained model with signature and input example
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="TechFuturesDemoModel",
        signature=signature,
        input_example=input_example_np
    )

run_id = run.info.run_id
print(f"Finished training run.  Run ID: {run_id}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register and manage the model
# MAGIC
# MAGIC After logging the model, we promote it to the MLflow Model Registry.
# MAGIC The registry provides lifecycle management (staging, production,
# MAGIC archived) and version control for models.  If you logged the
# MAGIC model using the `registered_model_name` argument above, the first
# MAGIC version will automatically appear in the registry.  You can use
# MAGIC the following code to transition the model to the **Staging** or
# MAGIC **Production** stage.  Access control policies let you govern
# MAGIC who can promote or deploy models.
# MAGIC
# MAGIC Note: this step requires that the MLflow Model Registry is
# MAGIC available in your Databricks workspace.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
registered_name = "workspace.default.TechFuturesDemoModel"

# Find the latest version number by examining all registered versions
versions = client.search_model_versions(f"name = '{registered_name}'")
# Pick the highest version number (convert version strings to ints for comparison)
latest_version_info = max(versions, key=lambda v: int(v.version))
latest_version = latest_version_info.version
print(f"Latest registered version: {latest_version}")

# Assign an alias (e.g. 'staging') to the latest version
client.set_registered_model_alias(
    name=registered_name,
    alias="staging",
    version=latest_version
)
print(f"Assigned alias 'staging' to {registered_name} version {latest_version}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy and perform inference
# MAGIC
# MAGIC Once a model is in the registry, you can deploy it as a REST API
# MAGIC using Databricks Model Serving or load it for batch scoring.
# MAGIC Here we demonstrate loading the staged model back into memory and
# MAGIC performing predictions on a batch of test images.  In a
# MAGIC production setting you could call the model via the Databricks
# MAGIC serving endpoint or embed it in a streaming pipeline.

# COMMAND ----------

import pandas as pd

# Load the model from MLflow Model Registry
model_uri = f"models:/{registered_name}@staging"
loaded_model = mlflow.pytorch.load_model(model_uri)

# Grab a batch of test data
images, labels = next(iter(test_dataloader))
images = images.to(device)

with torch.no_grad():
    preds = loaded_model(images).cpu()

predicted_classes = preds.argmax(dim=1).numpy()
results_df = pd.DataFrame({"true": labels.numpy(), "pred": predicted_classes})

print(results_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best practices and next steps
# MAGIC
# MAGIC Congratulations!  You have built a simple but complete MLOps
# MAGIC pipeline on Databricks.  Some best practices and extensions to
# MAGIC consider:
# MAGIC
# MAGIC * **Experiment with hyperparameters**: Use Databricks widgets or
# MAGIC   [MLflow’s hyperparameter search](https://www.mlflow.org/docs/latest/model-tracking-git.html)
# MAGIC   to automatically run multiple training jobs and compare results.
# MAGIC * **Feature Store**: Leverage the Databricks Feature Store to create
# MAGIC   versioned, sharable features that can be used across multiple
# MAGIC   models.
# MAGIC * **Automated workflows**: Use Databricks Jobs or Workflows to
# MAGIC   schedule training and deployment pipelines end‑to‑end.
# MAGIC * **Monitoring**: Capture production metrics and set up alerts for
# MAGIC   data drift or model performance degradation.
# MAGIC
# MAGIC We hope this notebook serves as a useful template for your own
# MAGIC projects.  Thank you for participating in the **TechFutures**
# MAGIC workshop!