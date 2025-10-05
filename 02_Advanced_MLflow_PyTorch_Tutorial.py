# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Advanced MLflow Pytorch Tutorial
# MAGIC # Author: Debu Sinha 
# MAGIC # email : debusinha2009@gmail.com
# MAGIC
# MAGIC In this comprehensive tutorial, we'll be covering the full lifecycle of experimentation, training, tuning, registration, evaluation, and deployment for a deep learning modeling project. You will be exposed to the latest quality-of-life APIs, recommended workflows, and discussions on how to simplify the process of building a PyTorch model while leveraging MLFlow to track every aspect of your development processes.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What You'll Learn
# MAGIC
# MAGIC In this step-by-step tutorial, you'll discover how to:
# MAGIC - **Generate and Visualize Data:** Create synthetic data to simulate real-world scenarios, and visualize feature relationships.
# MAGIC - **Design and Train Neural Networks:** Build a PyTorch neural network for regression and train it with proper optimization techniques.
# MAGIC - **Track with MLflow:** Log important metrics, parameters, artifacts, and models using MLflow, including visualizations.
# MAGIC - **Tune Hyperparameters:** Use Optuna for hyperparameter optimization with PyTorch models.
# MAGIC - **Register Models:** Register your model with Unity Catalog—Databricks' modern model registry—preparing it for review and future deployment.
# MAGIC - **Deploy Models:** Load your registered model, make predictions, and perform error analysis, both locally and in a distributed setting. 

# COMMAND ----------

# DBTITLE 1,(Optional) Install the latest version of MLflow
# MAGIC %pip install -Uqqq mlflow pytorch-lightning optuna skorch uv optuna-integration[pytorch_lightning]
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Importing Required Libraries
# MAGIC
# MAGIC We start by importing all the necessary libraries for our PyTorch deep learning workflow. These include:
# MAGIC
# MAGIC - **Data handling:** NumPy, Pandas, Scikit-learn
# MAGIC - **Visualization:** Matplotlib, Seaborn
# MAGIC - **Deep Learning:** PyTorch, PyTorch Lightning
# MAGIC - **Experimentation:** MLflow, Optuna
# MAGIC
# MAGIC The combination of these libraries provides a robust framework for developing, tracking, and deploying deep learning models.

# COMMAND ----------

from typing import Tuple, Optional, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param  

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Configuring the Model Registry with Unity Catalog
# MAGIC
# MAGIC Before diving into data exploration and model training, it's important to set up a robust system for managing your models. One of the key advantages of using MLflow on Databricks is the seamless integration with the **Unity Catalog**. This integration greatly simplifies model management and governance, ensuring that every model you develop is tracked, versioned, and secure.
# MAGIC
# MAGIC ### Why Unity Catalog?
# MAGIC
# MAGIC **Unity Catalog** is Databricks' unified governance solution for data and AI assets. Here's why it's a game changer for model development:
# MAGIC
# MAGIC - **Centralized Management:**  
# MAGIC   By storing models in Unity Catalog, you can keep all your models in the same system that your data is cataloged. This centralization makes it easier to manage and maintain consistency across your projects.
# MAGIC
# MAGIC - **Data and Model Lineage:**  
# MAGIC   Unity Catalog automatically tracks the lineage of your data and models. This means you can trace back a model to its original dataset and understand the series of transformations it underwent which is crucial for debugging and ensuring compliance.
# MAGIC
# MAGIC - **Role-Based Access Control (RBAC):**  
# MAGIC   Security is paramount in any collaborative environment. With RBAC, Unity Catalog enables you to control who can view, modify, or deploy models. This protects your valuable models from unauthorized changes and ensures that only the right team members have access.
# MAGIC
# MAGIC ### Setting the Registry URI
# MAGIC
# MAGIC The first step after importing your libraries is to configure MLflow to use Unity Catalog for model registration. This is done with a single line of code:
# MAGIC
# MAGIC ```python
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC ```
# MAGIC
# MAGIC This configuration doesn't change where your general logging happens. When you call any of the `log_<x>` APIs in MLflow, these are still either being stored in your Workspace or within Unity Catalog Volumes (depending on which path your specify when logging). 
# MAGIC
# MAGIC What this **does** change is where the models reside when they are registered. Without specifying Unity Catalog as your registry destination, you will be using the legacy Workspace-based model registry which will prevent you from using the UC-only lineage features.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Creating a Synthetic Regression Dataset
# MAGIC
# MAGIC We'll create a synthetic regression dataset that contains both linear and nonlinear relationships. This function generates data with meaningful patterns that our PyTorch model can learn from, including:
# MAGIC
# MAGIC - Linear relationships between features and target
# MAGIC - Nonlinear transformations (squared terms, interactions, sine functions)
# MAGIC - Controlled noise levels
# MAGIC - Features with varying importance
# MAGIC
# MAGIC This synthetic data allows us to demonstrate the full capabilities of PyTorch neural networks for regression tasks.

# COMMAND ----------

def create_regression_data(
    n_samples: int, 
    n_features: int,
    seed: int = 1994,
    noise_level: float = 0.3,
    nonlinear: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic regression data with interesting correlations for MLflow and PyTorch demonstrations.

    This function creates a DataFrame of continuous features and computes a target variable with nonlinear
    relationships and interactions between features. The data is designed to be complex enough to demonstrate
    the capabilities of deep learning, but not so complex that a reasonable model can't be learned.

    Args:
        n_samples (int): Number of samples (rows) to generate.
        n_features (int): Number of feature columns.
        seed (int, optional): Random seed for reproducibility. Defaults to 1994.
        noise_level (float, optional): Level of Gaussian noise to add to the target. Defaults to 0.3.
        nonlinear (bool, optional): Whether to add nonlinear feature transformations. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - pd.DataFrame: DataFrame containing the synthetic features.
            - pd.Series: Series containing the target labels.

    Example:
        >>> df, target = create_regression_data(n_samples=1000, n_features=10)
    """
    rng = np.random.RandomState(seed)
    
    # Generate random continuous features
    X = rng.uniform(-5, 5, size=(n_samples, n_features))
    
    # Create feature DataFrame with meaningful names
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    
    # Generate base target variable with linear relationship to a subset of features
    # We use only the first n_features//2 features to create some irrelevant features
    weights = rng.uniform(-2, 2, size=n_features//2)
    target = np.dot(X[:, :n_features//2], weights)
    
    # Add some nonlinear transformations if requested
    if nonlinear:
        # Add square term for first feature
        target += 0.5 * X[:, 0]**2
        
        # Add interaction between the second and third features
        if n_features >= 3:
            target += 1.5 * X[:, 1] * X[:, 2]
        
        # Add sine transformation of fourth feature
        if n_features >= 4:
            target += 2 * np.sin(X[:, 3])
        
        # Add exponential of fifth feature, scaled down
        if n_features >= 5:
            target += 0.1 * np.exp(X[:, 4] / 2)
            
        # Add threshold effect for sixth feature
        if n_features >= 6:
            target += 3 * (X[:, 5] > 1.5).astype(float)
    
    # Add Gaussian noise
    noise = rng.normal(0, noise_level * target.std(), size=n_samples)
    target += noise
    
    # Add a few more interesting features to the DataFrame
    
    # Add a correlated feature (but not used in target calculation)
    if n_features >= 7:
        df['feature_correlated'] = df['feature_0'] * 0.8 + rng.normal(0, 0.2, size=n_samples)
    
    # Add a cyclical feature
    df['feature_cyclical'] = np.sin(np.linspace(0, 4*np.pi, n_samples))
    
    # Add a feature with outliers
    df['feature_with_outliers'] = rng.normal(0, 1, size=n_samples)
    # Add outliers to ~1% of samples
    outlier_idx = rng.choice(n_samples, size=n_samples//100, replace=False)
    df.loc[outlier_idx, 'feature_with_outliers'] = rng.uniform(10, 15, size=len(outlier_idx))
    
    return df, pd.Series(target, name='target')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Exploratory Data Analysis (EDA) Visualizations
# MAGIC
# MAGIC Before building our neural network, we need to understand our data. We'll create several visualization functions to help us explore the dataset characteristics:
# MAGIC
# MAGIC - **Feature Distributions:** Histograms to visualize the distribution of each feature
# MAGIC - **Correlation Heatmap:** Visualize correlations between features and the target
# MAGIC - **Feature-Target Relationships:** Scatter plots to see how each feature relates to the target
# MAGIC - **Pairwise Relationships:** Visualize interactions between features
# MAGIC - **Outlier Detection:** Box plots to identify potential outliers
# MAGIC
# MAGIC These visualizations will not only help us understand the data but also serve as artifacts that can be logged with MLflow.

# COMMAND ----------

def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, n_cols: int = 3) -> plt.Figure:
    """
    Creates a grid of histograms for each feature in the dataset.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.
        n_cols (int): Number of columns in the grid layout.

    Returns:
        plt.Figure: The matplotlib Figure object containing the distribution plots.
    """
    features = X.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(X[feature], ax=ax, kde=True, color='skyblue')
            ax.set_title(f'Distribution of {feature}')
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Feature Distributions', y=1.02, fontsize=16)
    plt.close(fig)
    return fig

def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """
    Creates a correlation heatmap of all features and the target variable.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.

    Returns:
        plt.Figure: The matplotlib Figure object containing the heatmap.
    """
    # Combine features and target into one DataFrame
    data = X.copy()
    data['target'] = y
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw the heatmap with a color bar
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16)
    plt.close(fig)
    return fig

def plot_feature_target_relationships(X: pd.DataFrame, y: pd.Series, n_cols: int = 3) -> plt.Figure:
    """
    Creates a grid of scatter plots showing the relationship between each feature and the target.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.
        n_cols (int): Number of columns in the grid layout.

    Returns:
        plt.Figure: The matplotlib Figure object containing the relationship plots.
    """
    features = X.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            # Scatter plot with regression line
            sns.regplot(x=X[feature], y=y, ax=ax, 
                       scatter_kws={'alpha': 0.5, 'color': 'blue'}, 
                       line_kws={'color': 'red'})
            ax.set_title(f'{feature} vs Target')
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Feature vs Target Relationships', y=1.02, fontsize=16)
    plt.close(fig)
    return fig

def plot_pairwise_relationships(X: pd.DataFrame, y: pd.Series, features: list[str]) -> plt.Figure:
    """
    Creates a pairplot showing relationships between selected features and the target.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.
        features (List[str]): List of feature names to include in the plot.

    Returns:
        plt.Figure: The matplotlib Figure object containing the pairplot.
    """
    # Ensure features exist in the DataFrame
    valid_features = [f for f in features if f in X.columns]
    
    if not valid_features:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid features provided", ha='center', va='center')
        return fig
    
    # Combine selected features and target
    data = X[valid_features].copy()
    data['target'] = y
    
    # Create pairplot
    pairgrid = sns.pairplot(data, diag_kind="kde", 
                          plot_kws={"alpha": 0.6, "s": 50},
                          corner=True)
    
    pairgrid.fig.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=16)
    plt.close(pairgrid.fig)
    return pairgrid.fig

def plot_outliers(X: pd.DataFrame, n_cols: int = 3) -> plt.Figure:
    """
    Creates a grid of box plots to detect outliers in each feature.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        n_cols (int): Number of columns in the grid layout.

    Returns:
        plt.Figure: The matplotlib Figure object containing the outlier plots.
    """
    features = X.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            # Box plot to detect outliers
            sns.boxplot(x=X[feature], ax=ax, color='skyblue')
            ax.set_title(f'Outlier Detection for {feature}')
            ax.set_xlabel(feature)
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Outlier Detection for Features', y=1.02, fontsize=16)
    plt.close(fig)
    return fig

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    """
    Creates a residual plot to analyze model prediction errors.
    
    Args:
        y_true (pd.Series): True target values.
        y_pred (np.ndarray): Predicted target values.
        
    Returns:
        plt.Figure: The matplotlib Figure object containing the residual plot.
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of predicted values vs residuals
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='-')
    
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    
    plt.tight_layout()
    plt.close(fig)
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 Designing a PyTorch Neural Network for Regression
# MAGIC
# MAGIC Now we'll define our PyTorch model architecture. For regression tasks, we'll create a flexible neural network with the following characteristics:
# MAGIC
# MAGIC - **Configurable Architecture:** Adjustable number and size of hidden layers
# MAGIC - **Activation Functions:** ReLU for hidden layers, linear for output
# MAGIC - **Regularization:** Optional dropout for preventing overfitting
# MAGIC - **Layer Normalization:** For stabilizing training and accelerating convergence
# MAGIC
# MAGIC We'll implement this using both a standard PyTorch module and a PyTorch Lightning module to demonstrate different approaches.

# COMMAND ----------

class RegressionNN(nn.Module):
    """
    A flexible feedforward neural network for regression tasks.
    
    Attributes:
        input_dim (int): Number of input features.
        hidden_dims (List[int]): List of hidden layer dimensions.
        dropout_rate (float): Dropout probability for regularization.
        use_layer_norm (bool): Whether to use layer normalization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dims (List[int]): List of hidden layer dimensions.
            dropout_rate (float): Dropout probability for regularization.
            use_layer_norm (bool): Whether to use layer normalization.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Build layers dynamically based on hidden_dims
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
                
            layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = dim
        
        # Output layer (single output for regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x).squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters as a dictionary for MLflow logging."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm
        }

# COMMAND ----------

class RegressionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for regression tasks.
    
    This class wraps our RegressionNN model and adds training, validation,
    and testing logic using the PyTorch Lightning framework.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the Lightning module.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dims (List[int]): List of hidden layer dimensions.
            dropout_rate (float): Dropout probability for regularization.
            use_layer_norm (bool): Whether to use layer normalization.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for L2 regularization.
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create the model
        self.model = RegressionNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        
        # Calculate additional metrics
        rmse = torch.sqrt(loss)
        mae = torch.mean(torch.abs(y_pred - y))
        
        self.log('val_rmse', rmse, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        
        # Calculate metrics for test set
        rmse = torch.sqrt(loss)
        mae = torch.mean(torch.abs(y_pred - y))
        
        self.log('test_loss', loss)
        self.log('test_rmse', rmse)
        self.log('test_mae', mae)
        
        return loss
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters as a dictionary for MLflow logging."""
        return {
            "input_dim": self.hparams.input_dim,
            "hidden_dims": self.hparams.hidden_dims,
            "dropout_rate": self.hparams.dropout_rate,
            "use_layer_norm": self.hparams.use_layer_norm,
            "learning_rate": self.hparams.learning_rate,
            "weight_decay": self.hparams.weight_decay
        }

# COMMAND ----------

def prepare_dataloader(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int = 32
):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        X_test, y_test: Test data and labels.
        batch_size (int): Batch size for the DataLoaders.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Initialize a scaler
    scaler = StandardScaler()
    
    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors - explicitly set dtype to float32
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, scaler

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Standard Modeling Workflow
# MAGIC
# MAGIC Now we'll implement a standard PyTorch modeling workflow with MLflow integration:
# MAGIC
# MAGIC 1. **Generate and explore synthetic data**
# MAGIC 2. **Split the data into training, validation, and test sets**
# MAGIC 3. **Scale the data and create PyTorch DataLoaders**
# MAGIC 4. **Define and train a neural network model**
# MAGIC 5. **Evaluate the model's performance**
# MAGIC 6. **Log metrics, parameters, and artifacts to MLflow**
# MAGIC
# MAGIC This standard workflow will provide a baseline for our experiments, which we can later expand with hyperparameter tuning.

# COMMAND ----------

# Create the regression dataset
n_samples = 1000
n_features = 10
X, y = create_regression_data(n_samples=n_samples, n_features=n_features, nonlinear=True)

# Create EDA plots
dist_plot = plot_feature_distributions(X, y)
corr_plot = plot_correlation_heatmap(X, y)
scatter_plot = plot_feature_target_relationships(X, y)
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
top_features = corr_with_target.head(4).index.tolist()
pairwise_plot = plot_pairwise_relationships(X, y, top_features)
outlier_plot = plot_outliers(X)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Prepare DataLoaders
batch_size = 32
train_loader, val_loader, test_loader, scaler = prepare_dataloader(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size)

# Define model parameters
input_dim = X_train.shape[1]
hidden_dims = [64, 32]
dropout_rate = 0.1
use_layer_norm = True
learning_rate = 1e-3
weight_decay = 1e-5

# Create the PyTorch Lightning model
model = RegressionLightningModule(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    dropout_rate=dropout_rate,
    use_layer_norm=use_layer_norm,
    learning_rate=learning_rate,
    weight_decay=weight_decay
)

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkpoints',
    filename='pytorch-regression-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min'
)

# Define trainer
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[early_stopping, checkpoint_callback],
    enable_progress_bar=True,
    log_every_n_steps=5
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
test_results = trainer.test(model, test_loader)

# Make predictions on the test set for evaluation
model.eval()
test_preds = []
true_values = []

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        y_pred = model(x)
        test_preds.extend(y_pred.numpy())
        true_values.extend(y.numpy())

test_preds = np.array(test_preds)
true_values = np.array(true_values)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(true_values, test_preds))
mae = mean_absolute_error(true_values, test_preds)
r2 = r2_score(true_values, test_preds)

# Create residual plot
residual_plot = plot_residuals(pd.Series(true_values), test_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Logging and Evaluation with MLflow
# MAGIC
# MAGIC In this section, we integrate MLflow into our workflow to capture every detail of our experiment—from model parameters and metrics to artifacts such as visualizations and model signatures. By using MLflow's context manager, we ensure that the experiment run is automatically managed: the run is started when entering the block and properly terminated upon exit. This eliminates the need for manual run management and guarantees that all resources are cleanly closed and logged.
# MAGIC
# MAGIC ### Key Aspects of MLflow Logging
# MAGIC
# MAGIC - **Automatic Run Management:**  
# MAGIC   Leveraging the context manager (`with mlflow.start_run():`) simplifies the code and ensures that the MLflow run lifecycle is handled seamlessly. This means that once the block of code is executed, the run is finalized, and all logged artifacts, metrics, and parameters are securely stored.
# MAGIC
# MAGIC - **Metrics and Parameters Capture:**  
# MAGIC   During the run, critical metrics—such as the final RMSE values for both training and testing—are extracted and logged. In parallel, the model's hyperparameters and configuration settings are recorded. This dual logging of metrics and parameters is essential for understanding model performance and ensuring reproducibility.
# MAGIC
# MAGIC - **Model Signature Generation:**  
# MAGIC   A model signature is generated using MLflow's inference utility. This signature defines the expected input and output schema for the model, which is a necessary step for model registration in Unity Catalog. With a proper signature, the model can be easily queried and deployed, ensuring consistency between development and production environments.
# MAGIC
# MAGIC - **Artifact Logging:**  
# MAGIC   All supporting artifacts, including EDA visualizations such as feature distributions, correlation heatmaps, scatter plots, pairwise relationships, outlier detection, and residual analysis, are logged with MLflow. Capturing these visualizations provides valuable context for the model's performance and serves as a reference for any future troubleshooting or audits.
# MAGIC
# MAGIC - **Model Registration and Evaluation:**  
# MAGIC   The model is not only logged but also registered within Unity Catalog. This registration facilitates robust versioning and governance. Additionally, MLflow's evaluation capabilities are used to generate extra metrics automatically, standardizing the performance assessment of the model without additional manual implementation.

# COMMAND ----------

# Log the model and training results with MLflow
with mlflow.start_run() as run:
    # Create MLflow client for batch logging
    mlflow_client = MlflowClient()
    run_id = run.info.run_id
    
    # Extract metrics
    final_train_loss = trainer.callback_metrics.get("train_loss").item() if "train_loss" in trainer.callback_metrics else None
    final_val_loss = trainer.callback_metrics.get("val_loss").item() if "val_loss" in trainer.callback_metrics else None
    
    # Extract parameters for logging
    model_params = model.get_params()
     
    # Create a list to store all metrics for batch logging
    all_metrics = []
    
    # Add each metric to the list
    if final_train_loss is not None:
        all_metrics.append(Metric(key="train_loss", value=final_train_loss, timestamp=0, step=0))
    if final_val_loss is not None:
        all_metrics.append(Metric(key="val_loss", value=final_val_loss, timestamp=0, step=0))
    
    # Add test metrics
    all_metrics.append(Metric(key="test_rmse", value=rmse, timestamp=0, step=0))
    all_metrics.append(Metric(key="test_mae", value=mae, timestamp=0, step=0))
    all_metrics.append(Metric(key="test_r2", value=r2, timestamp=0, step=0))
    
    # Collect all parameters to log
    # Note: We'll continue using log_params for model_params since it could be many parameters,
    # but convert the individual param calls to batch
    from mlflow.entities import Param
    all_params = [
        Param(key="batch_size", value=str(batch_size)),
        Param(key="early_stopping_patience", value=str(early_stopping.patience)),
        Param(key="max_epochs", value=str(trainer.max_epochs)),
        Param(key="actual_epochs", value=str(trainer.current_epoch))
    ]
    
    # Generate a model signature using the infer signature utility in MLflow
    input_example = X_train.iloc[[0]].values.astype(np.float32)  # Ensure float32 type
    input_example_scaled = scaler.transform(input_example).astype(np.float32)
    
    model.eval()
    with torch.no_grad():
        # Ensure tensor is float32
        tensor_input = torch.tensor(input_example_scaled, dtype=torch.float32)
        signature_preds = model(tensor_input)
    
    # Ensure prediction is also float32
    signature = infer_signature(input_example, signature_preds.numpy().reshape(-1).astype(np.float32))
    
    # Log model parameters first (since these could be numerous)
    mlflow.log_params(model_params)
    
    # Log all metrics and remaining parameters in a single batch operation
    mlflow_client.log_batch(
        run_id=run_id,
        metrics=all_metrics,
        params=all_params
    )
    
    # Log the model to MLflow and register the model to Unity Catalog
    model_info = mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        registered_model_name="default.pytorch_regression_model",
    )
    
    # Log feature analysis plots
    mlflow.log_figure(dist_plot, "feature_distributions.png")
    mlflow.log_figure(corr_plot, "correlation_heatmap.png")
    mlflow.log_figure(scatter_plot, "feature_target_relationships.png")
    mlflow.log_figure(pairwise_plot, "pairwise_relationships.png")
    mlflow.log_figure(outlier_plot, "outlier_detection.png")
    mlflow.log_figure(residual_plot, "residual_plot.png")
    
    # Run MLflow evaluation to generate additional metrics without having to implement them
    evaluation_data = X_test.copy()
    evaluation_data["label"] = y_test
    
    # Skip mlflow.evaluate for now to avoid type mismatch issues
    # Instead, log the metrics directly
    print(f"Model logged: {model_info.model_uri}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Hyperparameter Tuning with Nested Runs and Optuna
# MAGIC
# MAGIC In this section, we take a deep dive into automating hyperparameter tuning by leveraging Optuna in conjunction with MLflow's nested run functionality. This setup enables an efficient and structured way to explore a range of parameter configurations while capturing all experimental details.
# MAGIC
# MAGIC ### What's Happening Under the Hood
# MAGIC
# MAGIC **1. Data Preparation and Setup:**  
# MAGIC We begin by generating a larger synthetic regression dataset and splitting it into training, validation, and testing sets. An evaluation dataset is prepared to later assess model performance. This step lays the foundation for a robust hyperparameter search.
# MAGIC
# MAGIC **2. Defining the Objective Function:**  
# MAGIC At the core of the hyperparameter tuning process is an objective function. Within this function, we define a search space for key hyperparameters of the PyTorch model, such as the number of layers, hidden dimensions, dropout rate, learning rate, and regularization parameters. Optuna's suggestion methods dynamically sample these values, ensuring that each trial tests a different combination of parameters.
# MAGIC
# MAGIC **3. Utilizing Nested MLflow Runs:**  
# MAGIC Inside the objective function, a nested MLflow run is initiated. This nested run automatically captures and logs all details specific to the current hyperparameter trial. By isolating each trial in its own nested run, we can keep a well-organized record of each configuration and its corresponding performance metrics. The nested run logs:
# MAGIC - The specific hyperparameters used for that trial.
# MAGIC - The performance metric (in this case, validation loss) computed on the validation set.
# MAGIC - The trained model instance is also stored as part of the trial's metadata, allowing easy retrieval of the best-performing model later.
# MAGIC
# MAGIC We purposefully do not record each model to MLflow. While doing hyperparameter tuning, each iteration is not guaranteed to be particularly good, so there is no reason to record the model artifact for something that we'll likely never use.
# MAGIC
# MAGIC **4. Automating the Search with Optuna:**  
# MAGIC With the objective function defined, we then set up an Optuna study aimed at minimizing the validation loss. Optuna runs a series of trials—each trial representing a unique combination of hyperparameters. During each trial, the nested MLflow run captures all the experiment details, making it simple to track and compare the performance of each model configuration.
# MAGIC
# MAGIC **5. Aggregating and Registering the Best Model:**  
# MAGIC Once the hyperparameter tuning process is complete, the study identifies the best trial based on the lowest validation loss. The best model, along with its optimal parameters, is then extracted. A model signature is generated to specify the expected input and output schema, which is crucial for consistent deployment and integration with systems like Unity Catalog. Finally, the best model is logged and registered, and additional artifacts such as EDA plots and feature importance charts are recorded to provide full context for the experiment.
# MAGIC
# MAGIC ### Bringing It All Together
# MAGIC
# MAGIC By combining Optuna's automatic hyperparameter search with MLflow's nested runs, we create a powerful and transparent workflow. Each hyperparameter trial is neatly encapsulated in its own nested run, providing a detailed audit trail of all experiments. This approach not only automates the tuning process but also ensures that every step—from data preparation and parameter sampling to model evaluation and registration—is reproducibly tracked and easily revisited.

# COMMAND ----------

# Create a custom pruning callback as a fallback
class PyTorchLightningPruningCallback(pl.Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    
    This is a simplified version for use when the optuna-integration package isn't available.
    """
    
    def __init__(self, trial, monitor):
        super().__init__()
        self._trial = trial
        self.monitor = monitor
        
    def on_validation_end(self, trainer, pl_module):
        # Report the validation metric to Optuna
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)
        
        if current_score is not None:
            self._trial.report(current_score.item(), trainer.current_epoch)
            
            # If trial should be pruned based on current value,
            # stop the training
            if self._trial.should_prune():
                message = "Trial was pruned at epoch {}.".format(trainer.current_epoch)
                raise optuna.TrialPruned(message)

# Generate a larger dataset for hyperparameter tuning
n_samples = 2000
n_features = 10

X, y = create_regression_data(n_samples=n_samples, n_features=n_features, nonlinear=True)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Prepare the evaluation data
evaluation_data = X_test.copy()
evaluation_data["label"] = y_test

# Create the data loaders
batch_size = 32
train_loader, val_loader, test_loader, scaler = prepare_dataloader(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size)

def objective(trial):
    """Optuna objective function to minimize validation loss."""
    
    # Define the hyperparameter search space
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    # Create hidden dimensions based on number of layers
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 16, 128))
    
    # Other hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])
    
    # Start a nested MLflow run for this trial
    with mlflow.start_run(nested=True) as child_run:
        # Create MLflow client for batch logging
        mlflow_client = MlflowClient()
        run_id = child_run.info.run_id
        
        # Prepare parameters for batch logging
        params_list = []
        param_dict = {
            "n_layers": n_layers,
            "hidden_dims": str(hidden_dims),  # Convert list to string
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "use_layer_norm": use_layer_norm,
            "batch_size": batch_size
        }
        
        # Convert parameters to Param objects
        for key, value in param_dict.items():
            params_list.append(Param(key, str(value)))
        
        # Create the model with these hyperparameters
        model = RegressionLightningModule(
            input_dim=X_train.shape[1],
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
        
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss"
        )
        
        # Define trainer with early stopping and pruning
        trainer = pl.Trainer(
            max_epochs=50,
            callbacks=[early_stopping, pruning_callback],
            enable_progress_bar=False,
            log_every_n_steps=10
        )
        
        # Train and validate the model
        trainer.fit(model, train_loader, val_loader)
        
        # Get the best validation loss
        best_val_loss = trainer.callback_metrics.get("val_loss").item()
        val_rmse = np.sqrt(best_val_loss)
        
        # Prepare metrics for batch logging
        current_time = int(time.time() * 1000)  # Current time in milliseconds
        metrics_list = [
            Metric("val_loss", best_val_loss, current_time, 0),
            Metric("val_rmse", val_rmse, current_time, 0)
        ]
        
        # Use log_batch through the client for efficient logging
        mlflow_client.log_batch(run_id, metrics=metrics_list, params=params_list)
        
    # Store the model in the trial's user attributes
    trial.set_user_attr("model", model)
    
    # Return the value to minimize (validation loss)
    return best_val_loss

best_model_version = None
# In the parent run, we will be storing the best iteration from the hyperparameter tuning execution
with mlflow.start_run() as run:
    # Create MLflow client for batch logging
    mlflow_client = MlflowClient()
    run_id = run.info.run_id
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_trial = study.best_trial
    best_model = best_trial.user_attrs["model"]
    
    # Test the best model
    trainer = pl.Trainer(
        enable_progress_bar=True,
        log_every_n_steps=5
    )
    test_results = trainer.test(best_model, test_loader)
    
    # Make predictions on the test set for evaluation
    best_model.eval()
    test_preds = []
    true_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_pred = best_model(x)
            test_preds.extend(y_pred.numpy())
            true_values.extend(y.numpy())
    
    test_preds = np.array(test_preds)
    true_values = np.array(true_values)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_values, test_preds))
    mae = mean_absolute_error(true_values, test_preds)
    r2 = r2_score(true_values, test_preds)
    
    # Prepare parameters for batch logging
    best_params_list = []
    for key, value in best_trial.params.items():
        best_params_list.append(Param(f"best_{key}", str(value)))
    
    # Prepare metrics for batch logging
    current_time = int(time.time() * 1000)  # Current time in milliseconds
    metrics_list = [
        Metric("best_val_loss", best_trial.value, current_time, 0),
        Metric("test_rmse", rmse, current_time, 0),
        Metric("test_mae", mae, current_time, 0),
        Metric("test_r2", r2, current_time, 0)
    ]
    
    # Log metrics and parameters in a single batch call
    mlflow_client.log_batch(run_id, metrics=metrics_list, params=best_params_list)

    # Generate model signature - ensure consistent float32 types
    input_example = X_train.iloc[[0]].values.astype(np.float32)
    input_example_scaled = scaler.transform(input_example).astype(np.float32)
    
    best_model.eval()
    with torch.no_grad():
        tensor_input = torch.tensor(input_example_scaled, dtype=torch.float32)
        signature_preds = best_model(tensor_input)
    
    signature = infer_signature(input_example, signature_preds.numpy().reshape(-1).astype(np.float32))

    # Log and register the PyTorch model
    model_info = mlflow.pytorch.log_model(
        best_model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        registered_model_name="default.pytorch_regression_optimized",
    )
    
    # Create residual plot
    residual_plot = plot_residuals(pd.Series(true_values), test_preds)
    
    # Log figures (no batch equivalent for figures)
    mlflow.log_figure(dist_plot, "feature_distributions.png")
    mlflow.log_figure(corr_plot, "correlation_heatmap.png")
    mlflow.log_figure(scatter_plot, "feature_target_relationships.png")
    mlflow.log_figure(pairwise_plot, "pairwise_relationships.png")
    mlflow.log_figure(outlier_plot, "outlier_detection.png")
    mlflow.log_figure(residual_plot, "residual_plot.png")

    # Skip mlflow.evaluate for now to avoid type mismatch issues
    # Instead, log the metrics directly
    print(f"Best model logged: {model_info.model_uri}")
    print(f"Best parameters: {best_trial.params}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    best_model_version = model_info.registered_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 What Gets Logged with Your Model
# MAGIC
# MAGIC When logging a model with MLflow on Databricks, a variety of important artifacts and metadata are captured. This ensures that your model is not only reproducible but also ready for deployment with all necessary dependencies and clear API contracts. Here's what is logged along with the model:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Signature**
# MAGIC
# MAGIC - **Purpose:**  
# MAGIC   The model signature enforces the API contract by defining the schema of the input data and expected output. This is crucial when serving models, as it ensures that the inputs provided during deployment match what the model was trained on.
# MAGIC
# MAGIC - **How It's Obtained:**  
# MAGIC   The signature is inferred automatically (using the `infer_signature` function) based on a sample of input data and the corresponding predictions.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **PyPI Requirements**
# MAGIC
# MAGIC - **Inferred Requirements:**  
# MAGIC   MLflow automatically infers the necessary Python package dependencies (PyPI requirements) for your model. This ensures that the serving environment will have all the libraries needed to run the model.
# MAGIC
# MAGIC - **Extra Requirements:**  
# MAGIC   You can also specify additional requirements via the `extra_pip_requirements` argument if your model depends on packages that are not captured automatically.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Custom Code Dependencies**
# MAGIC
# MAGIC - **Custom Scripts:**  
# MAGIC   With the `code_paths` argument, you have the option to include custom code dependencies. This is useful when your model relies on local scripts or modules that are not part of standard packages.
# MAGIC
# MAGIC - **Use Case:**  
# MAGIC   This flexibility allows you to log additional code files that your model requires, ensuring that all custom logic is preserved and deployed along with the model.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Input Example**
# MAGIC
# MAGIC - **Purpose:**  
# MAGIC   An input example is logged to provide a clear indication in the MLflow UI of how to call the model when serving it. This example helps users understand the expected format of inputs during deployment.
# MAGIC
# MAGIC - **Benefit:**  
# MAGIC   This is particularly useful during model serving, as it guides the configuration of the serving endpoint and aids in validating that the model receives the correct input structure.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Additional Metadata**
# MAGIC
# MAGIC - **Contextual Information:**  
# MAGIC   Extra metadata can be logged to provide more context about the model, such as details about the training environment, specific configuration parameters, or notes on the experiment.
# MAGIC
# MAGIC - **Benefit for Deployment:**  
# MAGIC   This metadata enhances transparency and aids in troubleshooting, as anyone reviewing the model can understand the conditions under which it was logged.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### How This is Integrated into the Model Logging Process
# MAGIC
# MAGIC When you call `mlflow.pytorch.log_model`, all the above components can be included by specifying the appropriate arguments:
# MAGIC
# MAGIC - **`signature`:** The inferred model signature.
# MAGIC - **`input_example`:** A sample input row from your dataset.
# MAGIC - **`extra_pip_requirements`:** Any additional PyPI packages needed.
# MAGIC - **`code_paths`:** Paths to custom code dependencies.
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC ```python
# MAGIC signature = infer_signature(input_example, predictions)
# MAGIC model_info = mlflow.pytorch.log_model(
# MAGIC     model,
# MAGIC     artifact_path="Volumes/PyTorchDemo/model",
# MAGIC     input_example=input_example,
# MAGIC     signature=signature,
# MAGIC     registered_model_name="catalog.schema.pytorch_model",
# MAGIC     extra_pip_requirements=["pytorch-lightning==2.0.0"],
# MAGIC     code_paths=["path/to/custom_code.py"]
# MAGIC )
# MAGIC ```end process of moving a model from development to production-ready status.

# COMMAND ----------

from mlflow import MlflowClient

# Initialize MLflow client
client = MlflowClient()

# Set a human-readable alias for our best model version
# This makes it easier to reference specific model versions programmatically
client.set_registered_model_alias("default.pytorch_regression_optimized", "best", int(best_model_version))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validating the Model with mlflow.models.predict
# MAGIC
# MAGIC Before moving to production inference, we should validate that our logged model is ready for deployment. MLflow provides the `mlflow.models.predict` utility to simulate a production environment and ensure the model works correctly.

# COMMAND ----------

# Reference the model by its alias
model_uri = "models:/default.pytorch_regression_optimized@best"

# Validate the model's deployment readiness
mlflow.models.predict(model_uri=model_uri, input_data=X_test, env_manager="virtualenv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the Registered Model for Local Predictions
# MAGIC
# MAGIC We can load our registered model and use it for local predictions. This is useful for testing or for batch inference scenarios.

# COMMAND ----------

# Convert the data type of X_test to float32
X_test = X_test.astype('float32')

# Load the model using the pyfunc interface (recommended for deployment)
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Make predictions with the loaded model
predictions = loaded_model.predict(X_test)

print(f"Shape of predictions: {predictions.shape}")
print(f"First 5 predictions: {predictions[:5]}")
print(f"First 5 actual values: {y_test.values[:5]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Prediction with Spark UDF
# MAGIC
# MAGIC For large-scale predictions, we can convert our model to a Spark UDF and apply it to a Spark DataFrame, enabling distributed inference.

# COMMAND ----------

from pyspark.sql.functions import array, col

X_spark = spark.createDataFrame(X_test)

X_spark_with_array = X_spark.withColumn(
    "features_array",
    array(*[col(c) for c in X_spark.columns])
)

model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=model_uri,
    env_manager="virtualenv"
)

X_spark_with_predictions = X_spark_with_array.withColumn(
    "prediction",
    model_udf("features_array")
)

display(X_spark_with_predictions.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏁 Conclusion
# MAGIC
# MAGIC In this tutorial, we've demonstrated a complete workflow for building, training, tracking, and deploying PyTorch deep learning models with MLflow on Databricks. We've covered:
# MAGIC
# MAGIC - Creating synthetic regression data with complex patterns
# MAGIC - Designing flexible neural network architectures using PyTorch
# MAGIC - Tracking experiments with MLflow, capturing parameters, metrics, and artifacts
# MAGIC - Optimizing hyperparameters with Optuna and nested MLflow runs
# MAGIC - Registering models in Unity Catalog with proper signatures and aliases
# MAGIC - Validating and deploying models for both local and distributed inference
# MAGIC
# MAGIC This comprehensive approach ensures that every step of the deep learning lifecycle is well-documented, reproducible, and production-ready. By leveraging PyTorch's flexibility and MLflow's tracking capabilities, you can build and deploy sophisticated models with confidence.

# COMMAND ----------

