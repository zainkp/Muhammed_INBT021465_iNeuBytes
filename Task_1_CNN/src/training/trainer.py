"""
Baseline Training Pipeline for CIFAR-10 Convolutional Neural Network.

This module implements the training engine for the baseline CNN model on CIFAR-10.
It manages model compilation, callback orchestration, training execution over a
fixed epoch budget, checkpoint persistence, and metric history logging.

Architectural & Experimental Guardrails:
- Model Compilation: Adam optimizer with configured learning rate (0.001) and
  Sparse Categorical Crossentropy loss for integer labels.
- Fixed Epoch Budget: Trains for exactly 30 epochs (DEFAULT_EPOCHS) without EarlyStopping.
  The training interface strictly enforces this budget and rejects attempts to override it.
- Pure Training Focus: Accepts only training and validation data streams.
- No Regularization: No Dropout, Batch Normalization, weight decay, or data augmentation.
- Checkpointing: Saves the best model checkpoint (.keras) based on validation accuracy
  to `Task_1_CNN/models/checkpoints/`.
- Logging: Persists training and validation metric history to `Task_1_CNN/results/logs/`
  in both CSV and JSON formats for downstream evaluation and plotting.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# ------------------------------------------------------------------------------
# Config & Path Handling (Support both module and direct execution)
# ------------------------------------------------------------------------------
try:
    from src.config import (
        CHECKPOINTS_DIR,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        LOGS_DIR,
        RANDOM_SEED,
    )
except (ImportError, ModuleNotFoundError):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import (
        CHECKPOINTS_DIR,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        LOGS_DIR,
        RANDOM_SEED,
    )

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


# ------------------------------------------------------------------------------
# Baseline Trainer Class
# ------------------------------------------------------------------------------
class BaselineTrainer:
    """
    Trainer for the Baseline CIFAR-10 Convolutional Neural Network.

    Handles model compilation, callback configuration, training execution across
    a fixed epoch budget (30 epochs), best-model checkpointing, and metric logging.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        checkpoints_dir: Union[str, Path] = CHECKPOINTS_DIR,
        checkpoint_filename: str = "baseline_cifar10_best.keras",
        logs_dir: Union[str, Path] = LOGS_DIR,
        history_filename: str = "baseline_training_history.json",
        csv_log_filename: str = "baseline_training_log.csv",
        monitor_metric: str = "val_accuracy",
        monitor_mode: str = "max",
        experiment_name: str = "baseline_cnn",
    ) -> None:
        """
        Initialize the Baseline Trainer.

        Args:
            model (tf.keras.Model): An already-built (uncompiled or compiled) Keras model.
            learning_rate (float): Initial learning rate for Adam optimizer. Defaults to DEFAULT_LEARNING_RATE (0.001).
            epochs (int): Fixed number of training epochs. Must equal DEFAULT_EPOCHS (30).
            batch_size (int): Mini-batch size. Defaults to DEFAULT_BATCH_SIZE (64).
            checkpoints_dir (Union[str, Path]): Directory to store model checkpoints. Defaults to CHECKPOINTS_DIR.
            checkpoint_filename (str): Name of the best model checkpoint file. Defaults to "baseline_cifar10_best.keras".
            logs_dir (Union[str, Path]): Directory to store training logs. Defaults to LOGS_DIR.
            history_filename (str): Filename for saving JSON history. Defaults to "baseline_training_history.json".
            csv_log_filename (str): Filename for saving CSV training logs. Defaults to "baseline_training_log.csv".
            monitor_metric (str): Metric to monitor for best checkpoint saving. Defaults to "val_accuracy".
            monitor_mode (str): Optimization mode ('max' or 'min') for monitor_metric. Defaults to "max".
            experiment_name (str): Experiment identifier for metadata logging. Defaults to "baseline_cnn".

        Raises:
            ValueError: If epochs is explicitly provided with a value other than DEFAULT_EPOCHS (30).
        """
        if epochs != DEFAULT_EPOCHS:
            raise ValueError(
                f"Baseline training protocol enforces a fixed epoch budget of {DEFAULT_EPOCHS} epochs. "
                f"Received: epochs={epochs}. Overriding the fixed epoch budget is not permitted."
            )

        self.model = model
        self.learning_rate = float(learning_rate)
        self.epochs = DEFAULT_EPOCHS
        self.batch_size = int(batch_size)
        self.experiment_name = experiment_name

        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_filepath = self.checkpoints_dir / self.checkpoint_filename

        self.logs_dir = Path(logs_dir)
        self.history_filename = history_filename
        self.history_filepath = self.logs_dir / self.history_filename
        self.csv_log_filename = csv_log_filename
        self.csv_log_filepath = self.logs_dir / self.csv_log_filename

        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        # Ensure destination directories exist
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self.history: Optional[tf.keras.callbacks.History] = None
        self.is_compiled: bool = False

    def compile_model(
        self,
        learning_rate: Optional[float] = None,
    ) -> tf.keras.Model:
        """
        Compile the Keras model with Adam optimizer, Sparse Categorical Crossentropy, and accuracy.

        Loss Function Rationale:
        - `SparseCategoricalCrossentropy(from_logits=False)` is used because the CIFAR-10 dataset
          labels are integer encoded ([0, 9]) and the baseline CNN architecture includes a
          Softmax activation in its final Dense output layer.

        Metric Rationale:
        - Accuracy is tracked as the primary metric for multiclass classification performance.

        Args:
            learning_rate (Optional[float]): Custom learning rate. If None, uses self.learning_rate.

        Returns:
            tf.keras.Model: Compiled Keras model instance.
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            name="adam_optimizer",
        )

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            name="sparse_categorical_crossentropy",
        )

        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]

        logger.info(
            f"Compiling model '{self.model.name}' with:\n"
            f"  - Optimizer : Adam (learning_rate={lr})\n"
            f"  - Loss      : SparseCategoricalCrossentropy(from_logits=False)\n"
            f"  - Metrics   : ['accuracy']"
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
        )
        self.is_compiled = True
        return self.model

    def _setup_callbacks(self, verbose: int = 1) -> List[tf.keras.callbacks.Callback]:
        """
        Configure callbacks for the baseline training run.

        Enforced Baseline Callbacks:
        1. ModelCheckpoint: Saves the model with the highest validation accuracy (.keras format).
        2. CSVLogger: Streams epoch metrics to a CSV file for reproducibility and inspection.

        Strict Constraints:
        - NO EarlyStopping callback (training must run for the full fixed 30-epoch budget).
        - NO learning rate schedulers / ReduceLROnPlateau in baseline.

        Args:
            verbose (int): Verbosity mode for callbacks. Defaults to 1.

        Returns:
            List[tf.keras.callbacks.Callback]: List of configured Keras callback instances.
        """
        callbacks: List[tf.keras.callbacks.Callback] = []

        # 1. ModelCheckpoint for preserving the best validation model
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.checkpoint_filepath),
            monitor=self.monitor_metric,
            mode=self.monitor_mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose,
        )
        callbacks.append(checkpoint_cb)

        # 2. CSVLogger for streaming metrics to disk
        csv_logger_cb = tf.keras.callbacks.CSVLogger(
            filename=str(self.csv_log_filepath),
            separator=",",
            append=False,
        )
        callbacks.append(csv_logger_cb)

        logger.info(
            f"Configured callbacks:\n"
            f"  - ModelCheckpoint: filepath='{self.checkpoint_filepath}', monitor='{self.monitor_metric}', mode='{self.monitor_mode}', save_best_only=True\n"
            f"  - CSVLogger      : filepath='{self.csv_log_filepath}'\n"
            f"  - EarlyStopping  : Disabled (Fixed epoch budget of {self.epochs} epochs)"
        )

        return callbacks

    def save_history_json(
        self,
        history: tf.keras.callbacks.History,
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Serialize and save the training history dictionary to a JSON file.

        Args:
            history (tf.keras.callbacks.History): Keras training history object.
            filepath (Optional[Union[str, Path]]): Destination path. If None, uses self.history_filepath.

        Returns:
            Path: Path to the saved JSON history file.
        """
        save_path = Path(filepath) if filepath is not None else self.history_filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Format history metrics to native Python floats for JSON serialization
        history_dict: Dict[str, Any] = {
            "experiment": self.experiment_name,
            "epochs_configured": self.epochs,
            "epochs_completed": len(history.epoch) if hasattr(history, "epoch") else len(next(iter(history.history.values()))),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "monitor_metric": self.monitor_metric,
            "history": {},
        }

        for metric_name, values in history.history.items():
            history_dict["history"][metric_name] = [float(v) for v in values]

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Saved complete training history JSON to: {save_path}")
        return save_path

    def train(
        self,
        train_data: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
        val_data: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
        batch_size: Optional[int] = None,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Execute the baseline training run for the fixed 30-epoch budget.

        Safety & Integrity Guarantees:
        - Complies strictly with the fixed epoch budget (DEFAULT_EPOCHS = 30).
        - Records and returns complete Keras History.
        - Persists best model checkpoint (.keras) to models/checkpoints/.
        - Saves CSV and JSON logs to results/logs/.

        Args:
            train_data (Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]):
                Training tf.data.Dataset or (x_train, y_train) tuple.
            val_data (Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]):
                Validation tf.data.Dataset or (x_val, y_val) tuple.
            batch_size (Optional[int]): Batch size if numpy arrays are passed. Defaults to self.batch_size (64).
            verbose (int): Training verbosity (0 = silent, 1 = progress bar, 2 = one line per epoch). Defaults to 1.

        Returns:
            tf.keras.callbacks.History: The complete Keras History object.
        """
        num_epochs = self.epochs
        bs = batch_size if batch_size is not None else self.batch_size

        # Compile model if not already compiled
        if not self.is_compiled:
            self.compile_model()

        # Setup callbacks
        callbacks = self._setup_callbacks(verbose=verbose)

        logger.info("=" * 70)
        logger.info(f"Starting Baseline CNN Training for Fixed Epoch Budget: {num_epochs} Epochs")
        logger.info(f"  - Model Name         : {self.model.name}")
        logger.info(f"  - Total Parameters   : {self.model.count_params():,}")
        logger.info(f"  - Initial LR         : {self.learning_rate}")
        logger.info(f"  - Batch Size         : {bs}")
        logger.info(f"  - Checkpoint Target  : {self.checkpoint_filepath}")
        logger.info(f"  - History JSON Target: {self.history_filepath}")
        logger.info(f"  - History CSV Target : {self.csv_log_filepath}")
        logger.info("=" * 70)

        # Handle tf.data.Dataset vs Tuple[np.ndarray, np.ndarray] inputs
        if isinstance(train_data, tuple):
            x_train, y_train = train_data
            x_val, y_val = val_data if isinstance(val_data, tuple) else (None, None)
            validation_data = (x_val, y_val) if x_val is not None else val_data

            history = self.model.fit(
                x=x_train,
                y=y_train,
                batch_size=bs,
                epochs=num_epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose,
            )
        else:
            history = self.model.fit(
                train_data,
                epochs=num_epochs,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=verbose,
            )

        self.history = history

        # Save JSON history artifact
        self.save_history_json(history)

        logger.info("=" * 70)
        logger.info("Baseline CNN Training Complete.")
        logger.info(f"Best model checkpoint saved to: {self.checkpoint_filepath}")
        logger.info(f"Training logs saved to: {self.csv_log_filepath} and {self.history_filepath}")
        logger.info("=" * 70)

        return history


# ------------------------------------------------------------------------------
# High-Level Functional Interface
# ------------------------------------------------------------------------------
def train_baseline_model(
    model: tf.keras.Model,
    train_dataset: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
    val_dataset: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoints_dir: Union[str, Path] = CHECKPOINTS_DIR,
    checkpoint_filename: str = "baseline_cifar10_best.keras",
    logs_dir: Union[str, Path] = LOGS_DIR,
    history_filename: str = "baseline_training_history.json",
    csv_log_filename: str = "baseline_training_log.csv",
    monitor_metric: str = "val_accuracy",
    monitor_mode: str = "max",
    verbose: int = 1,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Functional wrapper to compile and train the baseline CIFAR-10 CNN model.

    Args:
        model (tf.keras.Model): Uncompiled or pre-built Keras model.
        train_dataset (Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]): Training data.
        val_dataset (Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]): Validation data.
        learning_rate (float): Adam optimizer learning rate. Defaults to DEFAULT_LEARNING_RATE (0.001).
        epochs (int): Total epochs. Must equal DEFAULT_EPOCHS (30).
        batch_size (int): Batch size. Defaults to DEFAULT_BATCH_SIZE (64).
        checkpoints_dir (Union[str, Path]): Directory for checkpoint storage.
        checkpoint_filename (str): Name of best model checkpoint file.
        logs_dir (Union[str, Path]): Directory for log files.
        history_filename (str): Filename for JSON history.
        csv_log_filename (str): Filename for CSV history.
        monitor_metric (str): Checkpoint monitor metric (default 'val_accuracy').
        monitor_mode (str): Checkpoint monitor mode (default 'max').
        verbose (int): Verbosity mode.

    Raises:
        ValueError: If epochs does not equal DEFAULT_EPOCHS (30).

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: Trained model and Keras History object.
    """
    if epochs != DEFAULT_EPOCHS:
        raise ValueError(
            f"Baseline training protocol enforces a fixed epoch budget of {DEFAULT_EPOCHS} epochs. "
            f"Received: epochs={epochs}. Overriding the fixed epoch budget is not permitted."
        )

    trainer = BaselineTrainer(
        model=model,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        checkpoints_dir=checkpoints_dir,
        checkpoint_filename=checkpoint_filename,
        logs_dir=logs_dir,
        history_filename=history_filename,
        csv_log_filename=csv_log_filename,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
    )

    trainer.compile_model()
    history = trainer.train(
        train_data=train_dataset,
        val_data=val_dataset,
        batch_size=batch_size,
        verbose=verbose,
    )

    return model, history


# ------------------------------------------------------------------------------
# Module Verification & Inspection (No 30-Epoch Training Execution)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from src.models.cnn_architecture import build_baseline_cnn

    print("=" * 75)
    print("  Baseline CNN Trainer Module Verification")
    print("=" * 75)

    # 1. Instantiate baseline architecture
    sample_model = build_baseline_cnn()
    print(f"  - Model initialized: {sample_model.name}")
    print(f"  - Model initial compiled state: {sample_model.compiled}")

    # 2. Instantiate BaselineTrainer
    trainer = BaselineTrainer(
        model=sample_model,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    # 3. Test compilation
    compiled_model = trainer.compile_model()
    print(f"  - Model compiled state after trainer.compile_model(): {compiled_model.compiled}")
    print(f"  - Optimizer: {compiled_model.optimizer.name} (LR: {compiled_model.optimizer.learning_rate.numpy():.4f})")
    print(f"  - Loss function: {compiled_model.loss.name}")
    print(f"  - Metrics: {[m.name for m in compiled_model.metrics]}")

    # 4. Verify callback configuration
    callbacks = trainer._setup_callbacks(verbose=0)
    callback_names = [type(cb).__name__ for cb in callbacks]
    print(f"  - Configured callbacks: {callback_names}")
    print(f"  - Checkpoint filepath: {trainer.checkpoint_filepath}")
    print(f"  - CSV Log filepath: {trainer.csv_log_filepath}")
    print(f"  - JSON History filepath: {trainer.history_filepath}")
    print(f"  - EarlyStopping present: {'EarlyStopping' in callback_names} (Correct: False)")
    print(f"  - Epoch budget: {trainer.epochs} epochs (Fixed budget enforced)")

    # 5. Verify epoch budget enforcement guardrails
    try:
        BaselineTrainer(model=sample_model, epochs=10)
        print("  - BaselineTrainer(epochs=10) check: FAILED (did not raise ValueError)")
    except ValueError:
        print("  - BaselineTrainer(epochs=10) check: PASSED (Raised ValueError as expected)")

    try:
        train_baseline_model(model=sample_model, train_dataset=(None, None), val_dataset=(None, None), epochs=15)
        print("  - train_baseline_model(epochs=15) check: FAILED (did not raise ValueError)")
    except ValueError:
        print("  - train_baseline_model(epochs=15) check: PASSED (Raised ValueError as expected)")

    print("=" * 75)
    print("  Trainer verification completed successfully. (Full training run deferred).")
    print("=" * 75)
