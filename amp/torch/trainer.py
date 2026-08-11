import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks import Callback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Trainer(L.Trainer):
    
    def __init__(
            self,
            train_dataset=None,
            val_dataset=None,
            test_dataset=None,
            dataset=None,  # For backward compatibility
            train_indices=None,
            val_indices=None,
            test_indices=None,
            enable_plot=False,
            warmup_epochs=0,
            curriculum_epochs=None,  # For backward compatibility with linear strategy
            curriculum_strategy='linear',  # 'linear' or 'window_stages'
            curriculum_config=None,  # Dict for window_stages configuration
            patience=100,
            *args, 
            **kwargs
            ):
        
        # Handle backward compatibility: if 'dataset' is provided, use it for all three
        if dataset is not None:
            if train_dataset is None:
                train_dataset = dataset
            if val_dataset is None:
                val_dataset = dataset
            if test_dataset is None:
                test_dataset = dataset
        
        # Store datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # If indices are not provided, create default indices for each dataset
        if train_indices is None and train_dataset is not None:
            train_indices = list(range(len(train_dataset)))
        if val_indices is None and val_dataset is not None:
            val_indices = list(range(len(val_dataset)))
        if test_indices is None and test_dataset is not None:
            test_indices = list(range(len(test_dataset)))
        
        # Store the indices for use by callbacks
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # # Create datasets with chronological split
        # train_dataset = Subset(dataset, train_indices)
        # val_dataset = Subset(dataset, val_indices)

        # # Formulate dataloaders
        # self.train_loaders = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        # self.val_loaders = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        
        # Create callbacks with separate datasets for denormalization
        # Note: PredictionPlotCallback needs indices AFTER they are created above
        dataset_viz_cb = DatasetVisualizerCallback(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_indices=self.train_indices,
            val_indices=self.val_indices,
            test_indices=self.test_indices
        )
        prediction_cb = PredictionPlotCallback(
            num_samples=15, 
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_indices=self.train_indices,
            val_indices=self.val_indices,
            test_indices=self.test_indices
        )
        # Create curriculum callback based on strategy
        if curriculum_strategy == 'window_stages':
            if curriculum_config is None:
                raise ValueError("curriculum_strategy='window_stages' requires curriculum_config parameter")
            curriculum_cb = CurriculumLearningCallback(
                strategy='window_stages',
                curriculum_config=curriculum_config
            )
        else:
            # Default to linear strategy (backward compatible)
            curriculum_duration = curriculum_epochs if curriculum_epochs is not None else warmup_epochs
            curriculum_cb = CurriculumLearningCallback(
                strategy='linear',
                start_fraction=0.25,
                warmup_epochs=curriculum_duration
            )
        
        early_stop_cb = EarlyStoppingWithWarmup(warmup_epochs=warmup_epochs, monitor="val_loss", patience=patience, mode="min")
        checkpoint_cb = ModelCheckpoint(
            dirpath='checkpoints',
            filename='model-{epoch:02d}-{val_loss:.4f}',
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        timing_callback = TimingCallback()

        # Wrap callbacks with timing (except TimingCallback itself)
        callbacks = [
            timing_callback,  # Track performance metrics - NOT wrapped
            CallbackTimingWrapper(curriculum_cb, "CurriculumLearning", timing_callback),
            CallbackTimingWrapper(early_stop_cb, "EarlyStopping", timing_callback),
            CallbackTimingWrapper(checkpoint_cb, "ModelCheckpoint", timing_callback),
        ]
        
        # Add plot callbacks only if enabled (avoids macOS matplotlib segfault during training)
        if enable_plot:
            loss_plot_cb = LossPlotCallback()
            callbacks.insert(1, CallbackTimingWrapper(loss_plot_cb, "LossPlot", timing_callback))
            callbacks.insert(1, CallbackTimingWrapper(dataset_viz_cb, "DatasetVisualizer", timing_callback))
            callbacks.insert(1, CallbackTimingWrapper(prediction_cb, "PredictionPlot", timing_callback))
        
        # Merge with any callbacks passed via kwargs
        if 'callbacks' in kwargs:
            additional_callbacks = kwargs.pop('callbacks')
            if additional_callbacks:
                callbacks.extend(additional_callbacks)
        
        super().__init__(callbacks=callbacks, *args, **kwargs)
    
    def fit(self, *args, **kwargs):
        """Override fit to log stop reason after training."""
        # Call parent fit method
        result = super().fit(*args, **kwargs)
        
        # Log why training stopped
        print(f"\nTraining ended at epoch {self.current_epoch}/{self.max_epochs}")
        if self.interrupted:
            print("Reason: Interrupted (Ctrl+C or external signal)")
        elif self.should_stop:
            print("Reason: Early stopping or callback requested stop")
        elif self.current_epoch >= self.max_epochs - 1:
            print("Reason: Reached max_epochs")
        
        return result
        

class LossPlotCallback(Callback):
    """Callback to plot training and validation loss convergence using matplotlib."""
    
    def __init__(self, plot_frequency=10):
        """
        Args:
            plot_frequency: Update plot every N epochs to reduce overhead
        """
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.plot_frequency = plot_frequency
        
        # Setup matplotlib figure (will be created on first update)
        self.fig = None
        self.ax = None
        
    def _setup_plot(self):
        """Lazy initialization of plot."""
        if self.fig is None:
            plt.ion()  # Interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Loss')
            self.ax.set_title('Training Loss Convergence')
            self.ax.grid(True, alpha=0.3)
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the training epoch ends."""
        # Skip during sanity check
        if trainer.sanity_checking:
            return
            
        # Get current epoch
        current_epoch = trainer.current_epoch
        
        # Get training loss (epoch average)
        train_loss = trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is None:
            train_loss = trainer.callback_metrics.get('train_loss')
        
        if train_loss is not None:
            self.epochs.append(current_epoch)
            self.train_losses.append(train_loss.item())
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        # Skip during sanity check
        if trainer.sanity_checking:
            return
            
        # Get validation loss (epoch average)
        val_loss = trainer.callback_metrics.get('val_loss_epoch')
        if val_loss is None:
            val_loss = trainer.callback_metrics.get('val_loss')
        
        if val_loss is not None:
            # Only add val loss if we have a corresponding epoch
            if len(self.val_losses) < len(self.epochs):
                self.val_losses.append(val_loss.item())
            
        # Only update plot if we have data AND it's time to update
        if len(self.train_losses) == 0:
            return
        
        # Only update plot every N epochs to reduce overhead
        if trainer.current_epoch % self.plot_frequency != 0:
            return
            
        # Lazy init plot
        self._setup_plot()
        
        # Update plot
        self.ax.clear()
        self.ax.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        
        # Plot validation loss if we have data - use same number of epochs
        if len(self.val_losses) > 0:
            # Use only the epochs that have corresponding validation losses
            val_epochs = self.epochs[:len(self.val_losses)]
            self.ax.plot(val_epochs, self.val_losses, 'r--', label='Validation Loss', linewidth=2, marker='s')
        
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss Convergence')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Use log scale if losses span multiple orders of magnitude
        if len(self.train_losses) > 0 and max(self.train_losses) / min(self.train_losses) > 10:
            self.ax.set_yscale('log')
        
        plt.tight_layout()
        plt.pause(0.01)  # Small pause to update the plot
            
    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        if self.fig is None:
            return
            
        # Log final plot to TensorBoard
        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'add_figure'):
            trainer.logger.experiment.add_figure(
                'loss_convergence',
                self.fig,
                global_step=trainer.current_epoch
            )
        
        plt.ioff()  # Turn off interactive mode


class PredictionPlotCallback(Callback):
    """Callback to plot actual vs predicted values for random, best, and worst samples."""
    
    def __init__(self, num_samples=3, num_best=5, num_worst=5, 
                 train_dataset=None, val_dataset=None, test_dataset=None, 
                 dataset=None, train_indices=None, val_indices=None, test_indices=None):
        super().__init__()
        self.num_samples = num_samples
        self.num_best = num_best
        self.num_worst = num_worst
        
        # Handle backward compatibility: if 'dataset' is provided, use it for all three
        if dataset is not None:
            if train_dataset is None:
                train_dataset = dataset
            if val_dataset is None:
                val_dataset = dataset
            if test_dataset is None:
                test_dataset = dataset
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.train_indices = train_indices  # Original dataset indices for train split
        self.val_indices = val_indices      # Original dataset indices for val split  
        self.test_indices = test_indices    # Original dataset indices for test split
        self.train_outputs = None
        self.val_outputs = None
        self.test_outputs = None
        self.train_batch = None  # Store batch for uncertainty computation
        self.val_batch = None
        self.test_batch = None
        
        # Accumulate all test predictions for best/worst analysis
        self.test_predictions_all = []
        self.test_targets_all = []
        self.test_uncertainties_all = []
        
    def _plot_predictions(self, predictions, targets, stage, trainer, epoch=None, sample_indices=None, 
                         uncertainty=None, plot_type='random', sample_mses=None):
        """
        Plot actual vs predicted for selected samples.
        
        Args:
            predictions: Predicted values
            targets: Actual target values
            stage: Training stage (train/val/test)
            trainer: PyTorch Lightning trainer
            epoch: Current epoch number
            sample_indices: Specific sample indices to plot
            uncertainty: Optional uncertainty estimates (std) for predictions
            plot_type: Type of plot ('random', 'best', 'worst')
            sample_mses: Optional tensor of MSE values for each sample
        """
        batch_size = len(predictions)
        # Select random indices
        if sample_indices is None:
            indices = random.sample(range(batch_size), min(self.num_samples, batch_size))
        else:
            indices = sample_indices
        
        # Determine number of samples to plot
        num_samples_to_plot = len(indices)
        
        # Get the appropriate dataset and indices based on stage
        if stage == 'train':
            dataset = self.train_dataset
            dataset_indices = self.train_indices
        elif stage in ['validation', 'val']:
            dataset = self.val_dataset
            dataset_indices = self.val_indices
        elif stage == 'test':
            dataset = self.test_dataset
            dataset_indices = self.test_indices
        else:
            dataset = None
            dataset_indices = None
        
        # Determine number of targets
        if len(targets.shape) == 3:  # (batch, num_targets, time)
            num_targets = targets.shape[1]
        else:  # (batch, time)
            num_targets = 1
        
        # Create separate figure for each target
        for target_idx in range(num_targets):
            # Create figure with subplots for this target
            fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(14, 4 * num_samples_to_plot))
            if num_samples_to_plot == 1:
                axes = [axes]
            
            for idx, sample_idx in enumerate(indices):
                # Extract target
                if num_targets > 1:
                    actual = targets[sample_idx, target_idx, :].detach().cpu()
                    pred = predictions[sample_idx, target_idx, :].detach().cpu()
                else:
                    actual = targets[sample_idx].detach().cpu()
                    pred = predictions[sample_idx, 0, :].detach().cpu()
                
                # Convert to numpy first to get actual length
                actual = actual.numpy().flatten()
                pred = pred.numpy().flatten()
                
                # Use the actual length of predictions (may be shorter due to curriculum learning)
                pred_len = len(pred)
                
                # Denormalize if dataset is available
                if dataset is not None:
                    actual_tensor = torch.from_numpy(actual)
                    pred_tensor = torch.from_numpy(pred)
                    actual = dataset.denormalize_target(actual_tensor, target_idx=target_idx).numpy()
                    pred = dataset.denormalize_target(pred_tensor, target_idx=target_idx).numpy()
                    
                    # Get time index for this sample
                    # Map batch index to original dataset index
                    try:
                        if dataset_indices is not None:
                            original_idx = dataset_indices[sample_idx]
                        else:
                            original_idx = sample_idx
                        
                        # Slice time index to match actual prediction length
                        time_idx = dataset.get_time_index(original_idx)[dataset.hist_len:dataset.hist_len + pred_len]
                        x_values = pd.to_datetime(time_idx)
                        use_time = True
                    except Exception as e:
                        x_values = np.arange(pred_len)
                        use_time = False
                else:
                    x_values = np.arange(pred_len)
                    use_time = False
                
                # Plot actual and predicted
                axes[idx].plot(x_values, actual, 'b-', label='Actual', linewidth=2, marker='o', markersize=4)
                axes[idx].plot(x_values, pred, 'r--', label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.7)
                
                # Plot uncertainty bands if available
                if uncertainty is not None:
                    if num_targets > 1:
                        std = uncertainty[sample_idx, target_idx, :pred_len].detach().cpu()
                    else:
                        std = uncertainty[sample_idx, 0, :pred_len].detach().cpu()
                    
                    # Denormalize uncertainty if dataset available
                    # Note: uncertainty (std) should only be scaled, not shifted
                    if dataset is not None:
                        # dataset.std is indexed by target position (0 to n_targets-1)
                        target_std = dataset.std[target_idx]
                        std = std * target_std  # Only scale, don't add mean
                    
                    std = std.numpy().flatten()
                    
                    # Ensure std matches prediction length (in case of curriculum learning)
                    if len(std) != pred_len:
                        print(f"Warning: std length {len(std)} doesn't match prediction length {pred_len}, truncating")
                        std = std[:pred_len]
                    
                    # Plot 1-sigma and 2-sigma confidence intervals
                    axes[idx].fill_between(x_values, pred - std, pred + std, 
                                          color='red', alpha=0.2, label='±1σ (68%)')
                    axes[idx].fill_between(x_values, pred - 2*std, pred + 2*std, 
                                          color='red', alpha=0.1, label='±2σ (95%)')
                
                # Calculate metrics
                mse = ((actual - pred) ** 2).mean()
                mae = np.abs(actual - pred).mean()
                
                # Add feature name to title if available
                if dataset is not None and hasattr(dataset, 'target_feature_names'):
                    feature_name = dataset.target_feature_names[target_idx]
                else:
                    feature_name = f'Target {target_idx + 1}'
                
                # Build title with plot type and metrics
                title_parts = [f'{stage.capitalize()} ({plot_type.capitalize()})']
                title_parts.append(f'Sample {sample_idx + 1}')
                title_parts.append(feature_name)
                if sample_mses is not None:
                    overall_mse = sample_mses[idx].item()
                    title_parts.append(f'Overall MSE: {overall_mse:.4f}')
                title_parts.append(f'Target MSE: {mse:.4f}, MAE: {mae:.4f}')
                axes[idx].set_title(' | '.join(title_parts))
                
                if use_time:
                    axes[idx].set_xlabel('Time')
                    # Rotate x-axis labels for better readability
                    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
                else:
                    axes[idx].set_xlabel('Timestep')
                    
                axes[idx].set_ylabel('Value (Original Scale)' if dataset is not None else 'Value (Normalized)')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log to TensorBoard with target-specific name and plot type
            if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'add_figure'):
                if dataset is not None and hasattr(dataset, 'target_feature_names'):
                    target_name = dataset.target_feature_names[target_idx]
                else:
                    target_name = f'target_{target_idx}'
                
                tag = f'{stage}/predictions_{plot_type}_{target_name}'
                trainer.logger.experiment.add_figure(
                    tag,
                    fig,
                    global_step=trainer.current_epoch if epoch is None else epoch
                )
            
            plt.close(fig)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Store only the last training batch outputs to avoid memory overhead."""
        if trainer.sanity_checking:
            return
        # Only store outputs from the last batch to minimize memory usage
        # This is sufficient since we only plot samples from one batch
        if isinstance(outputs, dict) and 'predictions' in outputs and 'targets' in outputs:
            # Only keep outputs when we're at the last batch
            # Use num_training_batches which is available after first epoch
            total_batches = getattr(trainer, 'num_training_batches', None)
            if total_batches is None or batch_idx >= total_batches - 1:
                self.train_outputs = {
                    'predictions': outputs['predictions'].detach(),
                    'targets': outputs['targets'].detach(),
                    'uncertainties': outputs.get('uncertainties', None).detach() if outputs.get('uncertainties') is not None else None
                }
                self.train_batch = tuple(b.detach() if torch.is_tensor(b) else b for b in batch)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Store only the last validation batch outputs to avoid memory overhead."""
        if trainer.sanity_checking:
            return
        # Only store the last validation batch
        if isinstance(outputs, dict) and 'predictions' in outputs and 'targets' in outputs:
            # Store only last batch
            total_batches = getattr(trainer, 'num_val_batches', [None])[dataloader_idx] if hasattr(trainer, 'num_val_batches') else None
            if total_batches is None or batch_idx >= total_batches - 1:
                self.val_outputs = {
                    'predictions': outputs['predictions'].detach(),
                    'targets': outputs['targets'].detach(),
                    'uncertainties': outputs.get('uncertainties', None).detach() if outputs.get('uncertainties') is not None else None
                }
                self.val_batch = tuple(b.detach() if torch.is_tensor(b) else b for b in batch)
    
    def on_test_start(self, trainer, pl_module):
        """Clear accumulation lists at start of testing."""
        self.test_predictions_all = []
        self.test_targets_all = []
        self.test_uncertainties_all = []
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Accumulate all test batch outputs for best/worst analysis."""
        if isinstance(outputs, dict) and 'predictions' in outputs and 'targets' in outputs:
            # Accumulate predictions/targets from all batches
            self.test_predictions_all.append(outputs['predictions'].detach().cpu())
            self.test_targets_all.append(outputs['targets'].detach().cpu())
            if 'uncertainties' in outputs and outputs['uncertainties'] is not None:
                self.test_uncertainties_all.append(outputs['uncertainties'].detach().cpu())
            
            # Also store last batch for backward compatibility
            self.test_outputs = {
                'predictions': outputs['predictions'].detach(),
                'targets': outputs['targets'].detach(),
                'uncertainties': outputs.get('uncertainties', None).detach() if outputs.get('uncertainties') is not None else None
            }
            self.test_batch = tuple(b.detach() if torch.is_tensor(b) else b for b in batch)
    
    def _compute_uncertainty(self, pl_module, batch, num_samples=50):
        """
        Compute prediction uncertainty if model supports it (VAE models).
        
        Args:
            pl_module: The model
            batch: Input batch (current_features, lagged_features, targets)
            num_samples: Number of samples for uncertainty estimation
        
        Returns:
            uncertainty: Standard deviation of predictions, or None if not supported
        """
        if hasattr(pl_module, 'predict_with_uncertainty'):
            current_features, lagged_features, _ = batch
            pl_module.eval()
            with torch.no_grad():
                try:
                    _, std_pred = pl_module.predict_with_uncertainty(
                        current_features, lagged_features, num_samples=num_samples
                    )
                    return std_pred
                except Exception as e:
                    print(f"Could not compute uncertainty: {e}")
                    return None
        return None
    
    def on_train_end(self, trainer, pl_module):
        """Plot training and validation predictions at end of training."""
        print("[PredictionPlotCallback] on_train_end called - generating plots...")
        
        # Plot training predictions
        if self.train_outputs is not None:
            print(f"[PredictionPlotCallback] Generating train plots...")
            # Check if uncertainty is already in outputs (e.g., from LKF model)
            uncertainty = self.train_outputs.get('uncertainties', None)
            # Otherwise compute it for VAE models
            if uncertainty is None:
                uncertainty = self._compute_uncertainty(pl_module, self.train_batch) if self.train_batch else None
            self._plot_predictions(
                self.train_outputs['predictions'], 
                self.train_outputs['targets'], 
                'train',
                trainer,
                uncertainty=uncertainty
            )
            print(f"[PredictionPlotCallback] Train plots generated")
        else:
            print(f"[PredictionPlotCallback] No train outputs available")
        
        # Plot validation predictions
        if self.val_outputs is not None:
            print(f"[PredictionPlotCallback] Generating validation plots...")
            # Check if uncertainty is already in outputs (e.g., from LKF model)
            uncertainty = self.val_outputs.get('uncertainties', None)
            # Otherwise compute it for VAE models
            if uncertainty is None:
                uncertainty = self._compute_uncertainty(pl_module, self.val_batch) if self.val_batch else None
            self._plot_predictions(
                self.val_outputs['predictions'], 
                self.val_outputs['targets'], 
                'validation',
                trainer,
                uncertainty=uncertainty
            )
            print(f"[PredictionPlotCallback] Validation plots generated")
        else:
            print(f"[PredictionPlotCallback] No val outputs available")
    
    def on_validation_end(self, trainer, pl_module):
        """Store validation outputs but don't plot during training."""
        # Plots are now generated only at the end of training (on_train_end)
        pass
    
    def on_test_end(self, trainer, pl_module):
        """Plot test predictions at end of testing, including best/worst samples."""
        print("[PredictionPlotCallback] on_test_end called - generating test plots...")
        
        # Plot random samples from last batch (backward compatibility)
        if self.test_outputs is not None:
            print(f"[PredictionPlotCallback] Generating random sample plots...")
            uncertainty = self.test_outputs.get('uncertainties', None)
            if uncertainty is None:
                uncertainty = self._compute_uncertainty(pl_module, self.test_batch) if self.test_batch else None
            self._plot_predictions(
                self.test_outputs['predictions'], 
                self.test_outputs['targets'], 
                'test',
                trainer,
                epoch=trainer.current_epoch,
                uncertainty=uncertainty,
                plot_type='random'
            )
            print(f"[PredictionPlotCallback] Random sample plots generated")
        
        # Plot best/worst samples if we have accumulated all test predictions
        if len(self.test_predictions_all) > 0:
            print(f"[PredictionPlotCallback] Analyzing {len(self.test_predictions_all)} test batches for best/worst samples...")
            
            # Concatenate all batches
            all_predictions = torch.cat(self.test_predictions_all, dim=0)
            all_targets = torch.cat(self.test_targets_all, dim=0)
            all_uncertainties = None
            if len(self.test_uncertainties_all) > 0:
                all_uncertainties = torch.cat(self.test_uncertainties_all, dim=0)
            
            print(f"[PredictionPlotCallback] Total test samples: {all_predictions.shape[0]}")
            
            # Compute per-sample MSE (average across time and targets)
            sample_mses = ((all_predictions - all_targets) ** 2).mean(dim=(1, 2))
            
            # Sort by MSE
            sorted_indices = torch.argsort(sample_mses)
            
            # Get best (lowest MSE) and worst (highest MSE) indices
            best_indices = sorted_indices[:self.num_best]
            worst_indices = sorted_indices[-self.num_worst:]
            
            print(f"[PredictionPlotCallback] Best sample MSEs: {sample_mses[best_indices].tolist()}")
            print(f"[PredictionPlotCallback] Worst sample MSEs: {sample_mses[worst_indices].tolist()}")
            
            # Plot best samples
            if self.num_best > 0:
                best_predictions = all_predictions[best_indices]
                best_targets = all_targets[best_indices]
                best_uncertainties = all_uncertainties[best_indices] if all_uncertainties is not None else None
                
                self._plot_predictions(
                    best_predictions,
                    best_targets,
                    'test',
                    trainer,
                    epoch=trainer.current_epoch,
                    uncertainty=best_uncertainties,
                    plot_type='best',
                    sample_mses=sample_mses[best_indices]
                )
                print(f"[PredictionPlotCallback] Best sample plots generated")
            
            # Plot worst samples
            if self.num_worst > 0:
                worst_predictions = all_predictions[worst_indices]
                worst_targets = all_targets[worst_indices]
                worst_uncertainties = all_uncertainties[worst_indices] if all_uncertainties is not None else None
                
                self._plot_predictions(
                    worst_predictions,
                    worst_targets,
                    'test',
                    trainer,
                    epoch=trainer.current_epoch,
                    uncertainty=worst_uncertainties,
                    plot_type='worst',
                    sample_mses=sample_mses[worst_indices]
                )
                print(f"[PredictionPlotCallback] Worst sample plots generated")
        else:
            print(f"[PredictionPlotCallback] No accumulated test predictions for best/worst analysis")


class CurriculumLearningCallback(Callback):
    """
    Callback for curriculum learning - progressively increase task difficulty.
    
    Supports two strategies:
    1. 'linear': Smooth ramp from start_fraction to 1.0 over warmup_epochs
    2. 'window_stages': Discrete stages matching horizon loss windows
    """
    
    def __init__(
        self, 
        strategy='linear',
        start_fraction=0.25,
        warmup_epochs=20,
        curriculum_config=None,
    ):
        """
        Args:
            strategy: 'linear' or 'window_stages'
            start_fraction: Starting difficulty for linear strategy (0.25 = 25% of horizon)
            warmup_epochs: Number of epochs for linear strategy
            curriculum_config: Dict with configuration for window_stages:
                {
                    'windows': [(0, 4), (4, None)],  # Last window can use None (will use model's forecast_len)
                    'window_weights': [10.0, 5.0],  # Loss weights per window (optional, default 1.0)
                    'epochs_per_window': [500, None],  # Epochs per stage (None = remaining)
                }
                Note: 
                - Windows must be contiguous with no gaps: end of window i = start of window i+1
                - Last window can have None as end - will use model.forecast_len
                - If last window has explicit end, forecast_len is derived from it
                - First window must start at 0
        """
        super().__init__()
        self.strategy = strategy
        self.start_fraction = start_fraction
        self.warmup_epochs = warmup_epochs
        self.curriculum_config = curriculum_config or {}
        
        # Parse window_stages configuration
        if strategy == 'window_stages':
            if 'windows' not in self.curriculum_config:
                raise ValueError("window_stages strategy requires 'windows' in curriculum_config")
            if 'epochs_per_window' not in self.curriculum_config:
                raise ValueError("window_stages strategy requires 'epochs_per_window' in curriculum_config")
            
            self.windows = self.curriculum_config['windows']
            self.window_weights = self.curriculum_config.get('window_weights', [1.0] * len(self.windows))
            self.epochs_per_window = self.curriculum_config['epochs_per_window']
            
            # Validate window count consistency
            if len(self.epochs_per_window) != len(self.windows):
                raise ValueError(f"epochs_per_window length ({len(self.epochs_per_window)}) must match "
                               f"windows length ({len(self.windows)})")
            if len(self.window_weights) != len(self.windows):
                raise ValueError(f"window_weights length ({len(self.window_weights)}) must match "
                               f"windows length ({len(self.windows)})")
            
            # Validate windows are contiguous with no gaps (except last window can have None)
            for i in range(len(self.windows) - 1):
                current_end = self.windows[i][1]
                next_start = self.windows[i + 1][0]
                
                # Current window cannot have None end (only last window can)
                if current_end is None:
                    raise ValueError(f"Only the last window can have None as end. "
                                   f"Window {i} has None: {self.windows[i]}")
                
                if current_end != next_start:
                    raise ValueError(f"Window gap detected: window {i} ends at {current_end} "
                                   f"but window {i+1} starts at {next_start}. "
                                   f"Windows must be contiguous: [(0,4), (4,8), (8,12)]")
            
            # Validate first window starts at 0
            if self.windows[0][0] != 0:
                raise ValueError(f"First window must start at 0, got {self.windows[0][0]}")
            
            # Validate last window end (can be None - will be resolved in setup())
            if self.windows[-1][1] is None:
                print("Last window end is None - will use model's forecast_len")
            else:
                # Derive forecast_len from last window's end
                self.forecast_len = self.windows[-1][1]
                print(f"Derived forecast_len={self.forecast_len} from windows")
            
            # Build stage definitions (will be rebuilt in setup() if last window has None)
            self.stages = []
            for i in range(len(self.windows)):
                stage_windows = self.windows[:i+1]
                stage_weights = self.window_weights[:i+1]
                stage_end = self.windows[i][1]
                if stage_end is None:
                    # Will be resolved in setup()
                    stage_steps = None
                else:
                    stage_steps = stage_end
                stage_epochs = self.epochs_per_window[i]
                
                self.stages.append({
                    'windows': stage_windows,
                    'weights': stage_weights,
                    'steps': stage_steps,
                    'epochs': stage_epochs,
                })
        
        self.current_fraction = start_fraction
        self.current_steps = None
        self.forecast_len = None  # Will be set in setup() from model
    
    def setup(self, trainer, pl_module, stage):
        """Called at the beginning of fit - get forecast_len from model and inject horizon config."""
        if stage == 'fit' and self.strategy == 'window_stages':
            # Get forecast_len from model
            if not hasattr(pl_module, 'forecast_len'):
                raise ValueError("Model must have 'forecast_len' attribute for window_stages strategy")
            
            # Handle None in last window - replace with model's forecast_len
            if self.windows[-1][1] is None:
                self.windows = list(self.windows)  # Make mutable
                last_window = self.windows[-1]
                self.windows[-1] = (last_window[0], pl_module.forecast_len)
                print(f"Last window end was None - set to model's forecast_len={pl_module.forecast_len}")
            
            # Now derive forecast_len from windows
            self.forecast_len = self.windows[-1][1]
            
            # Rebuild stages with resolved windows
            self.stages = []
            for i in range(len(self.windows)):
                stage_windows = self.windows[:i+1]
                stage_weights = self.window_weights[:i+1]
                stage_steps = self.windows[i][1]
                stage_epochs = self.epochs_per_window[i]
                
                self.stages.append({
                    'windows': stage_windows,
                    'weights': stage_weights,
                    'steps': stage_steps,
                    'epochs': stage_epochs,
                })
            
            # Inject horizon loss windows into model
            print(f"\nCurriculum callback injecting horizon loss configuration into model:")
            print(f"  - Windows: {self.windows}")
            print(f"  - Weights: {self.window_weights}")
            print(f"  - Forecast length: {self.forecast_len}")
            print(f"  - Stages: {len(self.stages)} stages over {self.epochs_per_window}\n")
            
            # TODO: Why are these set here? If these attributes need to be set to the model instance, they should be done via some function instead of monkey patching.
            pl_module.horizon_loss_windows = self.windows
            pl_module.horizon_loss_weights = self.window_weights
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Update difficulty at the start of each epoch."""
        if trainer.sanity_checking:
            return
        
        forecast_len = pl_module.forecast_len
        
        current_epoch = trainer.current_epoch
        
        if self.strategy == 'linear':
            # Original linear curriculum
            progress = min(1.0, current_epoch / self.warmup_epochs)
            self.current_fraction = self.start_fraction + (1.0 - self.start_fraction) * progress
            pl_module.curriculum_fraction = self.current_fraction
            self.current_steps = None  # Not used in linear mode
            
            if current_epoch % 5 == 0:
                print(f"Epoch {current_epoch}: Curriculum difficulty = {self.current_fraction:.2%}")
        
        elif self.strategy == 'window_stages':
            # Window-based staged curriculum
            cumulative_epochs = 0
            current_stage_idx = len(self.stages) - 1  # Default to last stage
            
            for idx, stage in enumerate(self.stages):
                stage_epochs = stage['epochs'] if stage['epochs'] is not None else float('inf')
                if current_epoch < cumulative_epochs + stage_epochs:
                    current_stage_idx = idx
                    break
                cumulative_epochs += stage_epochs
            
            current_stage = self.stages[current_stage_idx]
            self.current_steps = current_stage['steps']
            
            # Also compute fraction for compatibility (forecast_len set in setup())
            self.current_fraction = self.current_steps / forecast_len
            
            # Set in model
            pl_module.curriculum_steps = self.current_steps
            pl_module.curriculum_fraction = self.current_fraction
            
            # Update model's active windows for this stage (dynamic window activation)
            pl_module.horizon_loss_windows = current_stage['windows']
            pl_module.horizon_loss_weights = current_stage['weights']
            
            # Print status on stage transitions or every 50 epochs
            if current_epoch == 0 or \
               (current_epoch == cumulative_epochs) or \
               (current_epoch % 50 == 0):
                print(f"Epoch {current_epoch}: Curriculum stage {current_stage_idx + 1}/{len(self.stages)} "
                      f"- predicting {self.current_steps} steps "
                      f"(windows: {current_stage['windows']})")


class EarlyStoppingWithWarmup(EarlyStopping):
    """EarlyStopping with warmup period - only starts monitoring after warmup_epochs."""
    
    def __init__(self, warmup_epochs=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self._warmup_message_shown = False
        
    def on_validation_end(self, trainer, pl_module):
        """
        Override to skip early stopping check during warmup period.
        
        PyTorch Lightning's EarlyStopping calls _run_early_stopping_check() from both
        on_validation_end() and on_train_epoch_end(). We must override both methods to
        prevent early stopping from triggering during warmup, which is critical for:
        1. Curriculum learning - loss changes as task difficulty increases
        2. Multi-phase training - different optimization strategies per phase
        3. Warmup periods - allowing model to stabilize before monitoring
        """
        if trainer.current_epoch < self.warmup_epochs:
            # During warmup, don't run early stopping check
            if not self._warmup_message_shown:
                print(f"Early stopping warmup: will start monitoring after epoch {self.warmup_epochs}")
                self._warmup_message_shown = True
            return
        
        # After warmup, use normal early stopping behavior
        # NOTE: super().on_validation_end() does not work here - must use explicit parent class call
        EarlyStopping.on_validation_end(self, trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Override to skip early stopping check during warmup period.
        
        This is the second entry point for early stopping checks. The parent class
        runs _run_early_stopping_check() here when check_on_train_epoch_end=True.
        Must be overridden alongside on_validation_end() to fully disable early stopping
        during warmup, otherwise the callback will still monitor and potentially stop
        training prematurely during curriculum transitions or warmup phases.
        """
        if trainer.current_epoch < self.warmup_epochs:
            # During warmup, don't run early stopping check
            return
        
        # After warmup, use normal early stopping behavior
        # NOTE: super().on_train_epoch_end() does not work here - must use explicit parent class call
        EarlyStopping.on_train_epoch_end(self, trainer, pl_module)

class DatasetVisualizerCallback(Callback):
    """Callback to visualize original data split by train/val/test."""
    
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, 
                 dataset=None, train_indices=None, val_indices=None, test_indices=None):
        super().__init__()
        
        # Handle backward compatibility: if 'dataset' is provided, use it for all three
        if dataset is not None:
            if train_dataset is None:
                train_dataset = dataset
            if val_dataset is None:
                val_dataset = dataset
            if test_dataset is None:
                test_dataset = dataset
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.visualized = False
    
    def on_train_start(self, trainer, pl_module):
        """Visualize original data split by train/val/test at the start of training."""
        if self.visualized:
            return
        
        self.visualized = True
        
        # Collect datasets that are available
        datasets = []
        if self.train_dataset is not None:
            datasets.append(('Train', self.train_dataset, self.train_indices))
        if self.val_dataset is not None:
            datasets.append(('Validation', self.val_dataset, self.val_indices))
        if self.test_dataset is not None:
            datasets.append(('Test', self.test_dataset, self.test_indices))
        
        if not datasets:
            print("No datasets provided to DatasetVisualizerCallback")
            return
        
        # Get feature information from first available dataset
        first_dataset = datasets[0][1]
        
        # Determine features to plot: all targets + all input features (excluding targets)
        num_targets = len(first_dataset.targets)
        # Filter out targets from features to avoid duplicates
        input_features_only = [f for f in first_dataset.features if f not in first_dataset.targets]
        num_input_features = len(input_features_only)
        num_features = num_targets + num_input_features
        num_splits = len(datasets)
        
        # Create subplots: rows = features, cols = splits (train, val, test)
        fig, axes = plt.subplots(num_features, num_splits, figsize=(8 * num_splits, 4 * num_features))
        
        # Handle case where we have only 1 feature or 1 split
        if num_features == 1 and num_splits == 1:
            axes = [[axes]]
        elif num_features == 1:
            axes = [axes]
        elif num_splits == 1:
            axes = [[ax] for ax in axes]
        
        # Plot each split
        for col_idx, (split_name, dataset, indices) in enumerate(datasets):
            
            # Get reconstructed DataFrame from samples
            original_data = dataset.get_timeseries_from_samples()  # Returns pd.DataFrame
            
            # Denormalize the data for plotting
            if hasattr(dataset, 'denormalize_dataframe'):
                original_data = dataset.denormalize_dataframe(original_data)
            
            if original_data.empty:
                print(f"No data available for {split_name}")
                continue
            
            time_index = original_data.index
            
            # Use all available data
            data_indices = list(range(len(time_index)))
            
            # Get time values for this split
            split_time = time_index[data_indices]
            x_values = pd.to_datetime(split_time)
            
            row_idx = 0
            
            # Plot all target features (first rows)
            for target_idx, target_name in enumerate(dataset.targets):
                ax = axes[row_idx][col_idx]
                
                if target_name in original_data.columns:
                    target_data = original_data[target_name].iloc[data_indices].values
                    ax.plot(x_values, target_data, 'b-', linewidth=1, alpha=0.8)
                    ax.set_title(f'{split_name}: {target_name} (target)')
                else:
                    ax.set_title(f'{split_name}: {target_name} (not found)')
                       
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                row_idx += 1
            
            # Plot all input features (remaining rows) - exclude targets to avoid duplicates
            input_features_only = [f for f in dataset.features if f not in dataset.targets]
            for feature_name in input_features_only:
                ax = axes[row_idx][col_idx]
                
                # Get feature data from DataFrame by column name
                if feature_name in original_data.columns:
                    feature_data = original_data[feature_name].iloc[data_indices].values
                    
                    ax.plot(x_values, feature_data, 'g-', linewidth=1, alpha=0.8)
                    ax.set_title(f'{split_name}: {feature_name} (input)')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title(f'{split_name}: {feature_name} (not found)')
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                row_idx += 1
        
        # Set x-labels only on bottom row
        for col_idx in range(num_splits):
            ax = axes[-1][col_idx]
            ax.set_xlabel('Time')
        
        plt.tight_layout()
        
        # Log to TensorBoard if available
        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'add_figure'):
            trainer.logger.experiment.add_figure('datasets/visualization', fig, 0)
            print("Dataset visualization logged to TensorBoard")
        
        # Save to file
        plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
        print("Dataset visualization saved to dataset_visualization.png")
        plt.close(fig)


def normalize(x, _mean, _std):
    """Normalize input with mean and std."""
    return (x - _mean) / _std


class TimingCallback(Callback):
    """Callback to track and log time spent in different training phases."""
    
    def __init__(self):
        super().__init__()
        self.timers = defaultdict(lambda: {'total': 0.0, 'count': 0, 'start': None})
        self.epoch_start = None
        self.train_epoch_start = None
        self.val_epoch_start = None
        
    def on_train_start(self, trainer, pl_module):
        """Reset timers at training start."""
        self.timers.clear()
        print("\n[Timing] Starting performance tracking...")
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Mark start of training epoch."""
        self.epoch_start = time.time()
        self.train_epoch_start = time.time()
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Mark start of training batch."""
        self.timers['train_batch']['start'] = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record training batch time."""
        if self.timers['train_batch']['start'] is not None:
            elapsed = time.time() - self.timers['train_batch']['start']
            self.timers['train_batch']['total'] += elapsed
            self.timers['train_batch']['count'] += 1
            self.timers['train_batch']['start'] = None
            
    def on_train_epoch_end(self, trainer, pl_module):
        """Record training epoch time."""
        if self.train_epoch_start is not None:
            elapsed = time.time() - self.train_epoch_start
            self.timers['train_epoch']['total'] += elapsed
            self.timers['train_epoch']['count'] += 1
            self.train_epoch_start = None
            
    def on_validation_epoch_start(self, trainer, pl_module):
        """Mark start of validation epoch."""
        self.val_epoch_start = time.time()
        
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Mark start of validation batch."""
        self.timers['val_batch']['start'] = time.time()
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Record validation batch time."""
        if self.timers['val_batch']['start'] is not None:
            elapsed = time.time() - self.timers['val_batch']['start']
            self.timers['val_batch']['total'] += elapsed
            self.timers['val_batch']['count'] += 1
            self.timers['val_batch']['start'] = None
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """Record validation epoch time."""
        if self.val_epoch_start is not None:
            elapsed = time.time() - self.val_epoch_start
            self.timers['val_epoch']['total'] += elapsed
            self.timers['val_epoch']['count'] += 1
            self.val_epoch_start = None
            
    def on_train_end(self, trainer, pl_module):
        """Log timing summary to TensorBoard and console."""
        if not self.timers:
            return
            
        print("\n" + "="*80)
        print("TRAINING TIME ANALYSIS")
        print("="*80)
        
        # Calculate total time
        total_time = sum(timer['total'] for timer in self.timers.values())
        
        # Log summary to console
        for phase, timer in sorted(self.timers.items(), key=lambda x: x[1]['total'], reverse=True):
            avg_time = timer['total'] / timer['count'] if timer['count'] > 0 else 0
            percentage = (timer['total'] / total_time * 100) if total_time > 0 else 0
            
            print(f"\n{phase.upper()}:")
            print(f"  Total time: {timer['total']:.2f}s ({percentage:.1f}%)")
            print(f"  Count: {timer['count']}")
            print(f"  Average: {avg_time:.4f}s")
        
        print(f"\nTOTAL TRAINING TIME: {total_time:.2f}s ({total_time/60:.2f}m)")
        print("="*80 + "\n")
        
        # Log to TensorBoard
        if hasattr(trainer.logger, 'experiment'):
            logger = trainer.logger.experiment
            
            # Log absolute times
            for phase, timer in self.timers.items():
                logger.add_scalar(f'timing/{phase}_total_seconds', timer['total'], 0)
                if timer['count'] > 0:
                    logger.add_scalar(f'timing/{phase}_avg_seconds', 
                                    timer['total'] / timer['count'], 0)
            
            # Log relative percentages
            for phase, timer in self.timers.items():
                percentage = (timer['total'] / total_time * 100) if total_time > 0 else 0
                logger.add_scalar(f'timing/{phase}_percentage', percentage, 0)
            
            logger.add_scalar('timing/total_seconds', total_time, 0)
            logger.add_scalar('timing/total_minutes', total_time / 60, 0)
            
            # Create a bar chart showing relative time distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Absolute times
            phases = list(self.timers.keys())
            times = [self.timers[p]['total'] for p in phases]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(phases)))
            
            ax1.barh(phases, times, color=colors)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_title('Absolute Time Spent in Each Phase')
            ax1.grid(axis='x', alpha=0.3)
            
            # Percentage distribution
            percentages = [(self.timers[p]['total'] / total_time * 100) if total_time > 0 else 0 
                          for p in phases]
            
            ax2.barh(phases, percentages, color=colors)
            ax2.set_xlabel('Percentage of Total Time (%)')
            ax2.set_title('Relative Time Distribution')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            logger.add_figure('timing/distribution', fig, 0)
            plt.close(fig)
            
            print("[Timing] Performance metrics logged to TensorBoard")


class CallbackTimingWrapper(Callback):
    """Wrapper to track time spent in individual callbacks."""
    
    def __init__(self, callback, callback_name, timing_callback):
        """
        Args:
            callback: The callback to wrap
            callback_name: Name for logging
            timing_callback: Reference to TimingCallback for storing results
        """
        super().__init__()
        self.callback = callback
        self.callback_name = callback_name
        self.timing_callback = timing_callback
        
    def _time_method(self, method_name, *args, **kwargs):
        """Time a callback method execution."""
        if hasattr(self.callback, method_name):
            method = getattr(self.callback, method_name)
            start_time = time.time()
            result = method(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Record in timing callback
            timer_key = f'callback_{self.callback_name}'
            self.timing_callback.timers[timer_key]['total'] += elapsed
            self.timing_callback.timers[timer_key]['count'] += 1
            
            return result
        return None
    
    # Wrap all callback lifecycle methods
    def setup(self, trainer, pl_module, stage):
        return self._time_method('setup', trainer, pl_module, stage)
    
    def on_train_start(self, trainer, pl_module):
        return self._time_method('on_train_start', trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        return self._time_method('on_train_end', trainer, pl_module)
    
    def on_train_epoch_start(self, trainer, pl_module):
        return self._time_method('on_train_epoch_start', trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        return self._time_method('on_train_epoch_end', trainer, pl_module)
    
    def on_validation_epoch_start(self, trainer, pl_module):
        return self._time_method('on_validation_epoch_start', trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        return self._time_method('on_validation_epoch_end', trainer, pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        return self._time_method('on_validation_end', trainer, pl_module)
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        return self._time_method('on_train_batch_start', trainer, pl_module, batch, batch_idx)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return self._time_method('on_train_batch_end', trainer, pl_module, outputs, batch, batch_idx)
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        return self._time_method('on_validation_batch_start', trainer, pl_module, batch, batch_idx, dataloader_idx)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        return self._time_method('on_validation_batch_end', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_test_start(self, trainer, pl_module):
        return self._time_method('on_test_start', trainer, pl_module)
    
    def on_test_end(self, trainer, pl_module):
        return self._time_method('on_test_end', trainer, pl_module)
    
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        return self._time_method('on_test_batch_start', trainer, pl_module, batch, batch_idx, dataloader_idx)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        return self._time_method('on_test_batch_end', trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
