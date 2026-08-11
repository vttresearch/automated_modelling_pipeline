""" Torch based dataloader class. 

This class handles data loading and processing for torch models.
The dataloader is initialized with model-specific parameters and creates PyTorch
DataLoader objects that can be used for training.

MODEL FITTING
Models implement get_fit_data(dataloader) which returns PyTorch DataLoader objects:
- model.get_fit_data(dataloader) -> returns list of [train_loader, val_loader, test_loader]

MODEL PREDICTION
AMP models implement a predict() method that takes a dataframe as an input.
The dataloader provides evaluation dataframes through:
- dataloader.get_evaluation_dataframes(period) -> returns list of DataFrames

The dataloader must be initialized with:
- fcast_len: Forecast length in steps
- lead_time: Lead time in steps
- data_freq: Data frequency in minutes
- update_rate: Update rate in steps
- feature_windows: Dictionary mapping feature names to window specifications
- target_columns: List of target column names

"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoaderClass
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
import logging

from amp.dataloader import DataLoader, output_method
from amp.utils import create_dataset, create_eval_idx

logger = logging.getLogger(__name__)


class SamplesDataset(Dataset):
    """
    Minimal PyTorch Dataset wrapper for pre-created samples.
    
    This dataset wraps samples created by DataLoader.create_samples() and splits
    them into features and targets for PyTorch training. All windowing, filtering,
    and validation is handled by create_samples().
    
    Parameters
    ----------
    samples : list of dict
        Samples from create_samples() with 'data' key containing DataFrames
    features : list of str
        Column names to use as features
    targets : list of str
        Column names to use as targets
    fcast_len : int
        Forecast length (how many target steps to extract)
    lead_time : int
        Lead time offset
    norm_params : dict, optional
        Normalization parameters {'mean': pd.Series, 'std': pd.Series}
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        features: List[str],
        targets: List[str],
        input_window: Tuple[int, int],
        norm_params: Optional[Dict[str, pd.Series]] = None
    ):
        self.samples = samples
        self.features = features
        self.targets = targets
        self.input_window = input_window
        self.fcast_len = input_window[1] + 1
        self.hist_len = abs(input_window[0])
        self.norm_params = norm_params
        
        # Lazy-loaded attributes (computed on first access)
        self._target_feature_names = None
        self._target_indices = None
        self._mean = None
        self._std = None
        
        # Pre-create tensors for all samples to avoid recreating on every __getitem__ call
        self._feature_tensors = None
        self._target_tensors = None
    
    def _initialize_tensors(self):
        """
        Pre-create all feature and target tensors from samples.
        
        This is done lazily on first access to avoid overhead if dataset is created
        but not used (e.g., in some callbacks).
        """
        if self._feature_tensors is not None:
            return  # Already initialized
        
        feature_tensors = []
        target_tensors = []
        
        for sample in self.samples:
            sample_df = sample['data']
            
            # Extract features (all feature columns, all timesteps)
            # Use numpy nan_to_num to fill NaN before tensor creation (avoids torch.nan_to_num
            # crash on Apple Silicon with PyTorch 2.11+)
            feature_data = np.nan_to_num(sample_df[self.features].values.astype(np.float32), nan=0.0)
            feature_tensors.append(torch.FloatTensor(feature_data))
            
            # Extract targets from forecast horizon
            # Targets are at indices [lead_time : lead_time + fcast_len]
            target_data = np.nan_to_num(sample_df[self.targets].iloc[self.hist_len:].values.astype(np.float32), nan=0.0)
            target_tensors.append(torch.FloatTensor(target_data))
        
        self._feature_tensors = feature_tensors
        self._target_tensors = target_tensors
    
    @property
    def target_feature_names(self):
        """Lazy-loaded target feature names."""
        if self._target_feature_names is None:
            self._target_feature_names = self.targets
        return self._target_feature_names
    
    @property
    def target_indices(self):
        """Lazy-loaded target indices."""
        if self._target_indices is None:
            self._target_indices = [self.features.index(t) if t in self.features else len(self.features) + self.targets.index(t) 
                                   for t in self.targets]
        return self._target_indices
    
    @property
    def mean(self):
        """Lazy-loaded mean tensor."""
        if self._mean is None:
            if self.norm_params and 'mean' in self.norm_params:
                self._mean = torch.FloatTensor([self.norm_params['mean'][col] if col in self.norm_params['mean'].index else 0.0 
                                               for col in self.targets])
            else:
                self._mean = torch.zeros(len(self.targets))
        return self._mean
    
    @property
    def std(self):
        """Lazy-loaded std tensor."""
        if self._std is None:
            if self.norm_params and 'std' in self.norm_params:
                self._std = torch.FloatTensor([self.norm_params['std'][col] if col in self.norm_params['std'].index else 1.0 
                                              for col in self.targets])
            else:
                self._std = torch.ones(len(self.targets))
        return self._std
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and targets from a pre-created sample.
        
        Returns
        -------
        features : torch.Tensor
            Feature tensor of shape (window_size, n_features)
        targets : torch.Tensor
            Target tensor of shape (fcast_len, n_targets)
        """
        # Initialize tensors on first access
        if self._feature_tensors is None:
            self._initialize_tensors()
        
        # Return pre-created tensors by index
        return self._feature_tensors[idx], self._target_tensors[idx]
    
    def denormalize_target(self, tensor: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """
        Denormalize target predictions back to original scale.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Normalized tensor
        target_idx : int
            Index of the target to denormalize
            
        Returns
        -------
        torch.Tensor
            Denormalized tensor
        """
        if not self.norm_params or 'mean' not in self.norm_params:
            return tensor
        
        target_name = self.targets[target_idx]
        if target_name in self.norm_params['mean'].index:
            mean = self.norm_params['mean'][target_name]
            std = self.norm_params['std'][target_name]
            return tensor * std + mean
        return tensor
    
    def denormalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalize all features in a DataFrame back to original scale.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with normalized values
            
        Returns
        -------
        pd.DataFrame
            DataFrame with denormalized values
        """
        if not self.norm_params or 'mean' not in self.norm_params:
            return df
        
        df_denorm = df.copy()
        for col in df.columns:
            if col in self.norm_params['mean'].index:
                mean = self.norm_params['mean'][col]
                std = self.norm_params['std'][col]
                df_denorm[col] = df[col] * std + mean
        
        return df_denorm
    
    def get_time_index(self, idx: int) -> pd.DatetimeIndex:
        """
        Get time index for a sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        pd.DatetimeIndex
            Time index for the forecast horizon
        """
        return self.samples[idx]['data'].index
    
    def get_full_time_index(self) -> pd.DatetimeIndex:
        """
        Get concatenated time index for all samples.
        
        Returns
        -------
        pd.DatetimeIndex
            Combined time index from all samples
        """
        all_times = []
        for sample in self.samples:
            start_idx = self.lead_time
            end_idx = self.lead_time + self.fcast_len
            all_times.append(sample['data'].index[start_idx:end_idx])
        
        if len(all_times) > 0:
            return pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in all_times]))
        return pd.DatetimeIndex([])
    
    def get_timeseries_from_samples(self) -> pd.DataFrame:
        """
        Reconstruct original time series DataFrame from samples.
        
        This method combines all samples back into a single DataFrame,
        aligning them by their original timestamps. For overlapping samples,
        the first occurrence is kept.
        
        Returns
        -------
        pd.DataFrame
            Reconstructed DataFrame with original timestamps and all columns
        """
        all_dfs = []
        for sample in self.samples:
            df = sample['data']
            all_dfs.append(df)
        
        if len(all_dfs) == 0:
            return pd.DataFrame()
        
        # Concatenate all sample DataFrames
        combined_df = pd.concat(all_dfs)
        
        # Choose first occurrence of each timestamp
        reconstructed_df = combined_df.groupby(combined_df.index).first()
        
        return reconstructed_df


class TorchDataLoader(DataLoader):
    """
    DataLoader for PyTorch-based AMP models.
    
    Creates PyTorch DataLoader objects for training, validation, and testing.
    This is a data-focused class that does not store model-specific parameters.
    Model parameters (feature_windows, target_columns, fcast_len, lead_time) are
    passed to methods when creating datasets.
    
    Parameters
    ----------
    batch_size : int, optional
        Batch size for DataLoaders. If None, uses full dataset.
    num_workers : int, default=0
        Number of worker processes for data loading
    shuffle_train : bool, default=True
        Whether to shuffle training data
    **kwargs
        Additional arguments passed to base DataLoader (data_freq, update_rate, 
        normalize, targets, etc.)
        
    Examples
    --------
    >>> dataloader = TorchDataLoader(
    ...     batch_size=32,
    ...     data_freq=60,
    ...     update_rate=1,
    ...     targets=['temp_out']
    ... )
    >>> dataloader.load_and_split_data(df, data_periods={...})
    >>> # Model provides its configuration when requesting dataloaders
    >>> train_loader, val_loader, test_loader = dataloader.get_dataloaders(
    ...     feature_windows={'temp': [(-24, -1)], 'hour': [(0, 23)]},
    ...     target_columns=['temp_out'],
    ...     fcast_len=24,
    ...     lead_time=0
    ... )
    """
    
    def __init__(
        self,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        shuffle: bool = False,
        drop_last: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _create_datasets(
        self,
        features: List,
        targets: List[str],
        input_window: Tuple[int, int],
        stride: int = 1,
        **kwargs
    ) -> Dict[str, Union[SamplesDataset, None]]:
        """
        Create PyTorch datasets using base class create_samples.
        
        This method calls create_samples() for each period and wraps the results
        in a minimal SamplesDataset that splits features from targets.
        All windowing, filtering, and validation is handled by create_samples().
        All model-specific parameters are passed as arguments (not stored as instance attributes).
        
        Parameters
        ----------
        targets : list of str
            Target column name(s) to predict
        input_window : Tuple[int, int], optional
            If provided, override feature window lengths to this fixed size
        stride : int, default=1
            Stride between consecutive samples
        **kwargs
            Additional arguments passed to create_samples
            
        Returns
        -------
        dict
            Dictionary with keys 'training', 'validation', 'testing'
            Values are SamplesDataset or None if no data available
        """
        datasets = {}
                
        # Create dataset for each split
        for split_name in ['training', 'validation', 'testing']:
            # Use base class's create_samples - it handles windowing, filtering, validation
            samples = self.create_samples(
                features=features,
                targets=targets,
                input_window=input_window,
                period=split_name,
                output_format='list',  # Get list of dicts with metadata
                stride=stride,
                **kwargs
            )
            
            if len(samples) == 0:
                logger.warning(f"No valid samples created for {split_name}")
                datasets[split_name] = None
                continue
            
            # Wrap samples in minimal dataset with normalization parameters
            datasets[split_name] = SamplesDataset(
                samples=samples,
                features=features,
                targets=targets,
                input_window=input_window,
                norm_params=self.norm_params if self.is_fitted else None
            )
            
            logger.info(
                f"Created {split_name} dataset with {len(datasets[split_name])} samples, "
                f"{len(features)} feature columns, {len(targets)} target columns"
            )
        
        return datasets   
    
    def get_training_loader(self, dataset: SamplesDataset) -> TorchDataLoaderClass:
        """
        Get PyTorch DataLoader for training data.
        
        Parameters
        ----------
        dataset : SamplesDataset
            PyTorch dataset for training
            
        Returns
        -------
        torch.utils.data.DataLoader
            PyTorch DataLoader for training
        """
        
        return TorchDataLoaderClass(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )
    
    def get_validation_loader(self, dataset: SamplesDataset) -> TorchDataLoaderClass:
        """
        Get PyTorch DataLoader for validation data.
        
        Parameters
        ----------
        dataset : SamplesDataset
            PyTorch dataset for validation
            
        Returns
        -------
        torch.utils.data.DataLoader
            PyTorch DataLoader for validation
        """
        
        return TorchDataLoaderClass(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )
    
    def get_testing_loader(self, dataset: SamplesDataset) -> TorchDataLoaderClass:
        """
        Get PyTorch DataLoader for testing data.
        
        Parameters
        ----------
        dataset : SamplesDataset
            PyTorch dataset for validation
            
        Returns
        -------
        torch.utils.data.DataLoader
            PyTorch DataLoader for testing
        """
        
        return TorchDataLoaderClass(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )
    
    def get_dataloaders(
        self,
        **kwargs
    ) -> List[TorchDataLoaderClass]:
        """
        Get all DataLoaders (training, validation, testing).
        
        Model-specific parameters must be provided by the caller (typically the model's
        get_fit_data() method). This keeps the dataloader data-focused and model-agnostic.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to _create_datasets (e.g., stride, required_features)
        
        Returns
        -------
        list of torch.utils.data.DataLoader
            List containing [train_loader, val_loader, test_loader]
        """
        datasets = self._create_datasets(
            **kwargs
        )
        return [
            self.get_training_loader(datasets['training']),
            self.get_validation_loader(datasets['validation']),
            self.get_testing_loader(datasets['testing'])
        ]
    