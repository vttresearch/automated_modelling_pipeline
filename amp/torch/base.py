"""
Docstring for amp.torch.base

Contains the base class for Torch-based models. These models are trained using Lightning framework. The models here should 
"""
from abc import abstractmethod
import logging
from amp.utils import form_multi_output  # This method is used to create output in case of output_mode='target'
from torch.utils.data import DataLoader
import torch
import pandas as pandas
from lightning.pytorch.loggers import TensorBoardLogger
logger = logging.getLogger(__name__)
    
class BaseTorchModel:

    def __init__(self, trainer_kwargs=None, tb_name='default', *args, **kwargs):
        """
        Initialize the BaseTorchModel with a trainer and other parameters.
        
        Parameters
        ----------
        trainer_kwargs : dict, optional
            Keyword arguments for creating the Trainer if trainer is None.
        tb_name : str, optional
            Name used for TensorBoard logging (experiment sub-directory). Defaults to 'default'.
        """
        self.trainer_kwargs = trainer_kwargs or {}
        self.tb_name = tb_name
        self.ckpt_path = None  # Set to a checkpoint path to resume training from that checkpoint

    def _predict_preprocess(self, df):
        # Any preprocessing steps before prediction can be added here
        return df
    
    @abstractmethod
    def _single_predict(self, df):
        # Implement single output prediction logic here
        pass

    @abstractmethod
    def _target_predict(self, df):
        # Implement target-wise output prediction logic here
        pass

    @abstractmethod
    def _get_dataloaders(self, training_set, validation_set=[], testing_set=[]) -> list:
        # Create and return dataloaders for training, validation, and testing datasets
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        # Implement the training step logic here
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        # Implement the validation step logic here
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        # Implement the test step logic here
        pass

    @abstractmethod
    def configure_optimizers(self):
        # Implement optimizer configuration here
        pass

    @abstractmethod
    def forward(self, x):
        # Implement the forward pass logic here
        pass

    def predict(self, df, output_mode='single', history_len=None, forecast_len=None, current_index=None):
        """
        Predict method for the model. This method will handle the prediction logic based on the output mode.
        Prediction window parameters (history_len, forecast_len, current_index) can be provided directly or will default to input_window values.
        If parameters are not provided, the given dataframe shape should be size (history_len + forecast_len, num_features) to ensure correct slicing for prediction.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data for prediction. Should contain the necessary features as defined in the model.
        output_mode : str, optional
            The mode of output prediction. Can be 'single' for single output or 'target' for multiple output. Default is 'single'.
        history_len : int, optional
            The length of historical data to consider for prediction.   
        forecast_len : int, optional
            The length of the forecast horizon.
        current_index : int, optional
            The current index in the time series.
        """
        # If history_len is explicitly provided, ignore the input_window 
        history_len = history_len or -self.input_window[0]
        forecast_len = forecast_len or self.input_window[1] + 1
        current_index = current_index or history_len  # Default to history_len if not provided

        if len(df) < history_len + forecast_len:
            logger.warning(
                f"Input dataframe length ({len(df)}) does not contain enough data for the specified history_len and forecast_len. "
                f"(history_len = {history_len}, forecast_len = {forecast_len}). "
            )
        elif len(df) > history_len + forecast_len:
            logger.debug(
                f"Input dataframe length ({len(df)}) is greater than the required length for prediction (history_len = {history_len}, forecast_len = {forecast_len}). "
                f"The current_index is set to {current_index}, which will be used to slice the dataframe for prediction. "
            )


        df = df.copy()
        df = self._predict_preprocess(df)

        if output_mode == 'single':
            return self._single_predict(
                df, 
                history_len=history_len, 
                forecast_len=forecast_len, 
                current_index=current_index
            )
        elif output_mode == 'target':
            predictions = self._target_predict(
                df, 
                history_len=history_len, 
                forecast_len=forecast_len, 
                current_index=current_index
            )
            # Get parameters from model instance
            lead_time = getattr(self, 'lead_time')
            data_freq = getattr(self, 'data_freq')
            outputs = getattr(self, 'outputs')
            
            return form_multi_output(
                predictions,
                outputs,
                lead_time,
                forecast_len,
                data_freq
            )
        else:
            logger.warning('Wrong output mode selected for predicting!')
    
    def fit(self, training_set, validation_set=None, testing_set=None):
        """
        Fit the model using DataFrames for each data split.

        Calls ``self._get_dataloaders(training_set, validation_set, testing_set)``
        to let each concrete model build its own PyTorch DataLoaders, then runs
        the Lightning training and (optionally) test phase.

        To resume training from a checkpoint, set ``model.ckpt_path`` before calling fit::

            model.ckpt_path = 'checkpoints/model-epoch=49-val_loss=0.1234.ckpt'
            model.fit(training_set, validation_set)

        The checkpoint restores model weights, optimizer state, and epoch counter.
        Remember to also increase ``max_epochs`` in ``trainer_kwargs`` if you want
        to train beyond the original limit.

        Parameters
        ----------
        training_set : list of pd.DataFrame
            Training data (one DataFrame per disjoint time period).
        validation_set : list of pd.DataFrame, optional
            Validation data.
        testing_set : list of pd.DataFrame, optional
            Testing data. When None the test phase is skipped.
        """
        train_loader, val_loader, test_loader = self._get_dataloaders(
            training_set, validation_set, testing_set
        )
        
        # Create trainer
        from amp.torch.trainer import Trainer

        # Extract datasets from loaders for trainer callbacks
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset if test_loader is not None else None

        # Calculate indices for each split
        train_indices = list(range(0, len(train_dataset)))
        val_indices = list(range(0, len(val_dataset)))
        test_indices = list(range(0, len(test_dataset))) if test_dataset is not None else []

        # Create logger specific to model type
        tb_name = getattr(self, 'tb_name', 'default')
        logs_path = f'logs'
        tensorboard_logger = TensorBoardLogger(logs_path, name=tb_name)
        logger.info(f"Logging tensorboard logs to: logs/{tb_name}")
        
        # Use provided trainer_kwargs or defaults
        # Pass all three datasets for callbacks to access complete data
        trainer = Trainer(
            logger=tensorboard_logger,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            **self.trainer_kwargs
        )

        # Model fitting with validation
        ckpt_path = getattr(self, 'ckpt_path', None)
        if ckpt_path is not None:
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.fit(
            model=self,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )
        self.ckpt_path = None  # Clear after use so a subsequent fit() starts fresh

        # Run testing after training completes (skip if no test data)
        if test_loader is not None:
            logger.info("Running test phase...")
            trainer.test(
                model=self,
                dataloaders=test_loader
            )
        else:
            logger.info("No testing data provided, skipping test phase.")
        
        

    
    
