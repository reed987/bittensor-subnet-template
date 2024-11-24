# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 reed987

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import bittensor as bt

import crypto_ai
from crypto_ai.base.miner import BaseMinerNeuron

import os
import requests
import configparser
import joblib
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import typing

import keras
import sklearn

class ModelType:
    """Class containing static model type constants."""
    LR = "lr"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"

class Miner(BaseMinerNeuron):
    """
    Miner neuron class.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Load default symbols, model path and currency from config file
        self.crypto_symbols, self.model_path, self.currency = self.load_config()
        
        # Load model
        if self.model_path:
            self.model = self.load_model(self.model_path)  # Load the model based on its file type

            self.model_type = self.check_model_type(self.model)
        else:
            raise ValueError("Model path must be provided in the configuration file.")
        
    def load_config(self):
        # default values
        config = configparser.ConfigParser()
        config.read('config.properties')
        
        # Load symbols, model path, and currency from DEFAULT section
        if config.has_section('DEFAULT'):
            symbols = config.get('DEFAULT', 'symbols', fallback='tao').split(',')
            model_path = config.get('DEFAULT', 'model_path', fallback=None)
            currency = config.get('DEFAULT', 'currency', fallback='usd')  # Default to 'usd' if not specified
            
            return [symbol.strip() for symbol in symbols], model_path, currency  # Strip whitespace from symbols
        
        return [], None, 'usd'  # Return empty list, None for model path, and default currency

    def load_model(self, model_path):
        """Load a model from various formats based on its file extension."""
        _, ext = os.path.splitext(model_path)  # Get the file extension
        
        # Load PyTorch models (.pth or .pt)
        if ext == '.pth' or ext == '.pt':
            return torch.load(model_path)  # Load PyTorch models
            
        # Load Keras models (.h5)
        elif ext == '.h5':
            from tensorflow.keras.models import load_model
            return load_model(model_path)  # Load Keras models
            
        # Load joblib or pickle models (.pkl or .joblib)
        elif ext == '.pkl' or ext == '.joblib':
            return joblib.load(model_path)  # Load joblib or pickle models
            
        else:
            raise ValueError(f"Unsupported model format: {ext}")  # Raise error for unsupported formats

    def check_model_type(model):
        """Check the type of the loaded model."""
        
        # Check for Logistic Regression (sklearn .pkl or .joblib)
        if isinstance(model, sklearn.linear_model.LogisticRegression):
            return ModelType.LR
        
        # Check for LSTM (Keras .h5)
        elif isinstance(model, keras.Model) and any(ModelType.LSTM in layer.name.lower() for layer in model.layers):
            return ModelType.LSTM
        
        # Check for GRU (Keras .h5)
        elif isinstance(model, keras.Model) and any(ModelType.GRU in layer.name.lower() for layer in model.layers):
            return ModelType.GRU
        
        # Check for CNN (Keras .h5)
        elif isinstance(model, keras.Model) and any('conv' in layer.name.lower() for layer in model.layers):
            return ModelType.CNN
        
        # If the model is a PyTorch model, we can check its architecture (.pt or .pth)
        elif isinstance(model, torch.nn.Module):
            if any(isinstance(layer, torch.nn.LSTM) for layer in model.children()):
                return ModelType.LSTM
            elif any(isinstance(layer, torch.nn.GRU) for layer in model.children()):
                return ModelType.GRU
            elif any(isinstance(layer, torch.nn.Conv2d) for layer in model.children()):
                return ModelType.CNN
            elif isinstance(model, torch.nn.Linear):  # Assuming it's a simple linear model
                return ModelType.LR

        raise ValueError("Unknown or unsupported model type.")
    
    def get_historical_prices(self, symbols: list, currency: str):
        """
        # TODO Get/load historical prices
        """
        pass

    def generate_prediction(self, historical_prices):
        """
        Generate predictions using the loaded model.
        
        Args:
            historical_prices: Current prices of cryptocurrencies.

        Returns:
            predicted prices
        """
        if self.model_type == ModelType.LR:
            # TODO logic for Linear Regression
            pass
        elif self.model_type == ModelType.LSTM:
            # TODO logic for Long Short Term Memory
            pass
        elif self.model_type == ModelType.CNN:
            # TODO logic for Convolutional Neural Network
            pass
        elif self.model_type == ModelType.GRU:
            # TODO logic for Gated Recurrent Unit
            pass
        pass

    async def forward(self, synapse: crypto_ai.protocol.Dummy) -> crypto_ai.protocol.Dummy:
        requested_symbols = synapse.request_data.get('symbols', self.crypto_symbols)
        requested_currency = synapse.request_data.get('currency', self.currency)  # Use request currency or default

        is_valid = any((len(symbol) <= 5 or len(symbol) >= 2) for symbol in requested_symbols)

        if not is_valid:
            synapse.dummy_output = "No supported crypto symbols provided."
            return synapse

        historical_prices = self.get_historical_prices(requested_symbols, requested_currency)

        if historical_prices:
            predicted_prices = self.generate_prediction(historical_prices)
            
            predictions_with_symbols = {
                symbol: {
                    "predicted_price": predicted_prices[symbol],
                    "currency": requested_currency  # Currency returned ie USD
                } for symbol in requested_symbols if symbol in predicted_prices
            }

            synapse.dummy_output = predictions_with_symbols
            
            return synapse
        else:
            synapse.dummy_output = "Error fetching prices"
            return synapse
    
    async def blacklist(self, synapse: crypto_ai.protocol.Dummy) -> typing.Tuple[bool, str]:
        pass
        
    async def priority(self, synapse: crypto_ai.protocol.Dummy) -> float:
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
