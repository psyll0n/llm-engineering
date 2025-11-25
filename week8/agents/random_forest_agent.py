"""  
Random forest agent for ML-based price estimation.

This module provides the RandomForestAgent which uses a traditional machine
learning approach (Random Forest) combined with sentence embeddings to
estimate product prices.
"""

import logging
import joblib
from sentence_transformers import SentenceTransformer

from agents.agent import Agent

# Configure logging
logger = logging.getLogger(__name__)


class RandomForestAgent(Agent):
    """
    Agent that uses Random Forest regression for price estimation.
    
    This agent combines:
    1. Sentence embeddings from a transformer model to encode product descriptions
    2. A trained Random Forest regressor to predict prices from embeddings
    
    The approach is more traditional ML compared to LLM-based approaches,
    but can be very effective with sufficient training data.
    
    Attributes:
        name: Display name for logging
        color: Console color for log messages
        vectorizer: SentenceTransformer model for creating embeddings
        model: Trained scikit-learn RandomForestRegressor
    """

    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self) -> None:
        """
        Initialize the random forest agent.
        
        Loads the pre-trained sentence transformer model for creating
        embeddings and the trained random forest model for price prediction.
        
        Raises:
            FileNotFoundError: If random_forest_model.pkl is not found
            Exception: If models fail to load
        """
        self.log("Random Forest Agent is initializing")
        
        try:
            # Load sentence transformer for creating embeddings
            self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load pre-trained random forest model
            self.model = joblib.load('random_forest_model.pkl')
            
            self.log("Random Forest Agent is ready")
        except FileNotFoundError:
            logger.error("random_forest_model.pkl not found. Train the model first.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Random Forest Agent: {e}")
            raise

    def price(self, description: str) -> float:
        """
        Estimate product price using random forest regression.
        
        Workflow:
        1. Convert product description to vector embedding
        2. Feed embedding to random forest model for prediction
        3. Ensure non-negative result
        
        Args:
            description: Product description text to price
            
        Returns:
            Estimated price (guaranteed >= 0)
            
        Raises:
            Exception: If vectorization or prediction fails
            
        Example:
            >>> agent = RandomForestAgent()
            >>> price = agent.price("Bluetooth wireless headphones...")
            Random Forest Agent completed - predicting $35.50
        """        
        self.log("Random Forest Agent is starting a prediction")
        
        try:
            # Convert text to vector embedding
            vector = self.vectorizer.encode([description])
            
            # Get price prediction from random forest
            prediction = self.model.predict(vector)[0]
            
            # Ensure non-negative price
            result = max(0, prediction)
            
            self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in random forest price prediction: {e}")
            raise