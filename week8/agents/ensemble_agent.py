"""  
Ensemble agent combining multiple pricing models.

This module provides the EnsembleAgent which combines predictions from
multiple specialized agents (SpecialistAgent, FrontierAgent, RandomForestAgent)
using a trained linear regression ensemble model.
"""

import logging
import pandas as pd
import joblib

from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent

# Configure logging
logger = logging.getLogger(__name__)


class EnsembleAgent(Agent):
    """
    Meta-agent that combines multiple pricing models for improved accuracy.
    
    This agent orchestrates three specialized pricing agents and uses a
    trained linear regression model to weight their predictions optimally.
    The ensemble approach typically provides more accurate and robust
    estimates than any single model.
    
    Component Agents:
        - SpecialistAgent: Fine-tuned LLM hosted on Modal
        - FrontierAgent: RAG-based approach using similar products
        - RandomForestAgent: Traditional ML model with vector embeddings
        
    The ensemble model combines predictions by:
        1. Getting predictions from each component agent
        2. Computing min and max of all predictions
        3. Using linear regression to weight all features
        
    Attributes:
        name: Display name for logging
        color: Console color for log messages
        specialist: Instance of SpecialistAgent
        frontier: Instance of FrontierAgent
        random_forest: Instance of RandomForestAgent
        model: Trained scikit-learn LinearRegression model
    """

    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(self, collection) -> None:
        """
        Initialize the ensemble agent and its component models.
        
        Creates instances of all three component agents and loads the
        pre-trained ensemble weighting model from disk.
        
        Args:
            collection: ChromaDB collection for FrontierAgent's RAG system
            
        Raises:
            FileNotFoundError: If ensemble_model.pkl is not found
            Exception: If any component agent fails to initialize
        """
        self.log("Initializing Ensemble Agent")
        
        try:
            # Initialize component agents
            try:
                self.specialist = SpecialistAgent()
                self.specialist_available = True
            except Exception as e:
                logger.warning(f"Specialist Agent unavailable: {e}")
                self.specialist = None
                self.specialist_available = False
                
            self.frontier = FrontierAgent(collection)
            self.random_forest = RandomForestAgent()
            
            # Load the ensemble weighting model
            self.model = joblib.load('ensemble_model.pkl')
            
            self.log("Ensemble Agent is ready")
        except FileNotFoundError:
            logger.error("ensemble_model.pkl not found. Train the ensemble model first.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ensemble Agent: {e}")
            raise

    def price(self, description: str) -> float:
        """
        Estimate product price using the ensemble of models.
        
        Workflow:
        1. Request price predictions from all three component agents
        2. Construct feature vector with predictions + min/max
        3. Use linear regression model to compute weighted estimate
        4. Ensure non-negative result
        
        Args:
            description: Detailed product description to price
            
        Returns:
            Estimated price (guaranteed >= 0)
            
        Raises:
            Exception: If any component agent fails during prediction
            
        Example:
            >>> ensemble = EnsembleAgent(collection)
            >>> price = ensemble.price("Wireless gaming mouse with RGB...")
            Ensemble Agent complete - returning $45.99
        """
        self.log(
            "Running Ensemble Agent - collaborating with specialist, "
            "frontier and random forest agents"
        )
        
        try:
            # Get predictions from each component agent
            specialist = None
            if self.specialist_available:
                try:
                    specialist = self.specialist.price(description)
                except Exception as e:
                    logger.warning(f"Specialist Agent failed during pricing: {e}")
                    specialist = None
                    
            frontier = self.frontier.price(description)
            random_forest = self.random_forest.price(description)
            
            # Construct feature DataFrame for ensemble model
            if specialist is not None:
                X = pd.DataFrame({
                    'Specialist': [specialist],
                    'Frontier': [frontier],
                    'RandomForest': [random_forest],
                    'Min': [min(specialist, frontier, random_forest)],
                    'Max': [max(specialist, frontier, random_forest)],
                })
            else:
                # Fallback: use frontier as specialist estimate
                logger.info("Using fallback estimation without Specialist Agent")
                specialist_estimate = (frontier + random_forest) / 2
                X = pd.DataFrame({
                    'Specialist': [specialist_estimate],
                    'Frontier': [frontier],
                    'RandomForest': [random_forest],
                    'Min': [min(specialist_estimate, frontier, random_forest)],
                    'Max': [max(specialist_estimate, frontier, random_forest)],
                })
            
            # Get weighted prediction from ensemble model
            prediction = self.model.predict(X)[0]
            
            # Ensure non-negative price
            result = max(0, prediction)
            
            self.log(f"Ensemble Agent complete - returning ${result:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble price prediction: {e}")
            raise