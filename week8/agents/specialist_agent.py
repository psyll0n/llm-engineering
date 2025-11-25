"""
Specialist agent for price estimation using a fine-tuned LLM.

This module provides the SpecialistAgent which connects to a remotely-hosted
fine-tuned language model running on Modal to estimate product prices.
"""

import logging
import modal

from agents.agent import Agent

# Configure logging
logger = logging.getLogger(__name__)


class SpecialistAgent(Agent):
    """
    Agent that uses a fine-tuned LLM running on Modal for price estimation.
    
    This agent connects to a remote service hosted on Modal that runs a
    fine-tuned language model specifically trained for pricing products.
    The model is accessed via Modal's remote invocation capabilities.
    
    Attributes:
        name: Display name for logging
        color: Console color for log messages  
        pricer: Modal class instance for remote price estimation
        
    Note:
        Requires Modal to be properly configured with credentials and
        the 'pricer-service' deployment to be running.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self) -> None:
        """
        Initialize the specialist agent and connect to Modal.
        
        Establishes connection to the remotely-hosted fine-tuned model
        running on Modal's infrastructure.
        
        Raises:
            Exception: If connection to Modal service fails
        """
        self.log("Specialist Agent is initializing - connecting to Modal")
        
        try:
            # Connect to the Modal-hosted pricing service
            Pricer = modal.Cls.from_name("pricer-service", "Pricer")
            self.pricer = Pricer()
            self.log("Specialist Agent is ready")
        except Exception as e:
            logger.error(f"Failed to connect to Modal pricer service: {e}")
            raise
        
    def price(self, description: str) -> float:
        """
        Estimate product price using the remote fine-tuned model.
        
        Makes a remote procedure call to the Modal-hosted model which
        analyzes the product description and returns a price estimate
        based on its training data.
        
        Args:
            description: Detailed product description to price
            
        Returns:
            Estimated price as a float
            
        Raises:
            Exception: If remote call to Modal service fails
            
        Example:
            >>> agent = SpecialistAgent()
            >>> price = agent.price("High-end gaming laptop with RTX 4090...")
            >>> print(f"Estimated price: ${price:.2f}")
        """
        self.log("Specialist Agent is calling remote fine-tuned model")
        
        try:
            result = self.pricer.price.remote(description)
            self.log(f"Specialist Agent completed - predicting ${result:.2f}")
            return result
        except Exception as e:
            logger.error(f"Error calling remote pricer: {e}")
            # Re-raise to let caller handle
            raise
