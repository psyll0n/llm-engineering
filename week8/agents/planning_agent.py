"""
Planning agent for orchestrating the deal discovery workflow.

This module provides the PlanningAgent which coordinates the entire deal
hunting system by managing Scanner, Ensemble, and Messaging agents to
discover, evaluate, and notify about good deals.
"""

import logging
from typing import Optional, List

from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent

# Configure logging
logger = logging.getLogger(__name__)


class PlanningAgent(Agent):
    """
    Orchestrator agent that coordinates the multi-agent deal discovery system.
    
    This is the top-level agent that manages the complete workflow:
    1. Scanner Agent: Scrapes and filters deals from RSS feeds
    2. Ensemble Agent: Estimates fair market value of products
    3. Messaging Agent: Sends alerts for good deals
    
    The planning agent implements the business logic for what constitutes
    a "good deal" (configurable discount threshold) and maintains memory
    to avoid re-processing the same deals.
    
    Attributes:
        name: Display name for logging
        color: Console color for log messages
        DEAL_THRESHOLD: Minimum discount (in dollars) to trigger an alert
        scanner: ScannerAgent instance for finding deals
        ensemble: EnsembleAgent instance for price estimation
        messenger: MessagingAgent instance for sending alerts
        
    Example Workflow:
        >>> collection = ... # ChromaDB collection
        >>> planner = PlanningAgent(collection)
        >>> memory = []
        >>> opportunity = planner.plan(memory)
        >>> if opportunity:
        ...     print(f"Found deal: Save ${opportunity.discount:.2f}!")
    """

    name = "Planning Agent"
    color = Agent.GREEN
    
    # Minimum discount in dollars to trigger an alert
    DEAL_THRESHOLD = 50

    def __init__(self, collection) -> None:
        """
        Initialize the planning agent and its subordinate agents.
        
        Creates instances of Scanner, Ensemble, and Messaging agents
        that will be coordinated to find and alert about deals.
        
        Args:
            collection: ChromaDB collection for the Ensemble's Frontier agent
            
        Raises:
            Exception: If any subordinate agent fails to initialize
        """
        self.log("Planning Agent is initializing")
        
        try:
            self.scanner = ScannerAgent()
            self.ensemble = EnsembleAgent(collection)
            self.messenger = MessagingAgent()
            self.log("Planning Agent is ready")
        except Exception as e:
            logger.error(f"Failed to initialize Planning Agent: {e}")
            raise

    def evaluate_deal(self, deal: Deal) -> Opportunity:
        """
        Evaluate a single deal by estimating its value.
        
        Uses the ensemble agent to estimate the fair market value of
        a product and calculates the potential discount/savings.
        
        Args:
            deal: Deal object with product description and price
            
        Returns:
            Opportunity object with estimate and discount calculated
            
        Example:
            >>> deal = Deal(product_description="...", price=100.0, url="...")
            >>> opp = planner.evaluate_deal(deal)
            >>> print(f"Discount: ${opp.discount:.2f}")
        """
        self.log("Planning Agent is pricing a potential deal")
        
        # Get price estimate from ensemble model
        estimate = self.ensemble.price(deal.product_description)
        
        # Calculate discount (positive = good deal)
        discount = estimate - deal.price
        
        self.log(f"Planning Agent evaluated deal: discount=${discount:.2f}")
        
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: Optional[List] = None) -> Optional[Opportunity]:
        """
        Execute the complete deal discovery workflow.
        
        Complete workflow:
        1. Scanner Agent scrapes RSS feeds and filters to best deals
        2. Ensemble Agent estimates value for each deal
        3. Sort opportunities by discount (best first)
        4. If best deal exceeds threshold, send alert via Messaging Agent
        
        Args:
            memory: List of previously processed Opportunity objects to avoid
                   re-processing the same deals (default: [])
            
        Returns:
            Best Opportunity if discount exceeds threshold, otherwise None
            
        Note:
            Only evaluates up to 5 deals per run to manage API costs.
            Deals are sorted by discount to prioritize the best savings.
            
        Example:
            >>> planner = PlanningAgent(collection)
            >>> memory = []
            >>> while True:
            ...     opp = planner.plan(memory)
            ...     if opp:
            ...         memory.append(opp)
            ...     time.sleep(3600)  # Check hourly
        """
        if memory is None:
            memory = []
            
        self.log("Planning Agent is kicking off a run")
        
        # Step 1: Get filtered deals from Scanner Agent
        selection = self.scanner.scan(memory=memory)
        
        if selection and selection.deals:
            # Step 2: Evaluate each deal with Ensemble Agent (limit to 5 for cost control)
            opportunities = [
                self.evaluate_deal(deal) 
                for deal in selection.deals[:5]
            ]
            
            # Step 3: Sort by discount to find best deal
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            
            self.log(
                f"Planning Agent identified best deal with discount=${best.discount:.2f}"
            )
            
            # Step 4: Send alert if deal exceeds threshold
            if best.discount > self.DEAL_THRESHOLD:
                self.messenger.alert(best)
                self.log("Planning Agent sent alert for good deal")
                return best
            else:
                self.log(
                    f"Planning Agent found deals but none exceeded threshold "
                    f"of ${self.DEAL_THRESHOLD}"
                )
                return None
        else:
            self.log("Planning Agent found no new deals to evaluate")
            return None