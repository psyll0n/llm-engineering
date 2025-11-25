"""  
Scanner agent for discovering and filtering deals from RSS feeds.

This module provides the ScannerAgent which scrapes RSS feeds for deals,
uses OpenAI to filter and select the best deals with clear descriptions
and prices, and returns structured deal data.
"""

import os
import json
from typing import Optional, List
import logging

from openai import OpenAI

from agents.deals import ScrapedDeal, DealSelection
from agents.agent import Agent

# Configure logging
logger = logging.getLogger(__name__)


class ScannerAgent(Agent):
    """
    Agent responsible for scanning RSS feeds and selecting promising deals.
    
    This agent performs two main tasks:
    1. Scrapes deals from configured RSS feeds
    2. Uses OpenAI with structured outputs to identify the 5 best deals
       based on description quality and price clarity
    
    The agent filters for deals with:
    - Detailed, high-quality product descriptions
    - Clear, positive prices (not discounts or "$XX off" language)
    - Complete information (product details, price, URL)
    
    Attributes:
        name: Display name for logging
        color: Console color for log messages
        MODEL: OpenAI model to use for deal selection
        openai: OpenAI client instance
    """

    name = "Scanner Agent"
    color = Agent.CYAN
    
    # OpenAI model configuration
    MODEL = "gpt-4o-mini"
    
    # Number of deals to select from scraped results
    NUM_DEALS_TO_SELECT = 5

    # System prompt for OpenAI - defines the agent's task and output format
    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    {"deals": [
        {
            "product_description": "Your clearly expressed summary of the product in 4-5 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a paragraph of text for each item you choose.",
            "price": 99.99,
            "url": "the url as provided"
        },
        ...
    ]}"""
    # User prompt template - provides instructions and context for deal selection
    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 deals, no more."

    def __init__(self) -> None:
        """
        Initialize the scanner agent.
        
        Sets up the OpenAI client for making API calls to filter deals.
        """
        self.log("Scanner Agent is initializing")
        self.openai = OpenAI()
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory: List) -> List[ScrapedDeal]:
        """
        Fetch deals from RSS feeds, excluding previously seen deals.
        
        Args:
            memory: List of Opportunity objects representing previously
                   processed deals (to avoid duplicates)
            
        Returns:
            List of ScrapedDeal objects that haven't been seen before
            
        Note:
            Extracts URLs from memory to filter out duplicates
        """
        self.log("Scanner Agent is about to fetch deals from RSS feeds")
        
        # Extract URLs from memory to filter duplicates
        seen_urls = {opp.deal.url for opp in memory}
        
        # Fetch all deals from RSS feeds
        try:
            scraped = ScrapedDeal.fetch()
            # Filter out deals we've already processed
            result = [scrape for scrape in scraped if scrape.url not in seen_urls]
            self.log(f"Scanner Agent received {len(result)} new deals (not already in memory)")
            return result
        except Exception as e:
            logger.error(f"Error fetching deals from RSS feeds: {e}")
            return []

    def make_user_prompt(self, scraped: List[ScrapedDeal]) -> str:
        """
        Create the user prompt for OpenAI from scraped deals.
        
        Formats all scraped deals into a prompt that asks OpenAI to select
        the best 5 deals with detailed descriptions and clear prices.
        
        Args:
            scraped: List of ScrapedDeal objects to include in prompt
            
        Returns:
            Formatted prompt string ready for OpenAI API
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: Optional[List] = None) -> Optional[DealSelection]:
        """
        Scan RSS feeds and use OpenAI to select the best deals.
        
        Main workflow:
        1. Fetch deals from RSS feeds (excluding those in memory)
        2. If deals found, send to OpenAI for selection and summarization
        3. Use structured outputs to ensure valid JSON response
        4. Filter out any deals with price <= 0
        
        Args:
            memory: List of previously processed Opportunity objects (default: [])
            
        Returns:
            DealSelection with filtered deals, or None if no deals found
            
        Note:
            Uses OpenAI's beta.chat.completions.parse with structured outputs
            to ensure response conforms to DealSelection schema
        """
        if memory is None:
            memory = []
            
        scraped = self.fetch_deals(memory)
        
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling OpenAI using Structured Output")
            
            try:
                result = self.openai.beta.chat.completions.parse(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=DealSelection
                )
                
                deal_selection = result.choices[0].message.parsed
                
                # Filter out deals with invalid prices
                deal_selection.deals = [
                    deal for deal in deal_selection.deals 
                    if deal.price > 0
                ]
                
                self.log(
                    f"Scanner Agent received {len(deal_selection.deals)} "
                    f"selected deals with price>0 from OpenAI"
                )
                return deal_selection
                
            except Exception as e:
                logger.error(f"Error calling OpenAI for deal selection: {e}")
                return None
        
        self.log("Scanner Agent found no new deals to process")
        return None
                
