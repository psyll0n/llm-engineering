"""  
Frontier agent using RAG (Retrieval-Augmented Generation) for price estimation.

This module provides the FrontierAgent which combines vector similarity search
with LLM reasoning to estimate product prices based on similar products.
"""

import os
import re
import logging
from typing import List, Dict, Tuple

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

from agents.agent import Agent

# Configure logging
logger = logging.getLogger(__name__)


class FrontierAgent(Agent):
    """
    Agent that uses RAG (Retrieval-Augmented Generation) for price estimation.
    
    This agent implements a sophisticated pricing approach:
    1. Encodes product description into a vector embedding
    2. Searches ChromaDB for similar products (vector similarity)
    3. Provides similar products as context to an LLM
    4. LLM estimates price based on the contextual examples
    
    Supports both OpenAI and DeepSeek as the LLM backend, with automatic
    fallback to OpenAI if DeepSeek credentials aren't available.
    
    Attributes:
        name: Display name for logging
        color: Console color for log messages
        MODEL: LLM model identifier (gpt-4o-mini or deepseek-chat)
        client: OpenAI-compatible client instance
        collection: ChromaDB collection for similarity search
        model: SentenceTransformer for creating embeddings
    """

    name = "Frontier Agent"
    color = Agent.BLUE

    # Default model - will be overridden if DeepSeek is available
    MODEL = "gpt-4o-mini"
    
    def __init__(self, collection) -> None:
        """
        Initialize the frontier agent with LLM and vector search capabilities.
        
        Sets up connections to:
        - OpenAI or DeepSeek API for LLM inference
        - ChromaDB collection for vector similarity search
        - SentenceTransformer for creating embeddings
        
        Args:
            collection: ChromaDB collection containing product embeddings
            
        Raises:
            Exception: If initialization of any component fails
        """
        self.log("Initializing Frontier Agent")
        
        try:
            # Check for DeepSeek API key, otherwise use OpenAI
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_api_key:
                self.client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
                self.MODEL = "deepseek-chat"
                self.log("Frontier Agent configured with DeepSeek")
            else:
                self.client = OpenAI()
                self.MODEL = "gpt-4o-mini"
                self.log("Frontier Agent configured with OpenAI")
            
            # Store ChromaDB collection reference
            self.collection = collection
            
            # Load sentence transformer for embeddings
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            self.log("Frontier Agent is ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Frontier Agent: {e}")
            raise

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create contextual prompt text from similar products.
        
        Formats similar products and their prices into a text block that
        provides examples for the LLM to reference when estimating.
        
        Args:
            similars: List of product description strings
            prices: List of corresponding prices
            
        Returns:
            Formatted context string for inclusion in prompt
            
        Example:
            >>> context = agent.make_context(
            ...     ["Gaming mouse", "Office mouse"],
            ...     [45.99, 12.99]
            ... )
        """
        message = (
            "To provide some context, here are some other items that might be "
            "similar to the item you need to estimate.\n\n"
        )
        
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        
        return message

    def messages_for(
        self,
        description: str,
        similars: List[str],
        prices: List[float]
    ) -> List[Dict[str, str]]:
        """
        Construct message list for LLM API call.
        
        Creates a properly formatted conversation with:
        - System message defining the agent's role
        - User message with context and the pricing question
        - Assistant prefix to guide the response format
        
        Args:
            description: Product description to price
            similars: List of similar product descriptions for context
            prices: List of prices for similar products
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        system_message = (
            "You estimate prices of items. Reply only with the price, "
            "no explanation"
        )
        
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(
        self,
        description: str,
        n_results: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Find similar products using vector similarity search.
        
        Encodes the product description and queries ChromaDB to find
        the most similar products based on embedding similarity.
        
        Args:
            description: Product description to find matches for
            n_results: Number of similar products to return (default: 5)
            
        Returns:
            Tuple of (descriptions, prices) for similar products
            
        Raises:
            Exception: If ChromaDB query fails
        """
        self.log(
            f"Frontier Agent is performing RAG search in ChromaDB "
            f"to find {n_results} similar products"
        )
        
        try:
            # Create embedding vector for the query
            vector = self.model.encode([description])
            
            # Query ChromaDB for similar products
            results = self.collection.query(
                query_embeddings=vector.astype(float).tolist(),
                n_results=n_results
            )
            
            # Extract documents and prices from results
            documents = results['documents'][0][:]
            prices = [m['price'] for m in results['metadatas'][0][:]]
            
            self.log("Frontier Agent found similar products")
            return documents, prices
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise

    def extract_price(self, text: str) -> float:
        """
        Extract a numeric price from a text string.
        
        Uses regex to find floating point or integer numbers in the text,
        removing common price formatting like $ and commas.
        
        Args:
            text: String potentially containing a price
            
        Returns:
            Extracted price as float, or 0.0 if no number found
            
        Example:
            >>> agent.extract_price("Price is $1,234.56")
            1234.56
            >>> agent.extract_price("Cost: 99")
            99.0
        """
        # Remove common price formatting
        cleaned = text.replace('$', '').replace(',', '')
        
        # Find first number (integer or decimal)
        match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
        
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Estimate product price using RAG with LLM.
        
        Complete workflow:
        1. Find 5 similar products from ChromaDB
        2. Construct prompt with similar products as context
        3. Call LLM (OpenAI or DeepSeek) for price estimation
        4. Parse numeric price from LLM response
        
        Args:
            description: Detailed product description to price
            
        Returns:
            Estimated price as float
            
        Raises:
            Exception: If similarity search or LLM call fails
            
        Example:
            >>> agent = FrontierAgent(collection)
            >>> price = agent.price("Mechanical keyboard with RGB lighting...")
            Frontier Agent completed - predicting $89.99
        """
        try:
            # Find similar products for context
            documents, prices = self.find_similars(description)
            
            self.log(
                f"Frontier Agent calling {self.MODEL} with context "
                f"including {len(documents)} similar products"
            )
            
            # Call LLM with context
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages_for(description, documents, prices),
                seed=42,  # For reproducibility
                max_tokens=5  # We only need a price number
            )
            
            # Extract price from response
            reply = response.choices[0].message.content
            result = self.extract_price(reply)
            
            self.log(f"Frontier Agent completed - predicting ${result:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in frontier agent price prediction: {e}")
            raise
        