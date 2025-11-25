"""
Deal models and RSS feed scraping functionality.

This module provides data models for representing deals and opportunities,
as well as functionality to scrape deal information from RSS feeds.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Self, Optional
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

# RSS feed URLs for deal sources
DEAL_FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]


def extract_text_from_html(html_snippet: str) -> str:
    """
    Extract clean text from an HTML snippet.
    
    Uses BeautifulSoup to parse HTML and extract text content, specifically
    targeting divs with class 'snippet summary'. Removes HTML tags and
    normalizes whitespace.
    
    Args:
        html_snippet: Raw HTML string to process
        
    Returns:
        Cleaned text with HTML tags removed and newlines replaced with spaces
        
    Example:
        >>> html = '<div class="snippet summary">Great deal!</div>'
        >>> extract_text_from_html(html)
        'Great deal!'
    """
    try:
        soup = BeautifulSoup(html_snippet, 'html.parser')
        snippet_div = soup.find('div', class_='snippet summary')
        
        if snippet_div:
            description = snippet_div.get_text(strip=True)
            # Double-parse to handle any nested HTML entities
            description = BeautifulSoup(description, 'html.parser').get_text()
            # Remove any remaining HTML tags
            description = re.sub(r'<[^<]+?>', '', description)
            result = description.strip()
        else:
            result = html_snippet
        
        return result.replace('\n', ' ')
    except Exception as e:
        logger.warning(f"Error extracting text from HTML: {e}")
        return html_snippet.replace('\n', ' ')

class ScrapedDeal:
    """
    Represents a deal scraped from an RSS feed.
    
    This class handles fetching and parsing deal information from RSS feed entries,
    including making HTTP requests to fetch detailed product information.
    
    Attributes:
        category: Product category
        title: Deal title
        summary: Brief description of the deal
        url: Link to the full deal page
        details: Detailed product information
        features: Product features list
    """
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict) -> None:
        """
        Initialize ScrapedDeal from RSS feed entry.
        
        Args:
            entry: Dictionary containing RSS feed entry data with keys:
                  'title', 'summary', 'links'
                  
        Raises:
            RequestException: If fetching the deal URL fails
            KeyError: If required fields are missing from entry
        """
        self.title = entry['title']
        self.summary = extract_text_from_html(entry['summary'])
        self.url = entry['links'][0]['href']
        
        try:
            # Fetch full deal details with timeout
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content section
            content_section = soup.find('div', class_='content-section')
            if content_section:
                content = content_section.get_text()
                content = content.replace('\nmore', '').replace('\n', ' ')
                
                # Split into details and features if applicable
                if "Features" in content:
                    self.details, self.features = content.split("Features", 1)
                else:
                    self.details = content
                    self.features = ""
            else:
                logger.warning(f"No content section found for {self.url}")
                self.details = self.summary
                self.features = ""
        except requests.RequestException as e:
            logger.error(f"Failed to fetch deal details from {self.url}: {e}")
            self.details = self.summary
            self.features = ""
        except Exception as e:
            logger.error(f"Error parsing deal {self.url}: {e}")
            self.details = self.summary
            self.features = ""

    def __repr__(self) -> str:
        """
        Return a concise string representation of this deal.
        
        Returns:
            Deal title wrapped in angle brackets
        """
        return f"<{self.title}>"

    def describe(self) -> str:
        """
        Return a detailed multi-line description suitable for LLM input.
        
        This format is optimized for feeding to language models that need
        to understand and price products.
        
        Returns:
            Formatted string with title, details, features, and URL
        """
        return (
            f"Title: {self.title}\n"
            f"Details: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\n"
            f"URL: {self.url}"
        )

    @classmethod
    def fetch(cls, show_progress: bool = False, max_per_feed: int = 10) -> List[Self]:
        """
        Retrieve deals from configured RSS feeds.
        
        Fetches up to `max_per_feed` deals from each RSS feed source,
        with optional progress bar display.
        
        Args:
            show_progress: Whether to display a progress bar during fetching
            max_per_feed: Maximum number of deals to fetch per feed (default: 10)
            
        Returns:
            List of ScrapedDeal objects
            
        Note:
            Includes a 0.5 second delay between requests to be respectful
            to the RSS feed servers.
        """
        deals = []
        feed_iter = tqdm(DEAL_FEEDS) if show_progress else DEAL_FEEDS
        
        for feed_url in feed_iter:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_per_feed]:
                    try:
                        deals.append(cls(entry))
                    except Exception as e:
                        logger.error(f"Failed to process entry from {feed_url}: {e}")
                    # Be respectful to the server
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to parse feed {feed_url}: {e}")
                
        return deals

class Deal(BaseModel):
    """
    Represents a deal with essential product information.
    
    This is a simplified version of ScrapedDeal used for structured output
    from LLMs and inter-agent communication.
    
    Attributes:
        product_description: Detailed description of the product
        price: Product price in dollars
        url: Link to the deal page
    """
    product_description: str = Field(..., description="Detailed product description")
    price: float = Field(..., gt=0, description="Product price (must be positive)")
    url: str = Field(..., description="URL to the deal page")


class DealSelection(BaseModel):
    """
    Represents a curated collection of deals.
    
    Used as a structured output format for LLM responses when selecting
    and summarizing multiple deals.
    
    Attributes:
        deals: List of Deal objects
    """
    deals: List[Deal] = Field(default_factory=list, description="List of selected deals")

class Opportunity(BaseModel):
    """
    Represents a deal opportunity identified by the system.
    
    An Opportunity is created when the estimated value of a product exceeds
    its listed price, indicating a potentially good deal.
    
    Attributes:
        deal: The Deal object containing product information
        estimate: Estimated fair market value of the product
        discount: Difference between estimate and actual price (estimate - price)
        
    Example:
        >>> deal = Deal(product_description="Laptop", price=500.0, url="http://...")
        >>> opp = Opportunity(deal=deal, estimate=750.0, discount=250.0)
        >>> print(f"Save ${opp.discount:.2f}!")
        Save $250.00!
    """
    deal: Deal
    estimate: float = Field(..., description="Estimated market value")
    discount: float = Field(..., description="Potential savings (estimate - actual price)")