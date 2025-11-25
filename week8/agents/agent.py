"""
Base agent module providing abstract Agent class for multi-agent systems.

This module defines the base Agent class that all specialized agents inherit from.
It provides colored logging capabilities to differentiate between agents in console output.
"""

import logging
from typing import ClassVar


class Agent:
    """
    Abstract base class for all agents in the system.
    
    Provides a standardized logging interface with color-coded output to help
    identify different agents during execution. Each agent subclass should define
    its own name and color for visual differentiation.
    
    Attributes:
        name: The agent's display name used in log messages
        color: ANSI color code for this agent's log output
        
    Class Attributes:
        RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE: ANSI foreground color codes
        BG_BLACK: ANSI background color code
        RESET: ANSI reset code to return to default terminal colors
        
    Example:
        >>> class MyAgent(Agent):
        ...     name = "My Custom Agent"
        ...     color = Agent.BLUE
        ...     
        ...     def process(self):
        ...         self.log("Processing started")
    """

    # ANSI foreground color codes for terminal output
    RED: ClassVar[str] = '\033[31m'
    GREEN: ClassVar[str] = '\033[32m'
    YELLOW: ClassVar[str] = '\033[33m'
    BLUE: ClassVar[str] = '\033[34m'
    MAGENTA: ClassVar[str] = '\033[35m'
    CYAN: ClassVar[str] = '\033[36m'
    WHITE: ClassVar[str] = '\033[37m'
    
    # ANSI background color code
    BG_BLACK: ClassVar[str] = '\033[40m'
    
    # ANSI reset code to return to default terminal colors
    RESET: ClassVar[str] = '\033[0m'

    # Instance attributes to be overridden by subclasses
    name: str = ""
    color: str = '\033[37m'

    def log(self, message: str) -> None:
        """
        Log an info-level message with agent identification and color coding.
        
        This method prepends the agent's name to the message and applies the
        agent's designated color to make log output easily identifiable.
        
        Args:
            message: The log message to display
            
        Example:
            >>> agent = Agent()
            >>> agent.name = "TestAgent"
            >>> agent.log("Task completed")
            # Outputs: [TestAgent] Task completed (in white on black)
        """
        color_code = self.BG_BLACK + self.color
        formatted_message = f"[{self.name}] {message}"
        logging.info(color_code + formatted_message + self.RESET)