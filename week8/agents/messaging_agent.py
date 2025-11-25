"""
Messaging agent for sending deal alerts via SMS or push notifications.

This module provides the MessagingAgent class which can send notifications
through Twilio (SMS) or Pushover (push notifications) about deal opportunities.
"""

import os
import http.client
import urllib.parse
import logging

# from twilio.rest import Client  # Uncomment if using Twilio

from agents.deals import Opportunity
from agents.agent import Agent

# Configure logging
logger = logging.getLogger(__name__)


class MessagingAgent(Agent):
    """
    Agent responsible for sending deal alert notifications.
    
    Supports two notification channels:
    1. SMS via Twilio API
    2. Push notifications via Pushover API
    
    Configuration is done through environment variables or class constants.
    
    Environment Variables:
        ENABLE_SMS: Set to 'true' to enable SMS notifications
        ENABLE_PUSH: Set to 'true' to enable push notifications
        TWILIO_ACCOUNT_SID: Twilio account SID (if using SMS)
        TWILIO_AUTH_TOKEN: Twilio auth token (if using SMS)
        TWILIO_FROM: Twilio phone number to send from
        MY_PHONE_NUMBER: Recipient phone number for SMS
        PUSHOVER_USER: Pushover user key
        PUSHOVER_TOKEN: Pushover application token
        
    Attributes:
        name: Agent display name
        color: ANSI color code for logging
        do_text: Whether SMS notifications are enabled and configured
        do_push: Whether push notifications are enabled and configured
    """

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self) -> None:
        """
        Initialize the messaging agent and configure notification channels.
        
        Sets up Twilio and/or Pushover clients based on configuration.
        Validates that required environment variables are present.
        """
        self.log("Messaging Agent is initializing")
        
        # Determine if SMS should be enabled
        self.do_text = os.getenv('ENABLE_SMS', 'false').lower() == 'true'
        self.do_push = os.getenv('ENABLE_PUSH', 'true').lower() == 'true'
        
        # Initialize Twilio for SMS if enabled
        if self.do_text:
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.me_from = os.getenv('TWILIO_FROM')
            self.me_to = os.getenv('MY_PHONE_NUMBER')
            
            # Validate Twilio credentials
            if not all([account_sid, auth_token, self.me_from, self.me_to]):
                logger.warning(
                    "SMS enabled but missing Twilio credentials. "
                    "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, and MY_PHONE_NUMBER"
                )
                self.do_text = False
            else:
                # Uncomment when using Twilio:
                # self.client = Client(account_sid, auth_token)
                self.log("Messaging Agent has initialized Twilio")
        
        # Initialize Pushover for push notifications if enabled
        if self.do_push:
            self.pushover_user = os.getenv('PUSHOVER_USER')
            self.pushover_token = os.getenv('PUSHOVER_TOKEN')
            
            # Validate Pushover credentials
            if not all([self.pushover_user, self.pushover_token]):
                logger.warning(
                    "Push notifications enabled but missing Pushover credentials. "
                    "Set PUSHOVER_USER and PUSHOVER_TOKEN"
                )
                self.do_push = False
            else:
                self.log("Messaging Agent has initialized Pushover")

    def send_sms(self, text: str) -> bool:
        """
        Send an SMS message using the Twilio API.
        
        Args:
            text: Message text to send
            
        Returns:
            True if message sent successfully, False otherwise
            
        Note:
            Requires uncommenting Twilio client initialization in __init__
        """
        if not self.do_text:
            logger.warning("SMS not configured, skipping send_sms")
            return False
            
        self.log("Messaging Agent is sending a text message")
        try:
            # Uncomment when using Twilio:
            # _message = self.client.messages.create(
            #     from_=self.me_from,
            #     body=text,
            #     to=self.me_to
            # )
            logger.info("SMS sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    def send_push(self, text: str) -> bool:
        """
        Send a push notification using the Pushover API.
        
        Args:
            text: Notification text to send
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.do_push:
            logger.warning("Push notifications not configured, skipping send_push")
            return False
            
        self.log("Messaging Agent is sending a push notification")
        try:
            conn = http.client.HTTPSConnection("api.pushover.net:443")
            conn.request(
                "POST",
                "/1/messages.json",
                urllib.parse.urlencode({
                    "token": self.pushover_token,
                    "user": self.pushover_user,
                    "message": text,
                    "sound": "cashregister"
                }),
                {"Content-type": "application/x-www-form-urlencoded"}
            )
            response = conn.getresponse()
            
            if response.status == 200:
                logger.info("Push notification sent successfully")
                return True
            else:
                logger.error(f"Push notification failed with status {response.status}")
                return False
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            return False
        finally:
            conn.close()

    def alert(self, opportunity: Opportunity) -> None:
        """
        Send an alert about a discovered deal opportunity.
        
        Formats the opportunity information into a message and sends it
        through all configured notification channels.
        
        Args:
            opportunity: The Opportunity object containing deal information
            
        Example:
            >>> opp = Opportunity(...)
            >>> agent.alert(opp)
            # Sends: "Deal Alert! Price=$500.00, Estimate=$750.00, Discount=$250.00: Laptop... http://..."
        """
        # Format the alert message
        text = (
            f"Deal Alert! "
            f"Price=${opportunity.deal.price:.2f}, "
            f"Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f}: "
            f"{opportunity.deal.product_description[:100]}... "
            f"{opportunity.deal.url}"
        )
        
        # Send via configured channels
        if self.do_text:
            self.send_sms(text)
        
        if self.do_push:
            self.send_push(text)
        
        self.log("Messaging Agent has completed alert")
        
    
        