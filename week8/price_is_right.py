"""Gradio web interface for the Price is Right deal hunting application.

Provides a simple UI for monitoring and interacting with the autonomous deal hunting
agent framework. The interface displays discovered deals in a table format and allows
users to send alerts for selected opportunities.
"""

from typing import List, Optional
import gradio as gr
from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity, Deal

class App:
    """Gradio web application for the deal hunting agent framework.
    
    This class creates a simple web interface that:
    - Displays discovered deals in a table
    - Refreshes automatically every 60 seconds
    - Allows users to select deals to send alerts
    """

    def __init__(self):
        """Initialize the application with no agent framework (lazy initialization)."""
        self.agent_framework: Optional[DealAgentFramework] = None

    def run(self):
        """Launch the Gradio web interface.
        
        Creates and configures the UI with:
        - Header and description
        - Opportunities table
        - Auto-refresh timer (60 seconds)
        - Row selection handler for alerts
        """
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
        
            def table_for(opps: List[Opportunity]) -> List[List]:
                """Convert opportunities to table rows for Gradio Dataframe.
                
                Args:
                    opps: List of opportunity objects to display
                    
                Returns:
                    List of rows, each containing [description, price, estimate, discount, url]
                """
                return [[opp.deal.product_description, f"${opp.deal.price:.2f}", f"${opp.estimate:.2f}", f"${opp.discount:.2f}", opp.deal.url] for opp in opps]
        
            def start() -> List[List]:
                """Initialize the agent framework and load existing opportunities.
                
                Called when the UI first loads. Initializes all agents and retrieves
                any previously discovered deals from memory.
                
                Returns:
                    Table data for the opportunities dataframe
                """
                self.agent_framework = DealAgentFramework()
                self.agent_framework.init_agents_as_needed()
                opportunities = self.agent_framework.memory
                table = table_for(opportunities)
                return table
        
            def go() -> List[List]:
                """Run the agent framework to discover new deals.
                
                Called periodically by the timer. Executes the full agent workflow:
                1. Scanner agent fetches and filters deals
                2. Ensemble agent prices each deal
                3. Planning agent identifies best opportunities
                
                Returns:
                    Updated table data with new opportunities
                """
                if self.agent_framework is None:
                    return []
                self.agent_framework.run()
                new_opportunities = self.agent_framework.memory
                table = table_for(new_opportunities)
                return table
        
            def do_select(selected_index: gr.SelectData):
                """Handle row selection to send alert for selected opportunity.
                
                Args:
                    selected_index: Gradio SelectData event containing the selected row index
                """
                if self.agent_framework is None:
                    return
                opportunities = self.agent_framework.memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                # Send alert via configured messaging service (SMS or push notification)
                if self.agent_framework.planner and self.agent_framework.planner.messenger:
                    self.agent_framework.planner.messenger.alert(opportunity)
        
            # Header
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>')
            
            # Description
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Autonomous agent framework that finds online deals, collaborating with a proprietary fine-tuned LLM deployed on Modal, and a RAG pipeline with a frontier model and Chroma.</div>')
            
            # Subheading
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')
            
            # Opportunities table
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Description", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=[4, 1, 1, 1, 2],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
        
            # Load initial data when UI starts
            ui.load(start, inputs=[], outputs=[opportunities_dataframe])

            # Refresh data every 60 seconds
            timer = gr.Timer(value=60)
            timer.tick(go, inputs=[], outputs=[opportunities_dataframe])

            # Handle row selection to send alerts
            opportunities_dataframe.select(do_select)
        
        ui.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    """Entry point: Create and launch the Gradio application."""
    App().run()
    