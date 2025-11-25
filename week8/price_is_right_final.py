"""
The Price is Right - Gradio Web Application for Deal Hunting.

This module provides a web-based interface for the autonomous deal hunting
agent framework. It displays real-time logs, deal opportunities, and a 3D
visualization of the product vector space.
"""

import logging
import queue
import threading
import time
from typing import List, Tuple, Generator

import gradio as gr
import plotly.graph_objects as go

from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity
from log_utils import reformat


class QueueHandler(logging.Handler):
    """
    Custom logging handler that puts log records into a queue.
    
    This handler enables asynchronous log processing by placing formatted
    log messages into a thread-safe queue that can be consumed by the UI.
    
    Attributes:
        log_queue: Thread-safe queue for storing log messages
    """
    
    def __init__(self, log_queue: queue.Queue) -> None:
        """
        Initialize the queue handler.
        
        Args:
            log_queue: Queue to receive formatted log messages
        """
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by placing it in the queue.
        
        Args:
            record: LogRecord to be formatted and queued
        """
        self.log_queue.put(self.format(record))

def html_for(log_data: List[str]) -> str:
    """
    Format log data as HTML for display in the Gradio interface.
    
    Takes the most recent log entries and formats them into a scrollable
    HTML div with dark theme styling.
    
    Args:
        log_data: List of formatted log message strings
        
    Returns:
        HTML string with styled log output (last 18 entries)
    """
    # Take only the most recent 18 log entries to prevent overflow
    output = '<br>'.join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """

def setup_logging(log_queue: queue.Queue) -> None:
    """
    Configure logging to use a queue-based handler.
    
    Sets up a custom QueueHandler that routes all log messages to a queue
    for asynchronous processing by the UI thread.
    
    Args:
        log_queue: Queue to receive formatted log messages
    """
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
                

class App:
    """
    Main application class for The Price is Right web interface.
    
    Manages the Gradio UI, agent framework initialization, and coordinates
    between the deal hunting agents and the web interface.
    
    Attributes:
        agent_framework: Singleton instance of DealAgentFramework
    """

    def __init__(self) -> None:
        """
        Initialize the application.
        
        Sets up the app with a lazy-loaded agent framework that will be
        initialized on first use.
        """
        self.agent_framework: DealAgentFramework | None = None

    def get_agent_framework(self) -> DealAgentFramework:
        """
        Get or create the DealAgentFramework singleton.
        
        Lazily initializes the agent framework on first call to avoid
        expensive initialization during app startup.
        
        Returns:
            Initialized DealAgentFramework instance
        """
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
            self.agent_framework.init_agents_as_needed()
        return self.agent_framework

    def run(self) -> None:
        """
        Launch the Gradio web interface.
        
        Creates and configures the Gradio UI with:
        - Real-time log display
        - Deal opportunities table
        - 3D vector space visualization
        - Auto-refresh functionality
        """
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            # State to track log messages across updates
            log_data = gr.State([])
            
            def table_for(opps: List[Opportunity]) -> List[List[str]]:
                """
                Convert opportunities to table format for Gradio display.
                
                Args:
                    opps: List of Opportunity objects
                    
                Returns:
                    List of rows, each containing:
                    [description, price, estimate, discount, url]
                """
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url
                    ]
                    for opp in opps
                ]

            def update_output(
                log_data: List[str],
                log_queue: queue.Queue,
                result_queue: queue.Queue
            ) -> Generator[Tuple[List[str], str, List[List[str]]], None, None]:
                """
                Stream log updates and results to the UI.
                
                Continuously polls the log and result queues, yielding updates
                for the Gradio interface. Exits when final result is received.
                
                Args:
                    log_data: Accumulated log messages
                    log_queue: Queue containing new log messages
                    result_queue: Queue that will receive final results
                    
                Yields:
                    Tuple of (log_data, html_logs, table_data)
                """
                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None
                
                while True:
                    try:
                        # Check for new log messages
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield log_data, html_for(log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            # Check if agent run has completed
                            final_result = result_queue.get_nowait()
                            yield log_data, html_for(log_data), final_result or initial_result
                        except queue.Empty:
                            # Exit if we have final result, otherwise wait
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def get_initial_plot() -> go.Figure:
                """
                Create a placeholder plot while loading data.
                
                Returns:
                    Empty Plotly figure with loading message
                """
                fig = go.Figure()
                fig.update_layout(
                    title='Loading vector DB...',
                    height=400,
                )
                return fig

            def get_plot() -> go.Figure:
                """
                Generate 3D visualization of product vector embeddings.
                
                Uses t-SNE dimensionality reduction to visualize product embeddings
                from the ChromaDB vector store in 3D space. Products are colored
                by category.
                
                Returns:
                    Plotly 3D scatter plot figure (or placeholder if no data available)
                """
                try:
                    documents, vectors, colors = DealAgentFramework.get_plot_data(
                        max_datapoints=1000
                    )
                    
                    # Check if we have enough data for visualization
                    if len(vectors) == 0:
                        fig = go.Figure()
                        fig.update_layout(
                            title='No vector data available yet. Run the agents to populate the database.',
                            height=400,
                        )
                        return fig
                    
                    # Create the 3D scatter plot
                    fig = go.Figure(data=[go.Scatter3d(
                        x=vectors[:, 0],
                        y=vectors[:, 1],
                        z=vectors[:, 2],
                        mode='markers',
                        marker=dict(size=2, color=colors, opacity=0.7),
                    )])
                    
                    # Configure 3D scene with custom aspect ratio and camera
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='x',
                            yaxis_title='y',
                            zaxis_title='z',
                            aspectmode='manual',
                            aspectratio=dict(x=2.2, y=2.2, z=1),
                            camera=dict(
                                eye=dict(x=1.6, y=1.6, z=0.8)
                            )
                        ),
                        height=400,
                        margin=dict(r=5, b=1, l=5, t=2)
                    )

                    return fig
                except Exception as e:
                    # Handle any errors during plot generation
                    fig = go.Figure()
                    fig.update_layout(
                        title=f'Error generating plot: {str(e)}',
                        height=400,
                    )
                    return fig
        
            def do_run() -> List[List[str]]:
                """
                Execute the deal hunting agent workflow.
                
                Runs the Planning Agent to scan for deals, evaluate them,
                and send alerts for good opportunities.
                
                Returns:
                    Table-formatted list of all opportunities found
                """
                new_opportunities = self.get_agent_framework().run()
                table = table_for(new_opportunities)
                return table

            def run_with_logging(
                initial_log_data: List[str]
            ) -> Generator[Tuple[List[str], str, List[List[str]]], None, None]:
                """
                Run deal hunting with asynchronous logging.
                
                Executes the agent workflow in a background thread while
                streaming log updates to the UI in real-time.
                
                Args:
                    initial_log_data: Existing log messages to preserve
                    
                Yields:
                    Tuple of (log_data, html_logs, table_data)
                """
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)
                
                def worker() -> None:
                    """Worker thread that runs the agents and queues results."""
                    result = do_run()
                    result_queue.put(result)
                
                # Start agent workflow in background thread
                thread = threading.Thread(target=worker)
                thread.start()
                
                # Stream updates to UI
                for log_data, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, output, final_result

            def do_select(selected_index: gr.SelectData) -> None:
                """
                Handle user selection of a deal from the table.
                
                When a user clicks on a row in the opportunities table,
                re-send the alert for that specific deal.
                
                Args:
                    selected_index: Gradio SelectData containing row/column info
                """
                opportunities = self.get_agent_framework().memory
                row = selected_index.index[0]
                opportunity = opportunities[row]
                
                # Re-send alert for selected opportunity
                framework = self.get_agent_framework()
                if framework.planner and framework.planner.messenger:
                    framework.planner.messenger.alert(opportunity)
        
            # Header section
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:24px">'
                    '<strong>The Price is Right</strong> - '
                    'Autonomous Agent Framework that hunts for deals'
                    '</div>'
                )
            
            # Description section
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:14px">'
                    'A proprietary fine-tuned LLM deployed on Modal and a RAG '
                    'pipeline with a frontier model collaborate to send push '
                    'notifications with great online deals.'
                    '</div>'
                )
            
            # Opportunities table
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Deals found so far", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
            
            # Logs and visualization panels
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    plot = gr.Plot(value=get_plot(), show_label=False)
        
            # Run on initial page load
            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe]
            )

            # Auto-refresh every 5 minutes (300 seconds)
            timer = gr.Timer(value=300, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe]
            )

            # Enable row selection to re-send alerts
            opportunities_dataframe.select(do_select)
        
        # Launch the web interface
        ui.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    """Entry point for the application."""
    App().run()
    