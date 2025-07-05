import logging
import os
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure the logger
logger = logging.getLogger("research_agent")
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler(logs_dir / "research_agent.log")
file_handler.setLevel(logging.INFO)

# Create a console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Only errors to console

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger():
    """Returns the configured logger instance."""
    return logger 