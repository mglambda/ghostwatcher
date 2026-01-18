import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

import ghostbox
from loguru import logger

from .types import ExtractionStrategy
from .extraction import *

def setup_logging(debug: bool, log_timestamps: bool):
    """Configures loguru logger based on debug and timestamp flags."""
    logger.remove()  # Remove default handler

    log_level = "DEBUG" if debug else "INFO"
    log_format = "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    if log_timestamps:
        log_format = "<green>{time}</green> | " + log_format

    logger.add(sys.stderr, level=log_level, format=log_format)

def main() -> None:
    """Main entry point for the ghostwatcher command-line interface."""
    parser = argparse.ArgumentParser(description="Multimodal AI powered video description and commentary.")

    # Positional argument
    parser.add_argument(
        "video_file",
        type=Path,
        help="Path to the video file to describe/comment."
    )

    # Optional arguments
    parser.add_argument(
        "-o", "--output-directory",
        type=Path,
        default=Path("./output"),
        help="Directory where output descriptions will be stored."
    )
    parser.add_argument(
        "-w", "--work-directory",
        type=Path,
        help="Temporary work directory for storing intermediate files (e.g., images). If omitted, a temporary directory will be created."
    )
    parser.add_argument(
        "-e", "--extraction-strategy",
        type=ExtractionStrategy,
        choices=list(ExtractionStrategy),
        default=ExtractionStrategy.keyframes,
        help="Strategy for extracting frames from the video."
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="Enable verbose/debug output."
    )
    parser.add_argument(
        "--log-timestamps",
        action=argparse.BooleanOptionalAction,
        help="Enable timestamps in log output."
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug, args.log_timestamps)

    # Handle work directory
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    if args.work_directory:
        work_dir = args.work_directory
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Create a temporary directory that will be cleaned up on exit
        temp_dir_obj = tempfile.TemporaryDirectory()
        work_dir = Path(temp_dir_obj.name)
        logger.info(f"Using temporary work directory: {work_dir}")

    # Ensure output directory exists
    args.output_directory.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Video file: {args.video_file}")
    logger.debug(f"Output directory: {args.output_directory}")
    logger.debug(f"Work directory: {work_dir}")
    logger.debug(f"Extraction strategy: {args.extraction_strategy}")
    logger.debug(f"Debug enabled: {args.debug}")
    logger.debug(f"Log timestamps enabled: {args.log_timestamps}")

    # 1. step: extraction of images

    # 2. step: description generation
    
    # Explicitly clean up temporary directory if one was created
    if temp_dir_obj:
        temp_dir_obj.cleanup()

if __name__ == "__main__":
    main()
