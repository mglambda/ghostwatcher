import argparse
import sys
import tempfile
from pathlib import Path
from copy import deepcopy
from typing import Optional, assert_never

import ghostbox
from ghostbox import Ghostbox

from loguru import logger

from .types import *
from .extraction import KeyFrameExtractor

def describe_frames(frame_collection: FrameCollection, work_directory_filepath: str, box: Ghostbox, llm_config: LLMConfig) -> FrameCollection:
    """Fills in the description for all frames in the frame collection using multimodal AI, based on an LLM configuration."""
    # we don't alter the old collection, but construct a new one
    new_frame_collection = deepcopy(frame_collection)
    new_frame_collection.frames = []
    
    frame_count = len(frame_collection.frames)
    for i in range(frame_count):
        frame = deepcopy(frame_collection.frames[i])
        logger.info(f"Generating description for frame {i+1} of {frame_count}.")
        # FIXME: for now we are assuming batch_size == 1        
        context_images = [frame.filepath]
        try:
            box.clear_history()
            with box.images(context_images):
                frame.description = box.text(
                    llm_config.description_prompt
                )
        except Exception as e:
            logger.error(f"Failed to describe frame {i}: {e}")

        new_frame_collection.frames.append(frame)
    return new_frame_collection

    
def setup_logging(debug: bool, log_timestamps: bool) -> None:
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
        "-x", "--extraction-strategy",
        type=ExtractionStrategy,
        choices=list(ExtractionStrategy),
        default=ExtractionStrategy.keyframes,
        help="Strategy for extracting frames from the video."
    )

    parser.add_argument(
        "-c", "--character-folder",
        type=str,
        default="ghost",
        help="Character folders contain the system message and configuration options for ghostbox and the LLM backend."
    )    

    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=[backend.name for _, backend in enumerate(ghostbox.LLMBackend)],
        default="generic",
        help="Choose the backend for the multimodal AI to use. If you want to use a local AI, choose the generic or llamacpp backends. If you use another backend, such as from a cloud provider like google or deepseek, you may need to set the appropriate environment variables for credentials. Some backends allow you to set the http address to query for the API requests via the --endpoint option, such as the generic, openai, and llamacpp backends. Cloud provider backends such as google and deepseek have the endpoint built-in, and won't allow you to change it.",
    )

    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        default="http://localhost:8080",
        help="The HTTP endpoint to query. The default is compatible with llama.cpp's llama-server out of the box. When using the generic backend, it will expect an OpenAI compatible API at this address."
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
    parser.add_argument(
        "-f", "--force-frame-extraction",
        action=argparse.BooleanOptionalAction,
        help="Force frame extraction even if the output directory is not empty."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=LLMConfig.model_fields["batch_size"].default,
        help="How many images to describe in one batch. A higher batch size gives better results because the LLM will have more images in context simultaneously, but also requires substantially more memory and processing time."
    )
    parser.add_argument(
        "--description-prompt",
        type=str,
        default=LLMConfig.model_fields["description_prompt"].default,
        help="Prompt used for basic image descriptions."
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug, args.log_timestamps)

    # Handle work directory
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None # type: ignore
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
    logger.debug(f"Log timestamps enabled: {args.log_timestamps}")
    logger.debug(f"Force frame extraction: {args.force_frame_extraction}")

    # 1. step: extraction of images
    extractor: ImageExtractor
    extraction_output_dir = work_dir / "extracted_frames"
    extraction_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if extraction can be skipped
    skip_extraction = False
    if not args.force_frame_extraction and any(extraction_output_dir.iterdir()):
        logger.info(f"Skipping frame extraction: '{extraction_output_dir}' is not empty. Use --force-frame-extraction to override.")
        skip_extraction = True

    if not skip_extraction:
        match args.extraction_strategy:
            case ExtractionStrategy.keyframes:
                extractor = KeyFrameExtractor()
                extractor.process(
                    video_filepath=args.video_file,
                    output_directory_filepath=str(extraction_output_dir),
                    config=ImageExtractorConfig(use_keyframes=True)
                )
            case ExtractionStrategy.interval:
                # TODO: Implement IntervalExtractor
                logger.warning("Interval extraction not yet implemented.")
                pass
            case _ as unreachable: 
                assert_never(unreachable)

    # we have the images, now let's bundle them in a collection
    frame_collection = FrameCollection.from_directory(extraction_output_dir, str(args.video_file))
    logger.info(f"Extracted {len(frame_collection.frames)} images from {frame_collection.video_filepath}.")
    
    # 2. step: description generation
    logger.info(f"Setting up ghostbox with {args.backend} backend.")
    box = Ghostbox(
        character_folder = args.character_folder,
        backend = args.backend,
        endpoint = args.endpoint,
        stdout = False,
        stderr = args.debug
    )

    llm_config = LLMConfig(
        batch_size=args.batch_size,
        description_prompt=args.description_prompt
    )
    logger.debug(f"LLM Configuration: {llm_config.model_dump_json(indent=2)}")

    # 3. step: describe frames
    described_frame_collection = describe_frames(frame_collection, str(work_dir), box, llm_config)
    logger.info("Finished frame descriptions.")


    # temporary for development - just output the descriptions
    print(f"=== OUTPUT ===")
    for i, frame in enumerate(described_frame_collection.frames):
        print(f"# {i}:")
        print(frame.description)
        
    # Explicitly clean up temporary directory if one was created
    if temp_dir_obj:
        temp_dir_obj.cleanup()

if __name__ == "__main__":
    main()
