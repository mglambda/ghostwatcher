import argparse
import sys
import tempfile
import sys
from pathlib import Path
from copy import deepcopy
from typing import Optional, assert_never

import ghostbox
from ghostbox import Ghostbox

from loguru import logger

from .types import *
from .extraction import KeyFrameExtractor


def describe_frames(
    frame_collection: FrameCollection, llm_config: LLMConfig, prog: Program
) -> FrameCollection:
    """Fills in the description for all frames in the frame collection using multimodal AI, based on an LLM configuration."""
    # we don't alter the old collection, but construct a new one
    # this isn't strictly necessary right now, but may allow us to e.g. look up old descriptions in the future, so don't refactor this
    new_frame_collection = deepcopy(frame_collection)

    # we now modify the new collection
    frame_count = len(frame_collection.frames)
    for i in range(frame_count):
        frame = new_frame_collection.frames[i]
        if frame.description:
            logger.info(f"Frame {i+1} of {frame_count} already described. Skipping.")
            continue

        logger.info(f"Generating description for frame {i+1} of {frame_count}.")
        prompt = ""
        b = llm_config.batch_size
        batch_frames = frame_collection.frames[max(0, i + 1 - b) : i + 1]
        logger.debug(f"Got batch frames of size {len(batch_frames)}.")
        context_images = [f.filepath for f in batch_frames]
        if b > 1:
            prompt += llm_config.batch_description_prompt_part + "\n"
            prompt += (
                "Here is some timing information about the images from the video:\n"
            )
            for k, batch_frame in enumerate(batch_frames):
                prompt += f" - Image {k} occurs at {batch_frame.seek_pos} seconds into the video.\n"
        else:
            prompt += f"The image is a still frame from a video, occurring at {batch_frames[-1].seek_pos} seconds into it.\n"
            prompt += llm_config.description_prompt
        try:
            prog.box.clear_history()
            with prog.box.images(context_images):
                frame.description = prog.box.text(
                    prompt,
                )
                if error_str := prog.box.get_last_error():
                    logger.error(f"ghostbox: {error_str}")
                print(f"# debug {i}:\n{frame.description}", file=sys.stderr, flush=True)
        except Exception as e:
            logger.error(f"Failed to describe frame {i}: {e}")

        # save intermediate progress
        new_frame_collection.save(prog.get_frame_collection_path())
        
    return new_frame_collection

def caption_frames(frame_collection: FrameCollection, llm_config: LLMConfig, prog: Program) -> VideoCaptions:
    """Generates short time-sensitive video captions based on a frame collection."""
    video_captions = VideoCaptions(video_filepath=frame_collection.video_filepath, captions=[])
    
    # Load existing captions if available (no force flag for now)
    captions_path = prog.get_captions_path()
    if captions_path.is_file():
        try:
            existing_captions = VideoCaptions.load(captions_path)
            if existing_captions.video_filepath == frame_collection.video_filepath:
                video_captions = existing_captions
                logger.info(f"Loaded existing captions from {captions_path}.")
            else:
                logger.warning(f"Existing captions file {captions_path} belongs to a different video. Starting fresh.")
        except Exception as e:
            logger.error(f"Failed to load existing captions from {captions_path}: {e}. Starting fresh.")

    num_frames = len(frame_collection.frames)
    batch_size = llm_config.caption_batch_size

    for i in range(0, num_frames, batch_size):
        batch_frames = frame_collection.frames[i : i + batch_size]
        
        if not batch_frames:
            continue

        logger.info(f"Generating captions for batch {i // batch_size + 1} (frames {i+1} to {min(i + batch_size, num_frames)}).")

        batch_preamble_str = ""
        for k, frame in enumerate(batch_frames):
            # Only include frames that have a description
            if frame.description:
                batch_preamble_str += f"Frame {k+1} (at {frame.seek_pos:.2f}s): {frame.description}\n"
            else:
                logger.warning(f"Frame {k+1} in batch (at {frame.seek_pos:.2f}s) has no description. Skipping in preamble.")

        if not batch_preamble_str:
            logger.warning(f"No described frames in batch {i // batch_size + 1}. Skipping LLM call.")
            continue

        prompt = batch_preamble_str + "\n" + llm_config.caption_prompt
        logger.debug(f"Prompt for batch {i // batch_size + 1}:\n{prompt}")

        try:
            prog.box.clear_history() # Clear history for each batch to avoid context overflow
            batch_video_captions = prog.box.new(VideoCaptions, prompt)
            
            if error_str := prog.box.get_last_error():
                logger.error(f"ghostbox (captioning): {error_str}")

            video_captions.captions.extend(batch_video_captions.captions)
            logger.debug(f"Generated {len(batch_video_captions.captions)} captions for batch {i // batch_size + 1}.")
            for caption in batch_video_captions.captions:
                logger.debug(f"  - Caption at {caption.seek_pos:.2f}s: {caption.content}")

        except Exception as e:
            logger.error(f"Failed to generate captions for batch {i // batch_size + 1}: {e}")
            # Continue to next batch even if one fails, to save partial progress

        # Save intermediate progress after each batch
        video_captions.save(captions_path)
        logger.info(f"Saved intermediate captions to {captions_path}.")
        
    return video_captions
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
    parser = argparse.ArgumentParser(
        description="Multimodal AI powered video description and commentary."
    )

    # Positional argument
    parser.add_argument(
        "video_file", type=Path, help="Path to the video file to describe/comment."
    )

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        default=Path("./output"),
        help="Directory where output descriptions will be stored.",
    )

    parser.add_argument(
        "-w",
        "--work-directory",
        type=Path,
        help="Temporary work directory for storing intermediate files (e.g., images). If omitted, a temporary directory will be created.",
    )
    parser.add_argument(
        "-x",
        "--extraction-strategy",
        type=ExtractionStrategy,
        choices=list(ExtractionStrategy),
        default=ExtractionStrategy.keyframes,
        help="Strategy for extracting frames from the video.",
    )

    parser.add_argument(
        "-c",
        "--character-folder",
        type=str,
        default="ghost",
        help="Character folders contain the system message and configuration options for ghostbox and the LLM backend.",
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
        help="The HTTP endpoint to query. The default is compatible with llama.cpp's llama-server out of the box. When using the generic backend, it will expect an OpenAI compatible API at this address.",
    )

    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="Enable verbose/debug output.",
    )
    parser.add_argument(
        "--log-timestamps",
        action=argparse.BooleanOptionalAction,
        help="Enable timestamps in log output.",
    )
    
    parser.add_argument(
        "--force-frame-extraction",
        action=argparse.BooleanOptionalAction,
        help="Force frame extraction even if the output directory is not empty. This implies --force-frame-description.",
    )


    parser.add_argument(
        "--force-frame-description",
        action=argparse.BooleanOptionalAction,
        help="Force regeneration of frame collection descriptions.",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LLMConfig.model_fields["batch_size"].default,
        help="How many images to describe in one batch. A higher batch size gives better results because the LLM will have more images in context simultaneously, but also requires substantially more memory and processing time.",
    )
    parser.add_argument(
        "--description-prompt",
        type=str,
        default=LLMConfig.model_fields["description_prompt"].default,
        help="Prompt used for basic image descriptions.",
    )

    args = parser.parse_args()
    # ensure logical consistency with the forcing
    if args.force_frame_extraction:
        args.force_frame_description = True

    # Setup logging
    setup_logging(args.debug, args.log_timestamps)

    # Handle work directory
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None  # type: ignore
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

    logger.info(f"Setting up ghostbox with {args.backend} backend.")
    box = Ghostbox(
        character_folder=args.character_folder,
        backend=args.backend,
        endpoint=args.endpoint,
        stdout=False,
        stderr=args.debug,
    )

    # construct program object - this finishes intialization
    prog = Program(output_dir=args.output_directory, work_dir=work_dir, box=box)

    # 1. step: extraction of images
    try:
        extractor: ImageExtractor
        extraction_output_dir = prog.get_extraction_output_dir()

        # Check if extraction can be skipped
        skip_extraction = False
        if any(extraction_output_dir.iterdir()):
            if args.force_frame_extraction:
                logger.info(
                    f"Force frame extraction: clearing existing frames in '{extraction_output_dir}'."
                )
                for f in extraction_output_dir.iterdir():
                    if f.is_file():
                        f.unlink()
            else:
                logger.info(
                    f"Skipping frame extraction: '{extraction_output_dir}' is not empty. Use --force-frame-extraction to override."
                )
                skip_extraction = True

        if not skip_extraction:
            match args.extraction_strategy:
                case ExtractionStrategy.keyframes:
                    extractor = KeyFrameExtractor()
                    extractor.process(
                        video_filepath=args.video_file,
                        output_path=extraction_output_dir,
                        config=ImageExtractorConfig(use_keyframes=True),
                    )
                case ExtractionStrategy.interval:
                    # TODO: Implement IntervalExtractor
                    logger.warning("Interval extraction not yet implemented.")
                    pass
                case _ as unreachable:
                    assert_never(unreachable)

    except Exception as e:
        logger.error(f"Failed to extract images. Reason: {e}")
        sys.exit(1)
        
    try:
        # constructa frame collection, either from extracted images or from a previously saved collection
        frame_collection_path = prog.get_frame_collection_path()
        if frame_collection_path.is_file() and not(args.force_frame_extraction):
            logger.info(f"Continuing with existing frame collection.")
            frame_collection = FrameCollection.load(frame_collection_path)
        else:
            # happens on --force-frame-extraction. if we reextracted frames it makes no sense to keep the old descriptions
            logger.info(f"Constructing new frame collection.")
            frame_collection = FrameCollection.from_directory(
                extraction_output_dir, str(args.video_file)
            )
    except Exception as e:
        logger.error("Could not construct frame collection. {e}")
        sys.exit(1)
        
    logger.info(
        f"Extracted {len(frame_collection.frames)} images from {frame_collection.video_filepath}."
    )

    # 2. step: description generation
    if args.force_frame_description:
        logger.info(f"Ignoring previously generated frame descriptions (from --force-frame-description).")
        # we do this by nulling them all
        for frame in frame_collection.frames:
            frame.description = None
        
    
    llm_config = LLMConfig(
        batch_size=args.batch_size, description_prompt=args.description_prompt
    )
    logger.debug(f"LLM Configuration: {llm_config.model_dump_json(indent=2)}")

    described_frame_collection = describe_frames(frame_collection, llm_config, prog)
    logger.info("Finished frame descriptions.")

    # temporary for development - just output the descriptions
    print(f"=== OUTPUT ===")
    for i, frame in enumerate(described_frame_collection.frames):
        print(f"# {i} at {frame.seek_pos}s:")
        print(frame.description)

    # 3. step - caption generation
    logger.info(f"Generating captions.")
    video_captions = caption_frames(described_frame_collection, llm_config, prog)

    # output captions - again temporary during development
    print(f"=== captions ==")
    for i, caption in enumerate(video_captions.captions):
        print(f"#{i} at {caption.seek_pos}")
        print(f"{caption.content}")


    
    # Explicitly clean up temporary directory if one was created
    if temp_dir_obj:
        temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
