import subprocess
from pathlib import Path
from typing import *
from loguru import logger

from .types import ImageExtractor, ImageExtractorConfig



class KeyFrameExtractor:

    def process(self, video_filepath: str, output_directory_filepath: str, config: ImageExtractorConfig) -> None:
        output_path = Path(output_directory_filepath)
        output_path.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        logger.info(f"Starting keyframe extraction for video: {video_filepath} to {output_path}")

        # ffmpeg command to extract keyframes
        # -i: input file
        # -vf "select=eq(pict_type\\,I)": video filter to select only I-frames (keyframes)
        # -vsync vfr: variable frame rate, important to avoid duplicating frames
        # -q:v 2: quality for video output (images), 2 is good quality
        # frame-%04d.png: output pattern for image files (e.g., frame-0001.png)
        command = [
            "ffmpeg",
            "-i", str(video_filepath),
            "-vf", "select=eq(pict_type\\,I)",
            "-vsync", "vfr",
            "-q:v", "2",
            str(output_path / "frame-%04d.png")
        ]

        try:
            # Run the ffmpeg command
            # check=True will raise a CalledProcessError if the command returns a non-zero exit code
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"Keyframe extraction completed successfully for {video_filepath}.")
            logger.debug(f"FFmpeg stdout:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"FFmpeg stderr:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Keyframe extraction failed for {video_filepath}. Error: {e}")
            logger.error(f"FFmpeg stdout:\n{e.stdout}")
            logger.error(f"FFmpeg stderr:\n{e.stderr}")
            raise # Re-raise the exception to indicate failure
        except FileNotFoundError:
            logger.error("FFmpeg command not found. Please ensure FFmpeg is installed and available in your system's PATH.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during keyframe extraction: {e}")
            raise
