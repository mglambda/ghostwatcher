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
        # -vf "select=eq(pict_type\\,I),showinfo": video filter to select only I-frames (keyframes) and show metadata
        # -vsync vfr: variable frame rate, important to avoid duplicating frames
        # -q:v 2: quality for video output (images), 2 is good quality
        # frame-%04d.png: output pattern for image files (e.g., frame-0001.png)
        command = [
            "ffmpeg",
            "-i", str(video_filepath),
            "-vf", "select=eq(pict_type\\,I),showinfo",
            "-vsync", "vfr",
            "-q:v", "2",
            str(output_path / "frame-%04d.png")
        ]

        try:
            # Run the ffmpeg command
            # check=True will raise a CalledProcessError if the command returns a non-zero exit code
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"Keyframe extraction completed successfully for {video_filepath}.")
            
            # Parse timestamps from showinfo output in stderr
            import re
            timestamps = []
            # Look for lines like: [Parsed_showinfo_0 @ 0x...] n:   0 pts:      0 pts_time:0 ...
            for line in result.stderr.splitlines():
                if "pts_time:" in line:
                    match = re.search(r"pts_time:([\d\.]+)", line)
                    if match:
                        timestamps.append(match.group(1))
            
            # Rename files to include timestamps
            for i, ts in enumerate(timestamps):
                frame_num = i + 1
                old_filename = f"frame-{frame_num:04d}.png"
                new_filename = f"frame-{frame_num:04d}-ts-{float(ts)}.png"
                old_file = output_path / old_filename
                new_file = output_path / new_filename
                if old_file.exists():
                    old_file.rename(new_file)
                    logger.debug(f"Renamed {old_filename} to {new_filename}")

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
