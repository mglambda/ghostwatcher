# types.py
# types for the ghostwatcher project

from typing import *
from pydantic import BaseModel, Field


class ImageExtractorConfig(BaseModel):
    """Config object containing parameters for various extraction functions.
    Not all of the options may be used by all of the functions"""

    use_keyframes: bool = Field(
        default = True,
        description = "Use keyframes or iframes if available in the video."
    )

    min_interval: Optional[float] = Field(
        default = None,
        description = "Minimum interval in seconds between images."
    )

    max_interval: Optional[float] = Field(
        default = None,
        description = "Maximum interval in seconds between images."
    )

    # more options to come in the future
    
class ImageExtractor(Protocol):
    """Interface for video image extractor functions."""

    def process(self, video_filepath: str, output_directory_filepath: str, config: ImageExtractorConfig) -> None:
        pass

