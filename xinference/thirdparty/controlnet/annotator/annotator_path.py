import logging
import os

from ....constants import XINFERENCE_CACHE_DIR

logger = logging.getLogger(__name__)

clip_vision_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "clip_vision"
)
# clip vision is always inside controlnet "extensions\sd-webui-controlnet"
# and any problem can be solved by removing controlnet and reinstall

models_path = os.path.join(XINFERENCE_CACHE_DIR, "controlnet")
os.makedirs(models_path, exist_ok=True)
logger.info(f"ControlNet preprocessor location: {models_path}")
# Make sure that the default location is inside controlnet "extensions\sd-webui-controlnet"
# so that any problem can be solved by removing controlnet and reinstall
# if users do not change configs on their own (otherwise users will know what is wrong)
