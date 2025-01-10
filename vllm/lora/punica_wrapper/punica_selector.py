from vllm.logger import init_logger
from vllm.platforms import current_platform

from .punica_base import PunicaWrapperBase
import vllm.envs as envs

logger = init_logger(__name__)


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    if current_platform.is_cuda_alike():
        # Lazy import to avoid ImportError
        if envs.VLLM_USE_V1:
            from vllm.lora.punica_wrapper.v1_punica_gpu import V1PunicaWrapperGPU
            logger.info_once("Using V1PunicaWrapperGPU.")
            return V1PunicaWrapperGPU(*args, **kwargs)
        else:
            from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
            logger.info_once("Using PunicaWrapperGPU.")
            return PunicaWrapperGPU(*args, **kwargs)
    elif current_platform.is_hpu():
        # Lazy import to avoid ImportError
        from vllm.lora.punica_wrapper.punica_hpu import PunicaWrapperHPU
        logger.info_once("Using PunicaWrapperHPU.")
        return PunicaWrapperHPU(*args, **kwargs)
    else:
        raise NotImplementedError
