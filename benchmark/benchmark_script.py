import logging
import os
import time
from typing import List

from xinference.client import Client
from xinference.model.llm import LLM_FAMILIES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.getLevelName("INFO".upper()))


# Parameters
endpoint = "http://127.0.0.1:9997"
NUM_ITER = 5
GPU_ID = 0
SKIP_MODELS: List[str] = []


def get_gpu_mem_info(gpu_id=GPU_ID):
    """
    Obtain gpu memory usage information according to gpu id, in MB.
    Returns
    -------
    total: all gpu memory
    used: currently used gpu memory
    free: available gpu memory
    """
    try:
        import pynvml
    except ImportError:
        raise ImportError("Failed to import module 'pynvml', Please make sure 'pynvml' is installed.\n")

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        logger.info(f"gpu_id {gpu_id} does not exist!")
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    Get the memory information of the current machine, in MB
    Returns
    -------
    mem_total: all memory of the current machine
    mem_free: available memory of the current machine
    mem_process_used: memory used by the current process
    """
    try:
        import psutil
    except ImportError:
        raise ImportError("Failed to import module 'psutil', Please make sure 'psutil' is installed.\n")

    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(
        psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2
    )
    return mem_total, mem_free, mem_process_used


def get_speed_for_chat_model(model, prompt, chat_history):
    t1 = time.time()
    a = model.chat(prompt, chat_history, generate_config={"max_tokens": 1024})
    t2 = time.time()
    t = t2 - t1
    completion_tokens = a["usage"]["completion_tokens"]
    speed = completion_tokens / t
    logger.info(f"text: {a}")
    logger.info(f"tokens: {completion_tokens}")
    logger.info(f"time: {t}")
    logger.info(f"speed: {speed}")
    return speed


def get_speed_for_generate_model(model, prompt):
    t1 = time.time()
    a = model.generate(prompt, generate_config={"max_tokens": 1024})
    t2 = time.time()
    t = t2 - t1
    completion_tokens = a["usage"]["completion_tokens"]
    speed = completion_tokens / t
    logger.info(f"text: {a}")
    logger.info(f"tokens: {completion_tokens}")
    logger.info(f"time: {t}")
    logger.info(f"speed: {speed}")
    return speed


def run_model(endpoint):
    client = Client(endpoint)

    for model_family in LLM_FAMILIES:
        model_name = model_family.model_name
        if model_name in SKIP_MODELS:
            continue
        for model_spec in model_family.model_specs:
            model_format = model_spec.model_format
            model_size = model_spec.model_size_in_billions
            quantizations = model_spec.quantizations
            if model_format == "ggmlv3":
                # only test 1 quantization for ggml model
                quantizations = quantizations[:1]
            for quantization in quantizations:
                logger.info(
                    f"Model: {model_name}-{model_format}-{model_size}b-{quantization}"
                )
                try:
                    model_uid = client.launch_model(
                        model_name=model_name,
                        model_format=model_format,
                        model_size_in_billions=model_size,
                        quantization=quantization,
                    )
                    logger.info(
                        f"model launch success: {model_name}-{model_format}-{model_size}b-{quantization}"
                    )
                except Exception:
                    # raise f"model launch failed: {model_name}-{model_format}-{model_size}b-{quantization}"
                    logger.info(
                        f"model launch failed: {model_name}-{model_format}-{model_size}b-{quantization}"
                    )
                    continue

                try:
                    logger.info("After launch model:")
                    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(
                        gpu_id=GPU_ID
                    )
                    logger.info(
                        f"Current GPU memory usage: Total {gpu_mem_total} MB, Used {gpu_mem_used} MB, Remaining {gpu_mem_free} MB"
                    )

                    (
                        cpu_mem_total,
                        cpu_mem_free,
                        cpu_mem_process_used,
                    ) = get_cpu_mem_info()
                    logger.info(
                        f"Current machine memory usage：Total {cpu_mem_total} MB, Used {cpu_mem_process_used} MB by the current process, Remaining {cpu_mem_free} MB"
                    )

                    model = client.get_model(model_uid)

                    if "chat" in model_family.model_ability:
                        chat_history = []
                        prompt = "What't the top 10 largest animals in the world?"

                        list_speed = []
                        for _ in range(NUM_ITER):
                            s = get_speed_for_chat_model(model, prompt, chat_history)
                            list_speed.append(s)
                        logger.info(
                            f"average speed: {sum(list_speed[1:]) / len(list_speed[1:])}"
                        )
                    else:
                        prompt = "Once upon a time, there was a very old computer."

                        list_speed = []
                        for _ in range(NUM_ITER):
                            s = get_speed_for_generate_model(model, prompt)
                            list_speed.append(s)
                        logger.info(
                            f"average speed: {sum(list_speed[1:]) / len(list_speed[1:])}"
                        )

                    logger.info("\nAfter chat:")
                    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(
                        gpu_id=GPU_ID
                    )
                    logger.info(
                        f"Current GPU memory usage: Total {gpu_mem_total} MB, Used {gpu_mem_used} MB, Remaining {gpu_mem_free} MB"
                    )

                    (
                        cpu_mem_total,
                        cpu_mem_free,
                        cpu_mem_process_used,
                    ) = get_cpu_mem_info()
                    logger.info(
                        f"Current machine memory usage：Total {cpu_mem_total} MB, Used {cpu_mem_process_used} MB by the current process, Remaining {cpu_mem_free} MB"
                    )
                except Exception:
                    logger.info(
                        f"model chat failed: {model_name}-{model_format}-{model_size}b-{quantization}"
                    )

                client.terminate_model(model_uid)
                logger.info(
                    f"\n{model_name}-{model_format}-{model_size}b-{quantization} is terminated\n\n\n"
                )


if __name__ == "__main__":
    run_model(endpoint)
