# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def log_async(logger):
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
            start = time.time()
            ret = await func(*args, **kwargs)
            logger.debug(
                f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} ms"
            )
            return ret

        return wrapped

    return decorator


def log_sync(logger):
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
            start = time.time()
            ret = func(*args, **kwargs)
            logger.debug(
                f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} ms"
            )
            return ret

        return wrapped

    return decorator
