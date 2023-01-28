"""Source:
https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python/36926134
"""

from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import multiprocessing

_DEFAULT_POOL = ThreadPoolExecutor(multiprocessing.cpu_count() - 2)


def thread_pool(f, executor=None):
    @wraps(f)
    def wrap(*args, **kwargs):
        return (executor or _DEFAULT_POOL).submit(f, *args, **kwargs)

    return wrap
