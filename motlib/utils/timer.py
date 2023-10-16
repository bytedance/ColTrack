
import torch
import math
import torch
import time
from datetime import datetime, timezone, timedelta
from termcolor import colored
from time import perf_counter
import time
from typing import Optional
from functools import wraps
import logging


__all__ = ['timestamp',
           'time_tag',
           'TimerClock',
           'Timer',
           'beijing',
           'clock',
           'time_synchronized'
           ]


def time_type_transform(allTime):
    day = 24 * 60 * 60
    hour = 60 * 60
    min = 60
    if allTime < 60:
        return "%d sec" % math.ceil(allTime)
    elif allTime > day:
        days = divmod(allTime, day)
        return "%d days, %s" % (int(days[0]), time_type_transform(days[1]))
    elif allTime > hour:
        hours = divmod(allTime, hour)
        return '%d hours, %s' % (int(hours[0]), time_type_transform(hours[1]))
    else:
        mins = divmod(allTime, min)
        return "%d mins, %d sec" % (int(mins[0]), math.ceil(mins[1]))


def timestamp(fmt='%Y-%m-%d %H:%M:%S '):
    return colored(str(get_unified_time().strftime(fmt)), 'green')


def get_unified_time(d=None, offset=8):
    if d is not None:
        utc_dt = d.utcnow()
    else:
        utc_dt = datetime.utcnow()
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=offset)))
    return bj_dt


def beijing():
    beijing_time = datetime.now() + timedelta(hours=8)
    return beijing_time.timetuple()


def time_tag():
    # Don't modify this function.
    return str(get_unified_time().strftime('%Y%m%d%H%M%S'))


class TimerClock(object):
    def __init__(self):
        self._begin_time = datetime.utcnow()
        self._state_time = self._begin_time

        self._duration_count = 0
        self._duration_time = None

    def timing(self, str_format=True):
        time_now = datetime.utcnow()
        duration = time_now - self._state_time
        self._state_time = time_now
        if str_format:
            return time_type_transform(duration.total_seconds())
        else:
            return duration

    def duration(self, num, func, *args, **kwargs):
        time_begin = datetime.utcnow()
        result = func(*args, **kwargs)
        time_end = datetime.utcnow()
        if self._duration_time is None:
            self._duration_time = time_end - time_begin
        else:
            self._duration_time += time_end - time_begin
        self._duration_count += num
        return result

    @property
    def avg_duration(self):
        duration_avg = self._duration_time / self._duration_count
        self.duration_reset()
        return duration_avg.total_seconds()

    def duration_reset(self):
        self._duration_count = 0
        self._duration_time = None


class Timer:
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the timer.
        """
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self):
        """
        Pause the timer.
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        """
        Returns:
            bool: whether the timer is currently paused
        """
        return self._paused is not None

    def resume(self):
        """
        Resume the timer.
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        """
        Returns:
            (float): the total number of seconds since the start/reset of the
                timer, excluding the time when the timer is paused.
        """
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        Returns:
            (float): the average number of seconds between every start/reset and
            pause.
        """
        return self.seconds() / self._count_start


def clock(func):
    """this is outer clock function"""
    @wraps(func) 
    def clocked(*args, **kwargs):  
        """this is inner clocked function"""
        start_time = perf_counter()
        result = func(*args, **kwargs)  
        time_cost = perf_counter() - start_time
        logger = logging.getLogger(__name__)
        logger.info(func.__name__ + " func time_cost : {} s".format(time_cost))
        return result
    return clocked 


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()