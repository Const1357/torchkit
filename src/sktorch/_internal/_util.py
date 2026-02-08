import time

def _time_now():
    """
    DD-MM-YYYY--HH-MM-SS
    """
    return time.strftime("%d-%m-%Y--%H-%M-%S", time.localtime())

def _tag(s: str | None) -> str:
    if s is None:
        return _time_now()
    return s