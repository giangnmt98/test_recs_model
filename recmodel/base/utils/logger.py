import logging as logging_logger
import sys

from loguru import logger as loguru_logger

from recmodel.base.utils.singleton import SingletonMeta


class Logger(metaclass=SingletonMeta):
    def __init__(self, is_loguru=True, update=False):
        self.loguru_logger = loguru_logger
        self.loguru_logger.remove()
        self.loguru_logger.add(
            sys.stderr,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}"
            "</green> | <level>{level: <8}</level>| <cyan>{name}"
            "</cyan>:<cyan>{function}</cyan>:<cyan>{line"
            "}</cyan>- <level>{message}</level>",
            level="INFO",
        )
        self.logging_logger = logging_logger
        self.logging_logger.basicConfig(level=logging_logger.INFO)

        self.is_loguru = is_loguru

    def get_logger(self):
        if self.is_loguru:
            return self.loguru_logger
        else:
            return self.logging_logger


logger = Logger().get_logger()


def lgbm_format_eval_result(value, show_stdv=True):
    if len(value) == 4:
        return "{0}'s {1}:{2:.5f}".format(value[0], value[1], value[2])
    elif len(value) == 5:
        if show_stdv:
            return "{0}'s {1}:{2:.5f}+{3:.5f}".format(
                value[0], value[1], value[2], value[4]
            )
        return "{0}'s {1}:{2:.5f}".format(value[0], value[1], value[2])
    raise ValueError("Wrong metric value")


def xgb_format_eval_result(value, show_stdv=True):
    if len(value) == 2:
        return "{0}:{1:.5f}".format(value[0], value[1])
    if len(value) == 3:
        if show_stdv:
            return "{0}:{1:.5f}+{2:.5f}".format(value[0], value[1], value[2])
        return "{0}:{1:.5f}".format(value[0], value[1])
    raise ValueError("wrong metric value")


def boosting_logger_eval(model: str, period: int = 1, show_stdv: bool = True):
    """
    Create a callback that prints the evaluation results for xgboost and lgbm.

    Args:
        model (str): model name, lgbm or xgb
        period (int, optional):
            The period to print the evaluation results.
            Defaults to 1
        show_stdv (bool, optional):
            Whether to show stdv (if provided).
            Default to True

    Returns:
        function: The callback that prints the evaluation results
            every ``period`` iteration(s).
    """
    if model == "lgbm":
        format_result_func = lgbm_format_eval_result
    elif model == "xgb":
        format_result_func = xgb_format_eval_result
    else:
        raise TypeError(f"Unexpected model {model}. Only accept lgbm or xgb model.")

    def _callback(env):
        if (
            period > 0
            and env.evaluation_result_list
            and (env.iteration + 1) % period == 0
        ):
            result = "\t".join(
                [format_result_func(x, show_stdv) for x in env.evaluation_result_list]
            )
            logger.info(f"[{env.iteration}] {result}")

    return _callback
