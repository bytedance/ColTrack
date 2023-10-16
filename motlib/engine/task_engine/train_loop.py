# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import weakref
import logging
from motlib.engine.utils.hooks import HookBase


__all__ = ["TrainerBase"]


class TrainerBase:
    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self):
        logger = logging.getLogger(__name__)
        logger.info("Start training")
        try:
            self.before_train()
            while self.train_flag:
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
                self.epoch += 1
        except Exception:
            logger.error("Exception during training:")
            raise
        finally:
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()
    
    def before_epoch(self):
        for h in self._hooks:
            h.before_epoch()
    
    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch()
        

    def run_epoch(self):
        while self.epoch_flag:
            self.before_step()
            self.run_step()
            self.after_step()
            self.iter += 1

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError