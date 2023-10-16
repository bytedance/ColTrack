#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader as torchDataLoader
import itertools
from typing import Optional
import random
import util.misc as util


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will prepend a dimension, whilst ensuring it stays the same across one mini-batch.
    """

    def __init__(self, *args, input_dimension=None, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None
        self.mosaic = mosaic

    def __iter__(self):
        self.__set_input_dim()
        for batch in super().__iter__():
            yield [(self.input_dim, idx, self.mosaic) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)
        self.epoch = 0

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )
    def set_epoch(self, epoch):
        self.epoch=epoch

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self.epoch)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py

    Note:
        This dataloader only works with :class:`lightnet.data.Dataset` based datasets.

    Example:
        >>> class CustomSet(ln.data.Dataset):
        ...     def __len__(self):
        ...         return 4
        ...     @ln.data.Dataset.resize_getitem
        ...     def __getitem__(self, index):
        ...         # Should return (image, anno) but here we return (input_dim,)
        ...         return (self.input_dim,)
        >>> dl = ln.data.DataLoader(
        ...     CustomSet((200,200)),
        ...     batch_size = 2,
        ...     collate_fn = ln.data.list_collate   # We want the data to be grouped as a list
        ... )
        >>> dl.dataset.input_dim    # Default input_dim
        (200, 200)
        >>> for d in dl:
        ...     d
        [[(200, 200), (200, 200)]]
        [[(200, 200), (200, 200)]]
        >>> dl.change_input_dim(320, random_range=None)
        (320, 320)
        >>> for d in dl:
        ...     d
        [[(320, 320), (320, 320)]]
        [[(320, 320), (320, 320)]]
        >>> dl.change_input_dim((480, 320), random_range=None)
        (480, 320)
        >>> for d in dl:
        ...     d
        [[(480, 320), (480, 320)]]
        [[(480, 320), (480, 320)]]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                sampler,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False

    def change_input_dim(self, multiple=32, random_range=(10, 19)):
        """This function will compute a new size and update it on the next mini_batch.

        Args:
            multiple (int or tuple, optional): values to multiply the randomly generated range by.
                Default **32**
            random_range (tuple, optional): This (min, max) tuple sets the range
                for the randomisation; Default **(10, 19)**

        Return:
            tuple: width, height tuple with new dimension

        Note:
            The new size is generated as follows: |br|
            First we compute a random integer inside ``[random_range]``.
            We then multiply that number with the ``multiple`` argument,
            which gives our final new input size. |br|
            If ``multiple`` is an integer we generate a square size. If you give a tuple
            of **(width, height)**, the size is computed
            as :math:`rng * multiple[0], rng * multiple[1]`.

        Note:
            You can set the ``random_range`` argument to **None** to set
            an exact size of multiply. |br|
            See the example above for how this works.
        """
        if random_range is None:
            size = 1
        else:
            size = random.randint(*random_range)

        if isinstance(multiple, int):
            size = (size * multiple, size * multiple)
        else:
            size = (size * multiple[0], size * multiple[1])

        self.batch_sampler.new_input_dim = size

        return size
    
    def random_resize(self):
        tensor = torch.LongTensor(2).cuda()
        rank = util.get_rank()
        try:
            resize_range = self.dataset.args.resize_range
        except:
            resize_range = 32

        if rank == 0:
            input_size = (800, 1440)
            random_size = (18, resize_range)
            size_factor = input_size[1] * 1.0 / input_size[0]
            size = random.randint(*random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if util.is_dist_avail_and_initialized():
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = self.change_input_dim(
            multiple=(tensor[0].item(), tensor[1].item()), random_range=None
        )
        return input_size
