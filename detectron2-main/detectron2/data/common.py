# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import itertools
import logging
import numpy as np
import pickle
import random
from typing import Callable, Union
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "ToIterableDataset"]

logger = logging.getLogger(__name__)


# copied from: https://docs.python.org/3/library/itertools.html#recipes
def _roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def _shard_iterator_dataloader_worker(iterable, chunk_size=1):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        # worker0: 0, 1, ..., chunk_size-1, num_workers*chunk_size, num_workers*chunk_size+1, ...
        # worker1: chunk_size, chunk_size+1, ...
        # worker2: 2*chunk_size, 2*chunk_size+1, ...
        # ...
        yield from _roundrobin(
            *[
                itertools.islice(
                    iterable,
                    worker_info.id * chunk_size + chunk_i,
                    None,
                    worker_info.num_workers * chunk_size,
                )
                for chunk_i in range(chunk_size)
            ]
        )


class _MapIterableDataset(data.IterableDataset):
    """
    Map a function over elements in an IterableDataset.

    Similar to pytorch's MapIterDataPipe, but support filtering when map_func
    returns None.

    This class is not public-facing. Will be called by `MapDataset`.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for x in map(self._map_func, self._dataset):
            if x is not None:
                yield x


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset, map_func):
        """
        Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next.
        """
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __new__(cls, dataset, map_func):
        is_iterable = isinstance(dataset, data.IterableDataset)
        if is_iterable:
            return _MapIterableDataset(dataset, map_func)
        else:
            return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class _TorchSerializedList:
    """
    A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works.
    """

    def __init__(self, lst: list):
        self._lst = lst

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        logger.info(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(self._lst)
            )
        )
        self._lst = [_serialize(x) for x in self._lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(self._lst))
        logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())

        # @lint-ignore PYTHONPICKLEISBAD
        return pickle.loads(bytes)


_DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD = _TorchSerializedList


@contextlib.contextmanager
def set_default_dataset_from_list_serialize_method(new):
    """
    Context manager for using custom serialize function when creating DatasetFromList
    """

    global _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD
    orig = _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD
    _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD = new
    yield
    _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD = orig


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(
        self,
        lst: list,
        copy: bool = True,
        serialize: Union[bool, Callable] = True,
    ):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool or callable): whether to serialize the stroage to other
                backend. If `True`, the default serialize method will be used, if given
                a callable, the callable will be used as serialize method.
        """
        self._lst = lst
        self._copy = copy
        if not isinstance(serialize, (bool, Callable)):
            raise TypeError(f"Unsupported type for argument `serailzie`: {serialize}")
        self._serialize = serialize is not False

        if self._serialize:
            serialize_method = (
                serialize
                if isinstance(serialize, Callable)
                else _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD
            )
            logger.info(f"Serializing the dataset using: {serialize_method}")
            self._lst = serialize_method(self._lst)

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        if self._copy and not self._serialize:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        sampler: Sampler,
        shard_sampler: bool = True,
        shard_chunk_size: int = 1,
    ):
        """
        Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
            shard_chunk_size: when sharding the sampler, each worker will
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler
        self.shard_chunk_size = shard_chunk_size

    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = _shard_iterator_dataloader_worker(self.sampler, self.shard_chunk_size)
        for idx in sampler:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler)


import torch.utils.data as data

class AspectRatioGroupedDataset(data.IterableDataset):
    """
    将具有相似宽高比的图像分成一组进行批次处理。
    在这个实现中，宽高比小于（或大于）1的图像将被分在一组。
    这样可以提高训练速度，因为将这些图像组合成批次时
    需要的填充较少。

    假设基础数据集生成的每个数据项都是一个包含 "width" 和 "height" 键的字典。
    该类将生成一个原始字典列表，长度等于 batch_size，
    所有字典具有相似的宽高比。
    """

    def __init__(self, dataset, batch_size):
        """
        初始化方法。

        参数:
            dataset: 一个可迭代的数据集。每个元素必须是一个字典，字典中包含
                "width" 和 "height" 键，用于根据宽高比分组数据。
            batch_size (int): 每个批次的大小。
        """
        self.dataset = dataset  # 保存传入的数据集
        self.batch_size = batch_size  # 保存批次大小
        self._buckets = [[] for _ in range(2)]  # 初始化两个桶，用于存放不同宽高比的图像
        # 硬编码为两个宽高比组：宽度 > 高度 和 宽度 < 高度。
        # 可以增加对更多宽高比组的支持，但在此实现中不需要

    def __iter__(self):
        """
        迭代方法。
        遍历基础数据集，根据图像的宽高比将它们分配到不同的桶中。
        当某个桶的图像数量达到 batch_size 时，将该批次的图像返回。
        """
        for d in self.dataset:  # 遍历数据集中的每个数据项
            w, h = d["width"], d["height"]  # 获取图像的宽度和高度
            bucket_id = 0 if w > h else 1  # 根据宽高比选择桶：w > h 选择第一个桶，否则选择第二个桶
            bucket = self._buckets[bucket_id]  # 获取对应的桶
            bucket.append(d)  # 将当前图像添加到对应的桶中
            if len(bucket) == self.batch_size:  # 如果桶中的图像数量达到了 batch_size
                data = bucket[:]  # 创建桶的副本，用于生成批次
                del bucket[:]  # 清空桶，为接收新的图像做准备
                yield data  # 返回当前批次的图像

