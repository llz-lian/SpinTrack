import torch
import torch.utils.data.dataloader
import importlib
import collections
import random
import torch.utils.data
from pytracking import TensorDict,TensorList
#from torch._six import string_classes, int_classes
string_classes = str
int_classes = int

def no_processing(data):
    return data

class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of a set of template frames and search frames, used to train the TransT model.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'template frames' and
    'search frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template frames and the search frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the search frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,allow_invisible=False,force_invisible = False,valid = None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        # if force_invisible:
        #     valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        # else:
        if allow_invisible:
            valid_ids = [i for i in range(min_id, max_id) if valid[i]]
        else:
            valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = True
        # print(dataset.name)
        if dataset.name == 'COCO':
            is_video_dataset = False

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            valid = seq_info_dict['valid']
            # visible = seq_info_dict['valid']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_template_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                     min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if extra_template_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + extra_template_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                                                              min_id=template_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              allow_invisible=True,valid = valid)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                             max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    if visible[template_frame_ids] == 0:
                        continue
                    search_frame_ids = self._sample_visible_ids(valid, min_id=template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_search_frames,allow_invisible=True,valid = valid)
                    # Increase gap until a frame is found
                    gap_increase += 5
            elif self.frame_sample_mode == 'stark':
                template_frame_ids,search_frame_ids = self.get_frame_ids_stark(visible)
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * self.num_template_frames
            search_frame_ids = [1] * self.num_search_frames

        template_frames, template_anno, meta_obj_template = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
        search_frames, search_anno, meta_obj_search = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        data = TensorDict({'template_images': template_frames,
                           'template_anno': template_anno['bbox'],
                           'search_images': search_frames,
                           'search_anno': search_anno['bbox']})
        return self.processing(data)

    def get_frame_ids_stark(self, visible):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

class OnlineTrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of a set of template frames and search frames, used to train the TransT model.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'template frames' and
    'search frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template frames and the search frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the search frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        print('online sampler')
        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = True
        if dataset.name == 'COCO':
            is_video_dataset = False

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            template_frame_ids = None
            online_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_template_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                     min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if extra_template_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + extra_template_frame_ids
                    online_frame_ids = self._sample_visible_ids(visible,min_id=template_frame_ids[0] + 10,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase)
                    search_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                                                              min_id=template_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                             max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    online_frame_ids = self._sample_visible_ids(visible,min_id=template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase)

                    search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_search_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * self.num_template_frames
            search_frame_ids = [1] * self.num_search_frames
            online_frame_ids = [1] * self.num_template_frames

        template_frames, template_anno, meta_obj_template = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
        online_frames,online_anno,meta_obj_online = dataset.get_frames(seq_id,online_frame_ids,seq_info_dict)
        search_frames, search_anno, meta_obj_search = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        data = TensorDict({'template_images': template_frames,
                           'template_anno': template_anno['bbox'],
                           'online_images': online_frames,
                           'online_anno':online_anno['bbox'],
                           'search_images': search_frames,
                           'search_anno': search_anno['bbox']})
        return self.processing(data)




def _check_use_shared_memory():
    if hasattr(torch.utils.data.dataloader, '_use_shared_memory'):
        return getattr(torch.utils.data.dataloader, '_use_shared_memory')
    collate_lib = importlib.import_module('torch.utils.data._utils.collate')
    if hasattr(collate_lib, '_use_shared_memory'):
        return getattr(collate_lib, '_use_shared_memory')
    return torch.utils.data.get_worker_info() is not None


def ltr_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


def ltr_collate_stack1(batch):
    """Puts each data field into a tensor. The tensors are stacked at dim=1 to form the batch"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 1, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 1)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))

from prefetch_generator import BackgroundGenerator

class DataLoaderX(torch.utils.data.dataloader.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class LTRLoader(DataLoaderX):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Note: The only difference with default pytorch DataLoader is that an additional option stack_dim is available to
            select along which dimension the data should be stacked to form a batch.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        stack_dim (int): Dimension along which to stack to form the batch. (default: 0)
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.
    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')

        super(LTRLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, collate_fn, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim



class TransTSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_search_frames=num_search_frames, num_template_frames=num_template_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)