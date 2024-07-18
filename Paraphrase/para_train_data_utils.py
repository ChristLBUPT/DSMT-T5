import sys; sys.path.append('..')
try:
    from para_datasets import ParaNMTDataset, BlockDistributedSampler
except ImportError as e:
    from .para_datasets import ParaNMTDataset, BlockDistributedSampler
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from argparse import Namespace
import os
import random as rd
import logging

def get_datasets_and_dataloaders(opt, tokenizer):
    # distributed environments
    rank = int(getattr(os.environ, 'RANK', 0))
    local_rank = int(getattr(os.environ, 'LOCAL_RANK', 0))
    world_size = int(getattr(os.environ, 'WORLD_SIZE', 1))
    ddp_ignored = world_size == 1
    device = getattr(opt, 'cuda', 'cpu')
    if isinstance(device, int):
        device = f'cuda:{device}'
    # initialize datasets
    if not opt.test_only:
        if 'triplet' in opt.instance_format_name or 'triple' in opt.data:
            train_has_ref = True 
            logging.info('NOTE: training set is using exemplar sentences')
        else:
            train_has_ref = False
        train_dataset = ParaNMTDataset(
            data_dir=f'./data/{opt.data}/train' if not opt.use_opti else f'./data/{opt.data}_raw/train',
            model=opt.model_type,
            tokenizer=tokenizer,
            prune_height=opt.prune_height,
            # has_ref=False if 'triplet' not in opt.instance_format_name else True,
            has_ref=train_has_ref,
            use_num_postfix=opt.use_num_postfix,
            tgt_only=opt.target_only,
            instance_format_name=opt.instance_format_name,
            max_length=opt.max_length
        )
    else:
        train_dataset = None
    # test (val) dataset using template sentence (Y) as syntactic exemplar and target sentence (Z) as syntactic exemplar
    test_dataset_exemplar = ParaNMTDataset(
        data_dir=f'./data/{opt.data}/test' if not opt.use_opti else f'./data/{opt.data}_raw/test',
        model=opt.model_type,
        tokenizer=tokenizer,
        prune_height=opt.prune_height,
        has_ref=True,
        use_num_postfix=opt.use_num_postfix,
        tgt_only=opt.target_only,
        use_tgt_as_ref=False,
        instance_format_name=opt.instance_format_name,
        max_length=1000
    )

    test_dataset_target = ParaNMTDataset(
        data_dir=f'./data/{opt.data}/test' if not opt.use_opti else f'./data/{opt.data}_raw/test',
        model=opt.model_type,
        tokenizer=tokenizer,
        prune_height=opt.prune_height,
        has_ref=True,
        use_num_postfix=opt.use_num_postfix,
        tgt_only=opt.target_only,
        use_tgt_as_ref=True,
        instance_format_name=opt.instance_format_name,
        max_length=1000
    )

    val_dataset_exemplar = ParaNMTDataset(
        data_dir=f'./data/{opt.data}/test' if not opt.use_opti else f'./data/{opt.data}_raw/test',
        model=opt.model_type,
        tokenizer=tokenizer,
        prune_height=opt.prune_height,
        has_ref=True,
        use_num_postfix=opt.use_num_postfix,
        tgt_only=opt.target_only,
        use_tgt_as_ref=False,
        instance_format_name=opt.instance_format_name,
        max_length=1000
    )
    val_dataset_target = ParaNMTDataset(
        data_dir=f'./data/{opt.data}/test' if not opt.use_opti else f'./data/{opt.data}_raw/test',
        model=opt.model_type,
        tokenizer=tokenizer,
        prune_height=opt.prune_height,
        has_ref=True,
        use_num_postfix=opt.use_num_postfix,
        tgt_only=opt.target_only,
        use_tgt_as_ref=True,
        instance_format_name=opt.instance_format_name,
        max_length=1000
    )
    if opt.subset:
        subset_cnt = getattr(opt, 'subset_cnt', 1024)
        if not opt.test_only:
            if 0 < subset_cnt < 1:
                # subset_divisor = 1 / subset_cnt
                n_subset_samples = int(len(train_dataset) * subset_cnt)
                # train_dataset = Subset(train_dataset, range(round(len(train_dataset) * subset_cnt)))
            else:
                # subset_divisor = len(train_dataset) / subset_cnt
                n_subset_samples = subset_cnt
                # train_dataset = Subset(train_dataset, range(int(subset_cnt)))
            subset_allowed_indices = [*range(len(train_dataset))]
            rd.shuffle(subset_allowed_indices)
            subset_allowed_indices = subset_allowed_indices[: n_subset_samples]
            # subset_allowed_indices = [round(i * subset_divisor) for i in range(n_subset_samples) if i < len(train_dataset)]
            train_dataset = Subset(train_dataset, subset_allowed_indices)
        # val_dataset = Subset(val_dataset, range(50))
        # test_dataset = Subset(test_dataset, range(50))

    if not opt.test_only:
        if 'SEED' in os.environ:
            logging.info(f"training data (but not val and test data) will be reshuffled since a different `SEED` is set ({os.environ['SEED']})")
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True if 'SEED' in os.environ else 'FALSE', # shuffle for random testing, consistent for best reproductivity
            num_workers=opt.num_workers,
            prefetch_factor=opt.prefetch_factor,
            collate_fn=train_dataset.collate_fn if not isinstance(train_dataset, Subset) else train_dataset.dataset.collate_fn,
            sampler=DistributedSampler(train_dataset, drop_last=True) if not ddp_ignored else None,
            pin_memory=True,
            pin_memory_device=f"cuda:{local_rank}" if not ddp_ignored else device,
        )
    else:
        train_loader = None
    if opt.model_type == 'bart_teacherforcing' or opt.generation_debug:
        val_test_batch_size = 1
    else:
        val_test_batch_size = opt.batch_size // 2
    test_loader_exemplar = DataLoader(
        test_dataset_exemplar,
        batch_size=val_test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        prefetch_factor=opt.prefetch_factor,
        collate_fn=test_dataset_exemplar.collate_fn if not isinstance(test_dataset_exemplar, Subset) else test_dataset_exemplar.dataset.collate_fn,
        sampler=BlockDistributedSampler(test_dataset_exemplar, rank, world_size) if not ddp_ignored else None,
        pin_memory=True,
        pin_memory_device=f"cuda:{local_rank}" if not ddp_ignored else device,
    )
    test_loader_target = DataLoader(
        test_dataset_target,
        batch_size=val_test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        prefetch_factor=opt.prefetch_factor,
        collate_fn=test_dataset_target.collate_fn if not isinstance(test_dataset_target, Subset) else test_dataset_target.dataset.collate_fn,
        sampler=BlockDistributedSampler(test_dataset_target, rank, world_size) if not ddp_ignored else None,
        pin_memory=True,
        pin_memory_device=f"cuda:{local_rank}" if not ddp_ignored else device,
    )
    val_loader_exemplar = DataLoader(
        val_dataset_exemplar,
        batch_size=val_test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        prefetch_factor=opt.prefetch_factor,
        collate_fn=val_dataset_exemplar.collate_fn if not isinstance(val_dataset_exemplar, Subset) else val_dataset_exemplar.dataset_exemplar.collate_fn,
        sampler=BlockDistributedSampler(val_dataset_exemplar, rank, world_size) if not ddp_ignored else None,
        pin_memory=True,
        pin_memory_device=f"cuda:{local_rank}" if not ddp_ignored else device,
    )
    val_loader_target = DataLoader(
        val_dataset_target,
        batch_size=val_test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        prefetch_factor=opt.prefetch_factor,
        collate_fn=val_dataset_target.collate_fn if not isinstance(val_dataset_target, Subset) else val_dataset_target.dataset_target.collate_fn,
        sampler=BlockDistributedSampler(val_dataset_target, rank, world_size) if not ddp_ignored else None,
        pin_memory=True,
        pin_memory_device=f"cuda:{local_rank}" if not ddp_ignored else device,
    )

    return train_loader, val_loader_exemplar, val_loader_target, test_loader_exemplar, test_loader_target