from dataset.humanml3d_dataset import Text2MotionDatasetEval, collate_fn # TODO
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
import torch
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

def get_dataset_motion_loader(
    opt_path,
    batch_size,
    fname,
    device,
    deterministic=False,
    shuffle=True,
    num_workers=8,
    seed=None,
):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    # if opt.dataset_name == 'humanml3d' or opt.dataset_name == 'kit':
    print('Loading dataset %s ...' % opt.dataset_name)

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    split_file = pjoin(opt.data_root, '%s.txt'%fname)
    dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer, deterministic=deterministic)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=shuffle,
        generator=generator,
    )
    # else:
    #     raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset
