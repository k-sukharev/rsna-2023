import json
import logging
import os

import numpy as np
import lightning.pytorch as pl
import pandas as pd
import pydicom

from pathlib import Path
from typing import List

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from monai.data import DataLoader, Dataset, PydicomReader
from monai.data.utils import orientation_ras_lps
from monai.transforms import Transform
from monai.utils import MetaKeys
from tqdm import tqdm


logger = logging.getLogger(__name__)

AORTIC_HU_MEAN = 223.62237316917853
AORTIC_HU_STD = 103.7677617128023


def get_raw_data(data_dir, seg_dir, split, label_keys, seg_keys):
    images_dir = data_dir / f'{split}_images'
    if seg_dir is not None:
        seg_images_dir = seg_dir / f'{split}_images'
    patient_dirs = sorted(list(images_dir.iterdir()))
    if split == 'train':
        train_df = pd.read_csv(data_dir / 'train.csv').set_index(keys=['patient_id'])
    #     image_level_labels_df = pd.read_csv(data_dir / 'image_level_labels.csv').set_index(
    #         keys=['patient_id', 'series_id']
    #     ).sort_index(axis=0, ascending=True)
    #     train_dicom_tags = pd.read_parquet(data_dir / 'train_dicom_tags.parquet').set_index('path')
    #     train_dicom_tags['Z'] = train_dicom_tags['ImagePositionPatient'].apply(lambda x: json.loads(x)[-1])
    meta_df = pd.read_csv(data_dir / f'{split}_series_meta.csv').set_index(keys=['patient_id', 'series_id'])
    data = []
    for patient_dir in patient_dirs:
        series_dirs = sorted(list(patient_dir.iterdir()))
        for series_dir in series_dirs:
            # series_image_paths = sorted(list(series_dir.iterdir()), key=lambda x: int(x.name.split('.')[0]))
            # lowest_instance_number = int(series_image_paths[0].name.split('.')[0])
            patient_id = int(patient_dir.name)
            series_id = int(series_dir.name)
            data.append(
                {
                    'image': str(series_dir),
                    # 'lowest_instance_number': lowest_instance_number,
                    'patient_id': patient_id,
                    # 'series_id': series_id,
                    'aortic_hu': (float(meta_df.loc[(patient_id, series_id), 'aortic_hu']) - AORTIC_HU_MEAN) / AORTIC_HU_STD
                    # 'incomplete_organ': meta_df.loc[(patient_id, series_id), 'incomplete_organ']
                }
            )
            if split == 'train':
                data[-1]['labels'] = train_df.loc[patient_id, label_keys].values
                # series_len = len(series_image_paths)
                # if (patient_id, series_id) in image_level_labels_df.index:
                #     series_arange = np.arange(lowest_instance_number, lowest_instance_number + series_len)
                #     subset = image_level_labels_df.loc[(patient_id, series_id)]
                #     active_extravasation = np.isin(
                #         series_arange,
                #         subset.loc[subset['injury_name'] == 'Active_Extravasation'].instance_number
                #     ).astype(np.float32)
                #     bowel_injury = np.isin(
                #         series_arange,
                #         subset.loc[subset['injury_name'] == 'Bowel'].instance_number
                #     ).astype(np.float32)
                #     if len(series_image_paths) > 1:
                #         Z1 = train_dicom_tags.loc[f'train_images/{patient_id}/{series_id}/{lowest_instance_number}.dcm', 'Z']
                #         Z2 = train_dicom_tags.loc[f'train_images/{patient_id}/{series_id}/{lowest_instance_number + 1}.dcm', 'Z']
                #         if Z2 - Z1 < 0:
                #             active_extravasation = np.flip(active_extravasation, axis=0).copy()
                #             bowel_injury = np.flip(bowel_injury, axis=0).copy()
                #     else:
                #         active_extravasation = np.zeros(series_len)
                #         bowel_injury = np.zeros(series_len)
                # data[-1]['active_extravasation'] = active_extravasation
                # data[-1]['bowel_injury'] = bowel_injury
                if seg_dir is not None:
                    for seg_key in seg_keys:
                        data[-1][seg_key] = seg_images_dir / str(patient_id) / str(series_id) / f'{seg_key}.nii.gz'
    return data


def get_cached_data(data_dir, temp_img_dir, temp_seg_dir, split, label_keys, seg_keys):
    original_images_dir = data_dir / f'{split}_images'
    if temp_seg_dir is not None:
        temp_seg_images_dir = temp_seg_dir / f'{split}_images'
    patient_dirs = sorted(list(original_images_dir.iterdir()))
    if split == 'train':
        train_df = pd.read_csv(data_dir / 'train.csv').set_index(keys=['patient_id'])
        image_level_labels_df = pd.read_csv(data_dir / 'image_level_labels.csv').set_index(
            keys=['patient_id', 'series_id']
        ).sort_index(axis=0, ascending=True)
        train_dicom_tags = pd.read_parquet(data_dir / 'train_dicom_tags.parquet').set_index('path')
        train_dicom_tags['Z'] = train_dicom_tags['ImagePositionPatient'].apply(lambda x: json.loads(x)[-1])
    meta_df = pd.read_csv(data_dir / f'{split}_series_meta.csv').set_index(keys=['patient_id', 'series_id'])
    data = []
    for patient_dir in patient_dirs:
        series_dirs = sorted(list(patient_dir.iterdir()))
        for series_dir in series_dirs:
            series_image_paths = sorted(list(series_dir.iterdir()), key=lambda x: int(x.name.split('.')[0]))
            lowest_instance_number = int(series_image_paths[0].name.split('.')[0])
            patient_id = int(patient_dir.name)
            series_id = int(series_dir.name)
            data.append(
                {
                    # 'lowest_instance_number': lowest_instance_number,
                    'patient_id': patient_id,
                    # 'series_id': series_id,
                    'aortic_hu': (meta_df.loc[(patient_id, series_id), 'aortic_hu'] - AORTIC_HU_MEAN) / AORTIC_HU_STD
                    # 'incomplete_organ': meta_df.loc[(patient_id, series_id), 'incomplete_organ']
                }
            )
            if temp_img_dir is not None:
                data[-1]['image'] = temp_img_dir / f'{split}_images' / str(patient_id) / f'{series_id}.nii.gz',
            if split == 'train':
                data[-1]['labels'] = train_df.loc[patient_id, label_keys].values
                series_len = len(series_image_paths)
                if (patient_id, series_id) in image_level_labels_df.index:
                    series_arange = np.arange(lowest_instance_number, lowest_instance_number + series_len)
                    subset = image_level_labels_df.loc[(patient_id, series_id)]
                    active_extravasation = np.isin(
                        series_arange,
                        subset.loc[subset['injury_name'] == 'Active_Extravasation'].instance_number
                    ).astype(np.float32)
                    bowel_injury = np.isin(
                        series_arange,
                        subset.loc[subset['injury_name'] == 'Bowel'].instance_number
                    ).astype(np.float32)
                    if len(series_image_paths) > 1:
                        Z1 = train_dicom_tags.loc[f'train_images/{patient_id}/{series_id}/{lowest_instance_number}.dcm', 'Z']
                        Z2 = train_dicom_tags.loc[f'train_images/{patient_id}/{series_id}/{lowest_instance_number + 1}.dcm', 'Z']
                        if Z2 - Z1 < 0:
                            active_extravasation = np.flip(active_extravasation, axis=0).copy()
                            bowel_injury = np.flip(bowel_injury, axis=0).copy()
                    else:
                        active_extravasation = np.zeros(series_len)
                        bowel_injury = np.zeros(series_len)
                data[-1]['active_extravasation'] = active_extravasation
                data[-1]['bowel_injury'] = bowel_injury
                if temp_seg_dir is not None:
                    for seg_key in seg_keys:
                        data[-1][seg_key] = temp_seg_images_dir / str(patient_id) / str(series_id) / f'{seg_key}.nii.gz'
    return data


def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
    return pixel_array


class CustomPydicomReader(PydicomReader):
    def _get_affine(self, metadata: dict, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            metadata: metadata with dict type.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """

        affine: np.ndarray = np.eye(4)
        if not ("00200037" in metadata and "00200032" in metadata):
            return affine
        # "00200037" is the tag of `ImageOrientationPatient`
        rx, ry, rz, cx, cy, cz = metadata["00200037"]["Value"]
        # "00200032" is the tag of `ImagePositionPatient`
        sx, sy, sz = metadata["00200032"]["Value"]
        dr, dc = metadata.get("spacing", (1.0, 1.0))[:2]
        affine[0, 0] = cx * dr
        affine[0, 1] = rx * dc
        affine[0, 3] = sx
        affine[1, 0] = cy * dr
        affine[1, 1] = ry * dc
        affine[1, 3] = sy
        affine[2, 0] = cz * dr
        affine[2, 1] = rz * dc
        affine[2, 2] = 0
        affine[2, 3] = sz

        # 3d
        if "lastImagePositionPatient" in metadata:
            t1n, t2n, t3n = metadata["lastImagePositionPatient"]
            n = metadata[MetaKeys.SPATIAL_SHAPE][-1]
            k1, k2, k3 = (t1n - sx) / (n - 1), (t2n - sy) / (n - 1), (t3n - sz) / (n - 1)
            affine[0, 2] = k1
            affine[1, 2] = k2
            affine[2, 2] = k3

        if lps_to_ras:
            affine = orientation_ras_lps(affine)

        return affine

    def _get_array_data(self, img):
        """
        Get the array data of the image. If `RescaleSlope` and `RescaleIntercept` are available, the raw array data
        will be rescaled. The output data has the dtype np.float32 if the rescaling is applied.

        Args:
            img: a Pydicom dataset object.

        """
        # process Dicom series
        if not hasattr(img, "pixel_array"):
            raise ValueError(f"dicom data: {img.filename} does not have pixel_array.")
        data = standardize_pixel_array(img)
        data = data.astype(np.float32, copy=False)

        slope, offset = 1.0, 0.0
        rescale_flag = False
        if hasattr(img, "RescaleSlope"):
            slope = img.RescaleSlope
            rescale_flag = True
        if hasattr(img, "RescaleIntercept"):
            offset = img.RescaleIntercept
            rescale_flag = True
        if rescale_flag:
            data *= slope
            data += offset

        window_center = float(img.WindowCenter)
        window_width = float(img.WindowWidth)
        np.clip(
            data,
            window_center - window_width / 2,
            window_center + window_width / 2,
            out=data
        )

        min_value = data.min()
        max_value = data.max()
        data -= min_value
        data /= (max_value - min_value + 1e-6)

        if img.PhotometricInterpretation == "MONOCHROME1":
                data *= -1
                data += 1

        data -= 0.5
        data *= 2

        return data

# def create_seg(data_dir, seg_dir, split):
#     images_dir = data_dir / f'{split}_images'
#     patient_dirs = sorted(list(images_dir.iterdir()))

#     for patient_dir in tqdm(patient_dirs, desc=f'Creating segmentations for {split} split'):
#         series_dirs = sorted(list(patient_dir.iterdir()))
#         for series_dir in series_dirs:
#             totalsegmentator(
#                 series_dir,
#                 seg_dir / split / patient_dir.name / series_dir.name,
#                 fast=True,
#                 task='total',
#                 roi_subset=['colon', 'duodenum', 'kidney_left', 'kidney_right', 'liver', 'small_bowel', 'spleen', 'stomach'],
#                 statistics=False
#             )

class MonaiDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: Path,
            seg_dir: Path | None,
            temp_img_dir: Path,
            temp_seg_dir: Path | None,
            label_keys: List[str],
            seg_keys: List[str],
            splits_to_cache: List[str],
            cache_transforms: Transform,
            train_transforms: Transform,
            val_transforms: Transform,
            test_transforms: Transform,
            seed: int = 42,
            fold: int = 0,
            n_splits: int = 5,
            batch_size: int = 32,
            num_workers: int = os.cpu_count(),
            pin_memory: bool = False
        ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.temp_img_dir = temp_img_dir
        self.temp_seg_dir = temp_seg_dir
        self.label_keys = label_keys
        self.seg_keys = seg_keys
        self.splits_to_cache = splits_to_cache
        self.cache_transforms = cache_transforms
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.seed = seed
        self.fold = fold
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        for split in self.splits_to_cache:
            if self.temp_img_dir is not None and not (self.temp_img_dir / f'{split}_images').exists():
                files = get_raw_data(self.data_dir, self.seg_dir, split, self.label_keys, self.seg_keys)
                dataset = Dataset(data=files, transform=self.cache_transforms)
                dataloader = DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=1,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    drop_last=False
                )
                for batch in tqdm(dataloader, desc=f'Preprocessing {split} split'):
                    # logger.info(batch['image'].shape)
                    pass
            # if not (self.seg_dir / split).exists():
            #     create_seg(self.data_dir, self.seg_dir, split)

    def setup(self, stage: str | None = None) -> None:
        if stage == 'fit' or stage is None:
            train_val_data = np.array(get_cached_data(self.data_dir, self.temp_img_dir, self.temp_seg_dir, 'train', self.label_keys, self.seg_keys))
            train_val_labels = np.stack([item['labels'] for item in train_val_data], axis=0)
            mskf = MultilabelStratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed
            )
            folds = list(mskf.split(np.zeros(len(train_val_data)), train_val_labels))

            self.train_dataset = Dataset(data=train_val_data[folds[self.fold][0]], transform=self.train_transforms)
            self.val_dataset = Dataset(data=train_val_data[folds[self.fold][1]], transform=self.val_transforms)
        if stage == 'test' or stage is None:
            test_files = get_cached_data(self.data_dir, self.temp_img_dir, self.temp_seg_dir, 'test', self.label_keys, [])
            self.test_dataset = Dataset(data=test_files, transform=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
