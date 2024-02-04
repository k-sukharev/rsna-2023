from collections.abc import Hashable, Mapping

from monai.apps.detection.transforms.box_ops import resize_boxes
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils.enums import TransformBackends
from monai.transforms.utils import check_non_lazy_pending_ops


class ResizeBoxd(MapTransform):
    """
    Resize the input boxes when the corresponding image is
    resized to given spatial size (with scaling, not cropping/padding).

    Args:
        image_key: key of image with expected shape of spatial dimensions after resize operation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].
        box_ref_image_keys: Keys that represent the reference images to which ``box_keys`` are attached.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        image_key: str,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        allow_missing_keys: bool = False
    ) -> None:
        self.image_key = image_key
        self.box_keys = ensure_tuple(box_keys)
        super().__init__(self.box_keys, allow_missing_keys)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:  # type: ignore[override]
        d = dict(data)

        for box_key, box_ref_image_key in self.key_iterator(d, self.box_ref_image_keys):
            boxes = d[box_key]
            src_spatial_size = d[box_ref_image_key].shape[1:]
            dst_spatial_size = d[self.image_key].shape[1:]
            d[box_key] = resize_boxes(boxes, src_spatial_size, dst_spatial_size)
        return d


from collections.abc import Sequence
from itertools import chain

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import LazyTransform
from monai.transforms.croppad.array import Crop, Pad, BorderPad
from monai.transforms.croppad.dictionary import Cropd
from monai.transforms.utils import compute_divisible_spatial_size
from monai.utils import (
    LazyAttr,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    ensure_tuple,
    ensure_tuple_rep
)

from monai.config import KeysCollection, SequenceStr


def generate_spatial_bounding_box(
    img: NdarrayOrTensor,
    box: NdarrayOrTensor,
    margin: Sequence[int] | int = 0,
    allow_smaller: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Generate the spatial bounding box of foreground in the image with start-end positions (inclusive).
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:

        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]

    If `allow_smaller`, the bounding boxes edges are aligned with the input image edges.
    This function returns [0, 0, ...], [0, 0, ...] if there's no positive intensity.

    Args:
        img: a "channel-first" image of shape (C, spatial_dim1[, spatial_dim2, ...]) to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
        allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
            than box size, default to `True`.
    """
    check_non_lazy_pending_ops(img, name="generate_spatial_bounding_box")
    spatial_size = img.shape[1:]

    ndim = len(spatial_size)
    margin = ensure_tuple_rep(margin, ndim)
    for m in margin:
        if m < 0:
            raise ValueError(f"margin value should not be negative number, got {margin}.")

    box_start, box_end = box[0, 0::2], box[0, 1::2]

    for di, (min_d_, max_d_) in enumerate(zip(box_start, box_end)):
        min_d = min_d_ - margin[di]
        max_d = max_d_ + margin[di] + 1
        if allow_smaller:
            min_d = max(min_d, 0)
            max_d = min(max_d, spatial_size[di])

        box_start[di] = min_d.detach().cpu().item() if isinstance(min_d, torch.Tensor) else min_d
        box_end[di] = max_d.detach().cpu().item() if isinstance(max_d, torch.Tensor) else max_d

    return box_start, box_end


class CropBox(Crop):
    """
    Crop an image using a bounding box. The bounding box is generated by selecting foreground using select_fn
    at channels channel_indices. margin is added in each spatial dimension of the bounding box.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box of foreground object.
    For example:

    .. code-block:: python

        image = np.array(
            [[[0, 0, 0, 0, 0],
              [0, 1, 2, 1, 0],
              [0, 1, 3, 2, 0],
              [0, 1, 2, 1, 0],
              [0, 0, 0, 0, 0]]])  # 1x5x5, single channel 5x5 image


        def threshold_at_one(x):
            # threshold at 1
            return x > 1


        cropper = CropForeground(select_fn=threshold_at_one, margin=0)
        print(cropper(image))
        [[[2, 1],
          [3, 2],
          [2, 1]]]

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    """

    def __init__(
        self,
        margin: Sequence[int] | int = 0,
        allow_smaller: bool = True,
        return_coords: bool = False,
        k_divisible: Sequence[int] | int = 1,
        mode: str = PytorchPadMode.CONSTANT,
        lazy: bool = False,
        **pad_kwargs,
    ) -> None:
        """
        Args:
            select_fn: function to select expected foreground, default is to select values > 0.
            # channel_indices: if defined, select foreground only on the specified channels
            #     of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
                than box size, default to `True`. if the margined size is larger than image size, will pad with
                specified `mode`.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
            pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        LazyTransform.__init__(self, lazy)
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.padder = Pad(mode=mode, lazy=lazy, **pad_kwargs)

    @Crop.lazy.setter  # type: ignore
    def lazy(self, _val: bool):
        self._lazy = _val
        self.padder.lazy = _val

    @property
    def requires_current_data(self):
        return False

    def compute_bounding_box(self, img: NdarrayOrTensor, box: NdarrayOrTensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(
            img, box, self.margin, self.allow_smaller
        )
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        return box_start_, box_end_

    def crop_pad(
        self,
        img: torch.Tensor,
        box_start: np.ndarray,
        box_end: np.ndarray,
        mode: str | None = None,
        lazy: bool = False,
        **pad_kwargs,
    ) -> torch.Tensor:
        """
        Crop and pad based on the bounding box.

        """
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = super().__call__(img=img, slices=slices, lazy=lazy)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(
            box_end - np.asarray(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]), 0
        )
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(
            cropped.peek_pending_shape() if isinstance(cropped, MetaTensor) else cropped.shape[1:]
        )
        ret = self.padder.__call__(img=cropped, to_pad=pad_width, mode=mode, lazy=lazy, **pad_kwargs)
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta() and isinstance(ret, MetaTensor):
            if not lazy:
                ret.applied_operations[-1][TraceKeys.EXTRA_INFO]["pad_info"] = ret.applied_operations.pop()
            else:
                pad_info = ret.pending_operations.pop()
                crop_info = ret.pending_operations.pop()
                extra = crop_info[TraceKeys.EXTRA_INFO]
                extra["pad_info"] = pad_info
                self.push_transform(
                    ret,
                    orig_size=crop_info.get(TraceKeys.ORIG_SIZE),
                    sp_size=pad_info[LazyAttr.SHAPE],
                    affine=crop_info[LazyAttr.AFFINE] @ pad_info[LazyAttr.AFFINE],
                    lazy=lazy,
                    extra_info=extra,
                )
                breakpoint()
        return ret

    def __call__(  # type: ignore[override]
        self, img: torch.Tensor, box: NdarrayOrTensor, mode: str | None = None, lazy: bool | None = None, **pad_kwargs
    ) -> torch.Tensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img, box)
        lazy_ = self.lazy if lazy is None else lazy
        cropped = self.crop_pad(img, box_start, box_end, mode, lazy=lazy_, **pad_kwargs)

        if self.return_coords:
            return cropped, box_start, box_end  # type: ignore[return-value]
        return cropped

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.get_most_recent_transform(img)
        # we moved the padding info in the forward, so put it back for the inverse
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return super().inverse(inv)


class CropBoxd(Cropd):
    """
    Dictionary-based version :py:class:`monai.transforms.CropForeground`.
    Crop only the foreground object of the expected images.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        image_key: str,
        margin: Sequence[int] | int = 0,
        allow_smaller: bool = True,
        k_divisible: Sequence[int] | int = 1,
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **pad_kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            image_key: data source to crop the image of bbox, can be image or label, etc.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
                than box size, default to `True`. if the margined size is larger than image size, will pad with
                specified `mode`.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
            start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
            end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
            pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        self.image_key = image_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        cropper = CropBox(
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            lazy=lazy,
            **pad_kwargs,
        )
        super().__init__(box_keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.mode = mode

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self):
        return True

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.cropper: CropBox
        # TODO: remove key_iterator
        for box_key in self.key_iterator(d):
            box_start, box_end = self.cropper.compute_bounding_box(img=d[self.image_key], box=d[box_key])
            if self.start_coord_key is not None:
                d[self.start_coord_key] = box_start  # type: ignore
            if self.end_coord_key is not None:
                d[self.end_coord_key] = box_end  # type: ignore

            lazy_ = self.lazy if lazy is None else lazy
            d[self.image_key] = self.cropper.crop_pad(img=d[self.image_key], box_start=box_start, box_end=box_end, mode=self.mode, lazy=lazy_)
        return d


class ConcatItemsd(MapTransform):
    """
    Concatenate specified items from data dictionary together on the first dim to construct a big array.
    Expect all the items are numpy array or PyTorch Tensor or MetaTensor.
    Return the first input's meta information when items are MetaTensor.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, name: str, dim: int = 0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be concatenated together.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name corresponding to the key to store the concatenated data.
            dim: on which dimension to concatenate the items, default is 0.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.name = name
        self.dim = dim

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Raises:
            TypeError: When items in ``data`` differ in type.
            TypeError: When the item type is not in ``Union[numpy.ndarray, torch.Tensor, MetaTensor]``.

        """
        d = dict(data)
        output = []
        data_type = None
        for key in self.key_iterator(d):
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])

        if len(output) == 0:
            return d


        filename_or_obj = d[self.name].meta['filename_or_obj'] if self.name in d else None

        if data_type is np.ndarray:
            d[self.name] = np.concatenate(output, axis=self.dim)
        elif issubclass(data_type, torch.Tensor):  # type: ignore
            d[self.name] = torch.cat(output, dim=self.dim)  # type: ignore
        else:
            raise TypeError(
                f"Unsupported data type: {data_type}, available options are (numpy.ndarray, torch.Tensor, MetaTensor)."
            )

        if filename_or_obj is not None:
            d[self.name].meta['filename_or_obj'] = filename_or_obj

        return d
