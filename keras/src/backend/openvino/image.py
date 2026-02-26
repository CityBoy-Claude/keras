import openvino.opset15 as ov_opset

from keras.src import backend
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output


def rgb_to_grayscale(images, data_format=None):
    images = get_ov_output(images)
    data_format = backend.standardize_data_format(data_format)
    if images.get_partial_shape().rank not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    channel_axis = -3 if data_format == "channels_first" else -1
    if images.shape[channel_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )

    if images.shape[channel_axis] == 3:
        original_type = images.get_element_type()
        rgb_weights = ov_opset.constant(
            [0.2989, 0.5870, 0.1140], dtype=original_type
        ).output(0)
        if data_format == "channels_first":
            rgb_weights = ov_opset.unsqueeze(rgb_weights, axes=[-2, -1]).output(
                0
            )
        grayscales = ov_opset.multiply(images, rgb_weights).output(0)
        grayscales = ov_opset.reduce_sum(
            grayscales, reduction_axes=[channel_axis]
        ).output(0)
        grayscales = ov_opset.unsqueeze(grayscales, axes=[channel_axis]).output(
            0
        )
        if grayscales.get_element_type() != original_type:
            # Type of grayscales may be changed after unsqueeze, so we need to
            # convert it back to the original type.
            grayscales = ov_opset.convert(grayscales, original_type).output(0)

    return OpenVINOKerasTensor(grayscales)


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format="channels_last",
):
    raise NotImplementedError("`resize` is not supported with openvino backend")


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`affine_transform` is not supported with openvino backend"
    )


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`perspective_transform` is not supported with openvino backend"
    )


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0
):
    raise NotImplementedError(
        "`map_coordinates` is not supported with openvino backend"
    )


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    raise NotImplementedError(
        "`gaussian_blur` is not supported with openvino backend"
    )


def elastic_transform(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    raise NotImplementedError(
        "`elastic_transform` is not supported with openvino backend"
    )


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    raise NotImplementedError(
        "`scale_and_translate` is not supported with openvino backend"
    )
