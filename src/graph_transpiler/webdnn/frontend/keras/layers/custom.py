import keras

from webdnn.frontend.keras.converter import KerasConverter


@KerasConverter.register_handler("TorchBatchNorm2D")
def _convert_torch_batch_norm2d(converter: KerasConverter, k_op: "webdnn.frontend.keras.TorchBatchNorm2D"):
    x = converter.get_variable(converter.get_input_tensor(k_op)[0])

    converter.set_variable(converter.get_output_tensor(k_op)[0], x)
