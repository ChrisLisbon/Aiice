from torchcnnbuilder.models import ForecasterBase
from torch import nn
from typing import Union, List

class CustomEncoderDecoder(nn.Module):
    def __init__(self, num_layers:int, channels_in: int, channels_out: int, image_size:Union[List[int], int]) -> None:
        """
        A custom Encoder-Decoder model for image forecasting, built upon `ForecasterBase`.

        This class wraps the `ForecasterBase` model from `torchcnnbuilder` to provide
        a convenient interface for constructing a Encoder - Decoder architecture tailored
        for forecasting tasks. It handles the initial setup of the image size
        and delegates the core model logic to `ForecasterBase`.

        Args:
            num_layers (int): The number of layers (depth) for the Encoder-Decoder architecture.
                            This parameter is passed directly to `ForecasterBase`.
            channels_in (int): The number of input channels (channels are the smallest uni). This corresponds
                            to `in_time_points` in `ForecasterBase`.
            channels_out (int): The number of output channels (time duration that model will predict). 
                            This corresponds to `out_time_points` in `ForecasterBase`.
            image_size (Union[List[int], int]): The spatial dimensions of the input image.
                                                 If an integer is provided, it will be
                                                 interpreted as a square image (e.g., 256
                                                 becomes [256, 256]). This is used as
                                                 `input_size` for `ForecasterBase`.
        """
        super().__init__()
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        self.model = ForecasterBase(input_size = image_size,
                                    in_time_points = channels_in,
                                    out_time_points = channels_out,
                                    n_layers = num_layers)
    
    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor to the model. Expected shape depends on
                              the `ForecasterBase` requirements, typically
                              `(batch_size, channels_in, height, width)`.

        Returns:
            torch.Tensor: The output tensor from the model. Expected shape typically
                          `(batch_size, channels_out, height, width)`.
        """
        return self.model(x)
    