import torch
import torchvision.transforms.v2 as transforms
import numpy as np


class DinoV2(torch.nn.Module):
    """
    This class loads the DinoV2 model ("DINOv2: Learning Robust Visual Features without Supervision", Oquab et al.)
        and performs inference
    """
    def __init__(self, model_type, device, reg=True):
        """
        Initializes the DINOv2 model.

        Args:
            model_type (str): The ViT variant to use ('s', 'b', 'l', 'g').
            device (str): The device to run the model on.
            reg (bool): Whether to use registers, see ("Vision Transformers Need Registers", Darcet et al.). 
                Default is True.

        """
        super().__init__()
        reg = "_reg" if reg else ""
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_type}14{reg}').to(device).eval()
        self.intermediate_layer_idx = {
            's': [2, 5, 8, 11],
            'b': [2, 5, 8, 11], 
            'l': [4, 11, 17, 23], 
            'g': [9, 19, 29, 39]
        }[model_type]
        self.patch_size = 14
        self.image_size = 518  # Must be a multiple of the patch size.
        self.device = device

    def transform(self, image):
        """
        Transforms an input image into the format required by the DINOv2 model.

        Args:
            image (numpy.ndarray): The input image to transform (H, W, 3).

        Returns:
            torch.Tensor: The transformed image as a tensor, ready for input into the model.
        """
        height, width = image.shape[:2]
        if height < width:
            h_ = self.image_size
            w_ = int(np.round(width * h_ / height / self.patch_size)) * self.patch_size
        else:
            w_ = self.image_size
            h_ = int(np.round(height * w_ / width / self.patch_size)) * self.patch_size

        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((h_, w_)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform(image)

    def forward(self, x, transform=True):
        """
        Forward pass through the DINOv2 model to extract intermediate features.

        Args:
            x (torch.Tensor or numpy.ndarray): The input image or batch of images.
            transform (bool): Whether to apply the transformation to the input image before passing it to the model.

        Returns:
            List of 4 tuples, each containing (patch_embedding, class_token) from different stages.
        """
        if transform:
            x = self.transform(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            intermediate_features = self.dinov2.get_intermediate_layers(
                x, 
                self.intermediate_layer_idx,
                reshape=True,
                return_class_token=True
            )
        return intermediate_features