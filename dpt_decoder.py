import torch
import torch.nn as nn


class DPT(nn.Module):
    """
    Dense Prediction Transformer (DPT) ("Vision Transformers for Dense Prediction", Ranftl et al.). 
    This implementation is inspired by the DinoV2 and Depth Anything V2 repositories.
    """
    def __init__(
            self, 
            embed_dim, 
            features,
            output_dim, 
            use_class_token=False, 
            use_bn=False
        ):
        """
        Initializes the Dense Prediction Transformer (DPT) decoder.

        Args:
            embed_dim (int): The embedding dimension of the ViT backbone:
            features (int): The feature dimension to use when fusing feature maps from different stages.
            output_dim (int): The numer of channels to output.
            use_class_token (bool): Whether to use the class tokens from the different stages. Default is False.
            use_bn (bool): Whether to use batch norm. Default is False.
        """
        super(DPT, self).__init__()
        self.output_size = output_dim

        # Pre processing blocks to prepare the ViT backbone output
        self.pre_processing_block = PreProcessingBlock(
            embed_dim, 
            [embed_dim // 2 ** (3 - i) for i in range(4)], 
            features,
            use_class_token
        )

        # Fusion blocks to merge feature maps from different stages
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(features, use_bn) for _ in range(4)
        ])
        self.fusion_blocks[0].res_conv_unit1 = None

        # Post process output
        self.out_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, embeddings):
        """
        Forward pass through the DPT decoder.

        Args:
            embeddings (list): List of 4 tuples, each containing (patch_embedding, class_token) from different
                stages of the ViT backbone.

        Returns:
            An output tensor with recovered spatial resoultion and the required number of channels.
        """
        h, w = embeddings[0][0].shape[-2:]

        x = self.pre_processing_block(embeddings)
        out = self.fusion_blocks[0](x[-1], target_shape=x[-2].shape[-2:])
        for i in range(1, 4):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        
        out = self.out_conv1(out)
        out = nn.functional.interpolate(out, (int(h * 14), int(w * 14)), mode="bilinear", align_corners=True)
        out = self.out_conv2(out)

        return out


class FeatureFusionBlock(nn.Module):
    """
    FeatureFusionBlock, merges feature maps from different stages.
    """
    def __init__(self, in_channels, use_bn):
        """
        Initializes the Feature Fusion Block.
        
        Args:
            in_channels (int): Number of input channels for the feature maps.
            use_bn (bool): Whether to use batch norm.
        """
        super(FeatureFusionBlock, self).__init__()
        self.res_conv_unit1 = ResConvUnit(in_channels, use_bn)
        self.res_conv_unit2 = ResConvUnit(in_channels, use_bn)
        self.projection_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, *inputs, target_shape=None):
        """
        Forward pass for the Feature Fusion Block.
        
        Args:
            *inputs (torch.Tensor): First element is the main feature map. 
                If a second element is provided, it's fused with the first.
            target_shape (tuple, optional): Target spatial dimensions (H, W) for the output feature map. 
                If None, the output is upsampled by a factor of 2.
                                          
        Returns:
            torch.Tensor: Processed and fused feature map.
        """
        x = inputs[0]
        
        if len(inputs) == 2:
            x = x + self.res_conv_unit1(inputs[1])

        x = self.res_conv_unit2(x)

        kwargs = dict(scale_factor=2) if target_shape is None else dict(size=target_shape)
        x = nn.functional.interpolate(x, **kwargs, mode="bilinear", align_corners=True)

        x = self.projection_layer(x)
        return x


class ResConvUnit(nn.Module):
    """
    A simple two layer convolutional resnet block.
    """
    def __init__(self, in_channels, use_bn):
        """
        Initializes the ResConvUnit.
        
        Args:
            in_channels (int): Number of input channels.
            use_bn (bool): Whether to use batch norm.
        """
        super(ResConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.relu = nn.ReLU()

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, inputs):
        x = self.conv1(self.relu(inputs))
        if self.use_bn:
            x = self.bn1(x)

        x = self.conv2(self.relu(x))
        if self.use_bn:
            x = self.bn2(x)

        return x + inputs


class PreProcessingBlock(nn.Module):
    """
    Pre processing block to scale and prepare the feature maps from the different stages.
    """
    def __init__(self, in_channels, out_channels_intermediate, out_channels, use_class_token):
        """
        Initializes the PreProcessing Block for feature maps from the ViT backbone.
        
        Args:
            in_channels (int): Number of input channels from the ViT backbone.
            out_channels_intermediate (list): List of intermediate channel dimensions for each stage.
            out_channels (int): Number of output channels for all stages.
            use_class_token (bool): Whether to use the class token information.
        """
        super(PreProcessingBlock, self).__init__()
        self.use_class_token = use_class_token
        self.pre_processing_layers = nn.ModuleList([
            nn.Conv2d(in_channels, c, kernel_size=1) for c in out_channels_intermediate
        ])
        # Resize height and width to (4x, 2x, 1x, .5x)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels_intermediate[0], out_channels_intermediate[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels_intermediate[1], out_channels_intermediate[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels_intermediate[3], out_channels_intermediate[3], kernel_size=3, stride=2, padding=1),
        ])
        self.post_processing_layers = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=3, padding=1) for c in out_channels_intermediate
        ])

        if use_class_token:
            self.cls_token_layers = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(2*in_channels, in_channels, kernel_size=1),
                    nn.GELU()
                ) for _ in range(4)
            )

    def forward(self, inputs):
        """
        Forward pass for the PreProcessing Block.
        
        Args:
            inputs (list): List of 4 tuples, each containing (patch_embedding, class_token) from different
                stages of the ViT backbone.
                          
        Returns:
            list: List of processed feature maps with spatial resolutions scaled by (4, 2, 1, .5) respectively.
        """
        out = []
        for i, (x, cls_token) in enumerate(inputs):
            if self.use_class_token:
                cls_token = cls_token.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, C, 1, 1)
                cls_token = cls_token.expand(-1, -1, x.shape[2], x.shape[3])  # Shape: (B, C, H, W)
                x = torch.cat((x, cls_token), dim=1)  # Concatenate along channels -> (B, 2C, H, W)
                x = self.cls_token_layers[i](x)  # Project back to (B, C, H, W)

            x = self.pre_processing_layers[i](x)
            x = self.resize_layers[i](x)
            x = self.post_processing_layers[i](x)
            out.append(x)

        return out
    

def create_dpt_decoder(model_type, output_dim):
    embed_dim = {
        's': 384,
        'b': 768,
        'l': 1024,
        'g': 1536,
    }[model_type]

    model = DPT(
        embed_dim=embed_dim,
        features = embed_dim//4,
        output_dim=output_dim,
        use_class_token=False,
        use_bn=False
    )
    return model