import torch.nn as nn
import torchvision.models as pre_models


# Return first n layers of a pretrained model
class model_extractor(nn.Module):
    def __init__(self, arch, num_layers, fix_weights):
        """
        初始化模型提取器。

        参数:
        arch (str): 预训练模型的名称（例如 'alexnet', 'resnet', 'vgg16' 等）。
        num_layers (int): 需要提取的前 n 层。
        fix_weights (bool): 是否固定提取层的权重（True 表示固定权重）。
        """
        super(model_extractor, self).__init__()
        # 根据输入的模型架构名称选择对应的预训练模型
        if arch.startswith('alexnet') :
            original_model = pre_models.alexnet(pretrained=True)
        elif arch.startswith('resnet') :
            original_model = pre_models.resnet18(pretrained=True)
        elif arch.startswith('vgg16'):
            original_model = pre_models.vgg16_bn(pretrained=True)
        elif arch.startswith('inception_v3'):
            original_model = pre_models.inception_v3(pretrained=True)
        elif arch.startswith('densenet121'):
            original_model = pre_models.densenet121(pretrained=True)
        elif arch.startswith('googlenet'):
            original_model = pre_models.googlenet(pretrained=True)
        else :
            raise("Not support on this architecture yet")
        # 提取预训练模型的前 num_layers 层
        self.features = nn.Sequential(*list(original_model.children())[:num_layers])
        # 如果 fix_weights 为 True，则固定权重
        if fix_weights == True:
            # Freeze the Model's weights with unfixed Batch Norm
            self.features.train()                   # Unfix all the layers
            for p in self.features.parameters():
                p.requires_grad = False             # Fix all the layers excluding BatchNorm layers
        self.modelName = arch

    def forward(self, x):
        f = self.features(x)
        return f
