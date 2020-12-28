import os
import torch


class ResBlock(torch.nn.Module):
    """Define a resudial block with pre-activations
    """
    def __init__(self, input_maps, output_maps, stride):
        super(ResBlock, self).__init__()
        self.input_maps = input_maps
        self.bottleneck_maps = output_maps // 4
        self.output_maps = output_maps
        self.stride = stride

        self.conv0 = torch.nn.Conv2d(in_channels=self.input_maps,
                                     out_channels=self.bottleneck_maps,
                                     kernel_size=(1, 1), stride=self.stride,
                                     padding=(0, 0))
        self.conv1 = torch.nn.Conv2d(in_channels=self.bottleneck_maps,
                                     out_channels=self.bottleneck_maps,
                                     kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=self.bottleneck_maps,
                                     out_channels=self.output_maps,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     padding=(0, 0))
        self.conv_ident = torch.nn.Conv2d(in_channels=self.input_maps,
                                          out_channels=self.output_maps,
                                          kernel_size=(1, 1), stride=self.stride,
                                          padding=(0, 0))

        self.bn0 = torch.nn.BatchNorm2d(num_features=self.input_maps)
        self.bn1 = torch.nn.BatchNorm2d(num_features=self.bottleneck_maps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=self.bottleneck_maps)

    def forward(self, x_in):
        if self.input_maps == self.output_maps:
            # Conv 1x1
            x = self.bn0(x_in)
            x = torch.nn.functional.elu(x)
            x = self.conv0(x)

            # Conv 3x3
            x = self.bn1(x)
            x = torch.nn.functional.elu(x)
            x = self.conv1(x)

            # Conv 1x1
            x = self.bn2(x)
            x = torch.nn.functional.elu(x)
            x = self.conv2(x)

            # Output = Residual (x) + Shortcut (x_in)
            return torch.add(x, x_in)

        else:
            x = self.bn0(x_in)
            x_temp = torch.nn.functional.elu(x)

            # Conv 1x1
            x = self.conv0(x_temp)

            # Conv 3x3
            x = self.bn1(x)
            x = torch.nn.functional.elu(x)
            x = self.conv1(x)

            # Conv 1x1
            x = self.bn2(x)
            x = torch.nn.functional.elu(x)
            x = self.conv2(x)

            x_short = self.conv_ident(x_temp)

            # Output = Residual (x) + Shortcut (x_short)
            return torch.add(x, x_short)



class ResNet(torch.nn.Module):
    """Define a resudial block with pre-activations
    """
    def __init__(self):
        super(ResNet, self).__init__()

        # Simple ResNet with pre-activation architecture
        feature_maps = [8, 16, 32, 64, 128, 256, 512]

        self.first_conv = torch.nn.Conv2d(in_channels=3,
                                          out_channels=feature_maps[0],
                                          kernel_size=(3, 3), stride=(1, 1),
                                          padding=(1, 1))

        self.block0 = ResBlock(input_maps=feature_maps[0],
                               output_maps=feature_maps[1], stride=(2, 2))
        self.block0_1 = ResBlock(input_maps=feature_maps[1],
                                 output_maps=feature_maps[1], stride=(1, 1))

        self.block1 = ResBlock(input_maps=feature_maps[1],
                               output_maps=feature_maps[2], stride=(2, 2))
        self.block1_1 = ResBlock(input_maps=feature_maps[2],
                                 output_maps=feature_maps[2], stride=(1, 1))

        self.block2 = ResBlock(input_maps=feature_maps[2],
                               output_maps=feature_maps[3], stride=(2, 2))
        self.block2_1 = ResBlock(input_maps=feature_maps[3],
                                 output_maps=feature_maps[3], stride=(1, 1))

        self.block3 = ResBlock(input_maps=feature_maps[3],
                               output_maps=feature_maps[4], stride=(2, 2))
        self.block3_1 = ResBlock(input_maps=feature_maps[4],
                                 output_maps=feature_maps[4], stride=(1, 1))

        self.block4 = ResBlock(input_maps=feature_maps[4],
                               output_maps=feature_maps[5], stride=(2, 2))
        self.block4_1 = ResBlock(input_maps=feature_maps[5],
                                 output_maps=feature_maps[5], stride=(1, 1))

        self.block5 = ResBlock(input_maps=feature_maps[5],
                               output_maps=feature_maps[6], stride=(2, 2))
        self.block5_1 = ResBlock(input_maps=feature_maps[6],
                                 output_maps=feature_maps[6], stride=(1, 1))

        self.max_pool0 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1d_0 = torch.nn.Dropout(0.3)
        self.fc4 = torch.nn.Linear(in_features=(4 * 4 * feature_maps[6]), out_features=5)


    def forward(self, x_in):
        x = self.first_conv(x_in)
        x = self.block0(x)
        x = self.block0_1(x)
        x = self.block1(x)
        x = self.block1_1(x)
        x = self.block2(x)
        x = self.block2_1(x)
        x = self.block3(x)
        x = self.block3_1(x)
        x = self.block4(x)
        x = self.block4_1(x)
        x = self.block5(x)
        x = self.block5_1(x)
        x = self.max_pool0(x)

        x = torch.flatten(x, 1)

        x = self.dropout1d_0(x)
        x = self.fc4(x)
        out_softmax = torch.nn.functional.log_softmax(x, dim=-1)

        return out_softmax