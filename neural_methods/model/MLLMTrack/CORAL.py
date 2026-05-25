import torch
import torch.nn.functional as F


class CORAL:
    def __init__(self):
        """
        mmd_gamma: CORAL 损失的权重
        """
        super(CORAL, self).__init__()

    def coral(self, x, y):
        """
        计算两个域特征 x 和 y 之间的 CORAL 损失，
        适配于三维输入，假设输入 x, y 的形状为 [B, N, D]。

        计算步骤：
            1. 将 x 和 y 展平为二维数据 [B*N, D]。
            2. 计算展平后特征的均值和中心化后的协方差矩阵。
            3. 分别计算均值差异和协方差差异（取平方后求均值）。
            4. 返回均值差异与协方差差异之和作为 CORAL 损失。
        """
        # 假设输入 x 和 y 的形状为 [B, N, D]
        B, N, D = x.shape
        # 将 x 和 y 展平为二维张量 [B*N, D]
        x_flat = x.view(-1, D)
        y_flat = y.view(-1, D)

        # 计算均值
        mean_x = x_flat.mean(dim=0, keepdim=True)
        mean_y = y_flat.mean(dim=0, keepdim=True)
        # 中心化
        cent_x = x_flat - mean_x
        cent_y = y_flat - mean_y

        # 计算协方差矩阵，除以 (n - 1) 保证无偏估计
        n_x = x_flat.size(0)
        n_y = y_flat.size(0)
        cova_x = (cent_x.t() @ cent_x) / (n_x - 1)
        cova_y = (cent_y.t() @ cent_y) / (n_y - 1)

        # 计算均值差异和协方差差异（均取平方后求均值）
        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def forward(self, minibatches, features):
        """
        更新模型参数，仅利用 CORAL 损失对齐不同域的三维特征，
        从而更新 encoder 参数，使其提取域不变特征。

        参数：
            minibatches: 域的数量
            batchsize: 域的大小
        """
        loss = 0  # 用于累计 CORAL 损
        B, N, D = features.shape
        batchsize = features.shape[0] / minibatches # 域的大小
        batchsize = int(batchsize)

        # 计算任意两个域之间的 CORAL 损失
        for i in range(minibatches):
            for j in range(i + 1, minibatches):
                loss += self.coral(features[i*batchsize:(i+1)*batchsize , :], features[j*batchsize:(j+1)*batchsize , :])

        # 对所有域两两之间的 CORAL 损失求平均（组合对数）
        if minibatches > 1:
            loss /= (minibatches * (minibatches - 1) / 2)

        return loss

if __name__ == '__main__':

    features = torch.randn(4*2, 5, 3)

    coral_module = CORAL()
    loss_dict = coral_module.forward(2, features)
    print("CORAL 损失：", loss_dict)