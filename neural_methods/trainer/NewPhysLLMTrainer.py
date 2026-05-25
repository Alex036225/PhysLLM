import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from evaluation.metrics import calculate_metrics, save_metrics_to_file
from neural_methods.model.PhysLLMModules.InputProcessor import RPPGInputProcessor
from neural_methods.model.PhysLLMModules.facexformer import FaceXFormer
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from neural_methods.model.MLLMTrack.con_encoder import ContentProcessor
from neural_methods.model.MLLMTrack.NewPhysLLM_n import NewPhysLLM
# from neural_methods.model.MLLMTrack.NewPhysLLM_alignment import NewPhysLLM
# from neural_methods.model.MLLMTrack.NewPhysLLM_DG import NewPhysLLM
# from neural_methods.model.MLLMTrack.NewPhysLLM_change import  NewPhysLLM
# from neural_methods.model.MLLMTrack.NewPhysLLM_origion import NewPhysLLM
# from neural_methods.model.MLLMTrack.TorchLossComputer import TorchLossComputer
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from tools.mytools.test_plot_save import plot_view_epoch
from fastdtw import fastdtw
import numpy as np
from neural_methods.model.MLLMTrack.FreDFLoss import FreDLyLoss

class NewPhysLLMTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.config = config
        self.data_loader = data_loader

        # 模型参数
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.frame_rate = config.TRAIN.DATA.FS
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.min_valid_loss = None
        self.best_epoch = 0
        # self.num_train_batches = len(data_loader["train"])
        self.model_folder = f"{self.config.MODEL.FACE_MODEL}_{self.config.MODEL.ENV_MODEL}_{self.config.MODEL.VIDEO_ENC}"

        # self.con_encoder = ContentProcessor(config).to(self.device)

        # 实例化 PhysLLM 模型
        print("====== MewPhysLLM Loading =====")
        self.model = NewPhysLLM(config, )
        # print(self.model)
        print("====================================================================")
        # print("layers in the model:")
        # for name, param in self.model.named_parameters():
            # print(f"Layer: {name}, Requires Grad: {param.requires_grad}")
        print("====== Loading Finished =====")
        # print("After initialization:", next(self.model.parameters()).device)
        # print("Model parameter count after initialization:", sum(p.numel() for p in self.model.parameters()))
        self.model = self.model.to(self.device)  # 将模型移到 GPU
        # print("After moving to GPU:", next(self.model.parameters()).device)
        # print("Model parameter count after moving to GPU:", sum(p.numel() for p in self.model.parameters()))
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))
        print("After DataParallel:", next(self.model.parameters()).device)  # 打印模型所在设备
        print("Model parameter count after DataParallel:", sum(p.numel() for p in self.model.parameters()))
        total_params_model, trainable_params_model = self.count_parameters(self.model)
        print(f"Model - Total: {total_params_model}, Trainable: {trainable_params_model}")

        # 定义损失函数和优化器
        self.criterion1 = torch.nn.MSELoss()  # 根据需要选择合适的损失函数
        self.criterion2 = Neg_Pearson()
        self.criterion3 = FreDLyLoss()
        # self.criterion = CombinedLoss()
        # TODO
        lora_params = [p for n, p in self.model.named_parameters() if "lora" in n]
        other_params = [p for n, p in self.model.named_parameters() if "lora" not in n]
        self.optimizer = torch.optim.Adam(
            [
                {"params": lora_params, "lr": config.TRAIN.LR / 5},  # LoRA 参数学习率
                {"params": other_params, "lr": config.TRAIN.LR, "weight_decay":0.00005},  # 其他参数学习率
            ]
        )
        # self.optimizer = optim.Adam(trainable_parameters)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.00005)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # TODO:注释self.scheduler
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        # self.rppg_beseline_tested = False
        
        # 两阶段训练配置
        self.two_stage_training = getattr(config.TRAIN, 'TWO_STAGE_TRAINING', False)
        if self.two_stage_training:
            self.stage1_epochs = config.TRAIN.STAGE1_EPOCHS
            self.stage2_epochs = config.TRAIN.STAGE2_EPOCHS
            self.stage1_lr = config.TRAIN.STAGE1_LR
            self.stage2_lr = config.TRAIN.STAGE2_LR
            self.max_epoch_num = self.stage1_epochs + self.stage2_epochs
            print("\n" + "="*60)
            print("启用两阶段训练:")
            print(f"  Stage 1: {self.stage1_epochs} epochs, LR={self.stage1_lr} (冻结LLM)")
            print(f"  Stage 2: {self.stage2_epochs} epochs, LR={self.stage2_lr} (微调LLM)")
            print("="*60 + "\n")
            
            # 初始化为Stage 1
            self.model.module.set_training_stage('stage1')
            self.current_stage = 1
        else:
            self.current_stage = None

    def _switch_to_stage2(self):
        """切换到第二阶段训练"""
        print("\n" + "="*60)
        print(f"切换到 Stage 2: 解冻LLM，端到端微调")
        print("="*60)
        
        # 设置模型到stage2
        self.model.module.set_training_stage('stage2')
        
        # 重新创建优化器，使用更小的学习率
        lora_params = [p for n, p in self.model.named_parameters() if "lora" in n and p.requires_grad]
        other_params = [p for n, p in self.model.named_parameters() if "lora" not in n and p.requires_grad]
        
        self.optimizer = torch.optim.Adam([
            {"params": lora_params, "lr": self.stage2_lr / 2},  # LoRA参数用更小的学习率
            {"params": other_params, "lr": self.stage2_lr, "weight_decay": 0.00005},
        ])
        
        self.current_stage = 2
        print(f"Stage 2 优化器已重新初始化，LoRA LR={self.stage2_lr/2}, Others LR={self.stage2_lr}\n")

    def train(self, data_loader):
        """Training routine for PhysLLM model."""
        # if data_loader["train"] is None:
            # raise ValueError("No data for training.")

        # 初始化 GradScaler
        # scaler = GradScaler()
        # accelerator = Accelerator()
        # self.model, self.optimizer, data_loader["train"] = accelerator.prepare(self.model, self.optimizer, data_loader["train"])
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        if self.config.TOOLBOX_MODE == "multi_train_and_test":
            key_dict = []
            for i in range(1, self.config.TRAIN.DATA.MULTI_SOURCE.NUM_SOURCE + 1):
                key_dict.append("train" + str(i))
            iterators = {
                key: iter(data_loader[key])
                for key in key_dict
            }
            max_len = min(len(data_loader[key]) for key in key_dict)
            for epoch in range(self.max_epoch_num):
                self.current_epoch = epoch + 1
                
                # 检查是否需要切换到Stage 2
                if self.two_stage_training and epoch == self.stage1_epochs:
                    self._switch_to_stage2()
                
                # 打印当前训练阶段信息
                stage_info = f" [Stage {self.current_stage}]" if self.two_stage_training else ""
                print(f"\n==== Training Epoch: {epoch + 1}/{self.max_epoch_num}{stage_info} ====")
                self.model.train()
                epoch_losses = {key: [] for key in key_dict}
                running_losses = {key: 0.0 for key in key_dict}

                with tqdm(total=max_len, ncols=100) as tbar:
                    for idx in range(max_len):
                        self.optimizer.zero_grad()
                        total_loss = 0.0
                        current_losses = {}
                        # features = []
                        for dataset_key in key_dict:
                            try:
                                batch = next(iterators[dataset_key])
                            except StopIteration:
                                iterators[dataset_key] = iter(data_loader[dataset_key])
                                batch = next(iterators[dataset_key])
                            # 获取输入数据和标签
                            facial_video = batch[0].float().to(self.device)  # 面部视频数据

                            labels = batch[1].float().to(self.device)  # rPPG 信号标签
                            facial_video_delete = batch[-2].float().to(self.device)
                            strength = batch[-1]
                            # print("facial_video", facial_video.shape)
                            # print("facial_video_delete", facial_video_delete.shape)
                            # print("======================================================")
                            # print("数据位置:",facial_video.device)
                            #print("con_encoder位置:",self.con_encoder.device)

                            self.optimizer.zero_grad()

                            # 前向传播
                            # with autocast():
                            # prompt = self.con_encoder(facial_video, strength, batch[2], batch[3], self.config.TRAIN.DATA.DATASET)
                            # 新模型不需要prompt参数
                            prediction = self.model(facial_video)
                            prediction = (prediction-torch.mean(prediction, axis=-1).view(-1, 1))/torch.std(prediction, axis=-1).view(-1, 1)    # normalize
                            labels = labels - torch.mean(labels, dim=-1, keepdim=True)
                            labels = labels / torch.std(labels, dim=-1, keepdim=True)
                            # 计算损失
                            # loss = weighted_mse_loss(prediction, labels)
                            loss1 = self.criterion1(prediction, labels)
                            loss2 = self.criterion3(prediction, labels)
                            loss = loss1 
                            # print("")
                            # print("loss:", loss.item(), "mse:", loss1.item(), "fre:", loss2.item()/10)
                            # TODO: 加入频域损失
                            # pred_fft = torch.fft.rfft(prediction, dim=-1)
                            # target_fft = torch.fft.rfft(labels, dim=-1) 
                            # pred_magnitude = torch.abs(pred_fft)  # 幅值谱
                            # target_magnitude = torch.abs(target_fft)
                            # freq_loss = self.criterion(pred_magnitude, target_magnitude)
                            # loss = 1 * loss + 1 * freq_loss / 50
                            total_loss += loss
                            current_losses[dataset_key] = loss.item()
                            epoch_losses[dataset_key].append(loss.item())
                            running_losses[dataset_key] += loss.item()

                        # 反向传播和优化
                        # accelerator.backward(loss)
                        total_loss.backward()
                        # scaler.scale(loss).backward()
                        # 学习率调整
                        # lrs.append(self.scheduler.get_last_lr())
                        self.optimizer.step()
                        # self.scheduler.step()
                        self.optimizer.zero_grad()
                        # scaler.step(self.optimizer)
                        # scaler.update()

                        # epoch_losses.append(loss.item())
                        # running_loss += loss.item()

                        # # 每隔一定步数，更新一次平均损失
                        # avg_loss = running_loss / (idx + 1) 
                        # tbar.set_postfix(loss=f"{avg_loss:.4f}")
                        # # if idx % 100 == 99:
                        #     # print(f"Epoch [{epoch}/{self.max_epoch_num}], Step [{idx+1}/{len(data_loader['train'])}], Loss: {np.mean(epoch_losses[-100:]):.4f}")
                        avg_losses = {
                            key: running_losses[key] / (idx + 1)
                            for key in running_losses
                        }
                        tbar.set_postfix(**{
                            f"loss_{k}": f"{v:.4f}"
                            for k, v in avg_losses.items()
                        })
                        tbar.update(1)
                mean_training_losses.append(total_loss)
                # self.scheduler.step()

                # 保存模型
                self.epoch = epoch
                self.save_model(epoch)

                # 验证模型
                if not self.config.TEST.USE_LAST_EPOCH:
                    valid_loss = self.valid(data_loader)
                    mean_valid_losses.append(valid_loss)
                    print(f"Validation Loss: {valid_loss:.4f}")
                    if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        print(f"Update best model! Best epoch: {self.best_epoch}")

                self.test(data_loader)
        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"Best trained epoch: {self.best_epoch}, Min validation loss: {self.min_valid_loss:.4f}")

        # 绘制损失和学习率曲线（可选）
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """Validation routine for PhysLLM model."""
        if data_loader["valid"] is None:
            raise ValueError("No data for validation.")

        print("\n==== Validating ====")
        self.model.eval()
        valid_losses = []

        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=80)
            for idx, batch in enumerate(tbar):
                batch_size = batch[0].size(0)
                facial_video = batch[0].float().to(self.device)
                label = batch[1].float().to(self.device)
                facial_video_delete = batch[-2].float().to(self.device)
                strength = batch[-1]
                # 前向传播
                # prompt = self.con_encoder(facial_video_delete, strength, batch[2], batch[3], self.config.TEST.DATA.DATASET)
                # 新模型不需要prompt参数
                prediction = self.model(facial_video)
                prediction = (prediction-torch.mean(prediction, axis=-1).view(-1, 1))/torch.std(prediction, axis=-1).view(-1, 1)    # normalize
                label = label - torch.mean(label, dim=-1, keepdim=True)
                label = label / torch.std(label, dim=-1, keepdim=True)

                # 计算损失
                loss = self.criterion1(prediction, label)
                # loss1 = self.criterion1(prediction, label)
                # loss2 = self.criterion3(prediction, label)
                # loss = loss1 + loss2 / 22
                valid_losses.append(loss.item())

        mean_valid_loss = np.mean(valid_losses)
        return mean_valid_loss

    def test(self, data_loader):
        """Testing routine for PhysLLM model."""
        if data_loader["test"] is None:
            raise ValueError("No data for testing.")

        print("\n==== Testing ====")

        # 加载模型
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your config.")
            # self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location='cpu'))
            self.load_model(self.config.INFERENCE.MODEL_PATH)
            self.model.to(self.device)
            print(f"Testing uses pretrained model: {self.config.INFERENCE.MODEL_PATH}")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                folder_path = os.path.join(self.model_dir, self.model_folder)
                model_path = os.path.join(folder_path, self.model_file_name + f'{self.epoch}.pth')
                self.load_model(model_path)
                self.model.to(self.device)
                print(f"Testing uses last epoch model: {model_path}")
            else:
                folder_path = os.path.join(self.model_dir, self.model_folder)
                model_path = os.path.join(folder_path, self.model_file_name + f'{self.epoch}.pth')
                self.load_model(model_path)
                print(f"Testing uses best epoch model: {model_path}")

        self.model.eval()
        predictions = dict()
        labels = dict()
        baselines = dict()

        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=80)
            for idx, batch in enumerate(tbar):
                batch_size = batch[0].size(0)
                facial_video = batch[0].float().to(self.device)
                label = batch[1].float().to(self.device)
                facial_video_delete = batch[-2].float().to(self.device)
                strength = batch[-1]
                # 前向传播
                # prompt = self.con_encoder(facial_video_delete, strength, batch[2], batch[3], self.config.TEST.DATA.DATASET)
                # 新模型不需要prompt参数
                prediction = self.model(facial_video)
                prediction = (prediction-torch.mean(prediction, axis=-1).view(-1, 1))/torch.std(prediction, axis=-1).view(-1, 1)    # normalize
                label = label - torch.mean(label, dim=-1, keepdim=True)
                label = label / torch.std(label, dim=-1, keepdim=True)

                data_dict = {
                    'ground_truth': label.cpu(),
                    'rppg': prediction.cpu()
                }
                debug_dir = os.path.join(
                    self.config.TEST.OUTPUT_SAVE_DIR or os.path.join(self.model_dir, self.model_folder),
                    "debug_sequences",
                )
                os.makedirs(debug_dir, exist_ok=True)
                np.save(os.path.join(debug_dir, f"{idx}_sequences.npy"), data_dict)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    prediction = prediction.cpu()
                    # baseline_ppg_test = _.cpu()

                for idx in range(batch_size):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                        # baselines[subj_index] = dict()
                    predictions[subj_index][sort_index] = prediction[idx]
                    labels[subj_index][sort_index] = label[idx]
                    # baselines[subj_index][sort_index] = baseline_ppg_test[idx]

        # 计算评价指标
        # if not self.rppg_beseline_tested and self.model.module.input_type == 'rPPG_sequence':
        #     baseline_metrics_dict = calculate_metrics(baselines, labels, self.config)
        #     save_metrics_to_file(baseline_metrics_dict, self.config, 0, "baseline")
        #     self.rppg_beseline_tested = True
        PhysLLM_metrics_dict = calculate_metrics(predictions, labels, self.config)
        comment = "self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS)" # steps_per_epoch=self.num_train_batches
        # comment = "self.scheduler =optim.lr_scheduler.StepLR(self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)"
        save_metrics_to_file(PhysLLM_metrics_dict, self.config, self.epoch, comment)
        # plot_view_epoch(predictions, labels, self.config, self.epoch)


        # 保存测试输出（可选）
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    # def save_model(self, epoch):
    #     if not os.path.exists(self.model_dir):
    #         os.makedirs(self.model_dir)
    #     model_path = os.path.join(self.model_dir, self.model_file_name + f'.pth')
    #     torch.save(self.model.state_dict(), model_path)
    #     print(f'Saved Model Path: {model_path}')

    def save_model(self, epoch):
        folder_path = os.path.join(self.model_dir, self.model_folder)
        
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, self.model_file_name + f'{epoch}.pth')
        
        # 只保存可训练参数
        trainable_state_dict = {name: param.cpu() for name, param in self.model.named_parameters() if param.requires_grad}
        torch.save(trainable_state_dict, model_path)
        print(f'Saved Model Path: {model_path}')


    def get_hr(self, signal, sr=30, min_hr=40, max_hr=180):
        """
        Calculate heart rate from rPPG signal using Welch's method.
        """
        from scipy.signal import welch

        freqs, psd = welch(signal, fs=sr, nperseg=min(len(signal), 256))
        valid_range = (freqs >= min_hr / 60) & (freqs <= max_hr / 60)
        valid_freqs = freqs[valid_range]
        valid_psd = psd[valid_range]
        if len(valid_psd) == 0:
            return 0
        peak_freq = valid_freqs[np.argmax(valid_psd)]
        hr = peak_freq * 60
        return hr


    def load_model(self, model_path=None):
        folder_path = os.path.join(self.model_dir, )
        if model_path is None:
            model_path = os.path.join(folder_path, self.model_file_name + f'{self.epoch}.pth')
            
        # 加载保存的可训练参数
        trainable_state_dict = torch.load(model_path, map_location='cpu')
        
        # 获取模型的当前状态字典
        model_dict = self.model.state_dict()
        
        # 更新模型的状态字典
        model_dict.update(trainable_state_dict)
        self.model.load_state_dict(model_dict)
        print(f"Loaded trainable parameters from: {model_path}")

    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"{total_params:,}", f"{trainable_params:,}"
    

def weighted_mse_loss(predictions, targets, weight_type="amplitude"):
    """
    计算加权MSE损失
    :param predictions: 模型预测值 (Tensor, shape=[batch_size, seq_len])
    :param targets: 真实值 (Tensor, shape=[batch_size, seq_len])
    :param weight_type: 权重计算方式 ("amplitude" or "gradient")
    :return: 加权MSE损失
    """
    # 计算误差
    errors = predictions - targets
    
    # 根据权重类型计算权重
    if weight_type == "amplitude":
        # 使用真实信号的幅值作为权重
        weights = torch.abs(targets)
    elif weight_type == "gradient":
        # 使用信号梯度（差分值的绝对值）作为权重
        weights = torch.abs(torch.diff(targets, dim=1, append=targets[:, -1:]))
    else:
        raise ValueError("Unsupported weight_type. Use 'amplitude' or 'gradient'")
    
    # 确保权重非负且有正常的范围
    weights = weights + 1e-8  # 避免出现0权重
    weights = weights / weights.mean()  # 归一化权重
    
    # 加权MSE损失
    weighted_mse = torch.mean(weights * (errors ** 2))
    
    return weighted_mse


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        # 初始化可训练的时域损失权重 (alpha)
        self.alpha = nn.Parameter(torch.full((1,), 0.5))  # 初始值为0.5
        self.mse = torch.nn.MSELoss()
        # 可以视需求加入频域损失权重 (beta = 1 - alpha)
    
    def forward(self, predictions, targets):
        """
        计算结合时域和频域的损失
        :param predictions: 模型预测值 (Tensor)
        :param targets: 真实值 (Tensor)
        :return: 综合损失值
        """
        # 时域损失 (MSE)
        time_loss = self.mse(predictions, targets)
        
        # 频域损失
        freq_loss = self.frequency_loss(predictions, targets)
        
        # 将 alpha 限制在 [0, 1] 范围内
        alpha_clamped = torch.clamp(self.alpha, min=0.0, max=1.0).to(predictions.device)
        
        # 计算综合损失
        total_loss = alpha_clamped * time_loss + (1 - alpha_clamped) * freq_loss
        return total_loss
    
    @staticmethod
    def frequency_loss(predictions, targets):
        """
        计算频域损失
        :param predictions: 模型预测值 (Tensor)
        :param targets: 真实值 (Tensor)
        :return: 频域损失值
        """
        # 快速傅里叶变换 (FFT)
        pred_fft = torch.fft.fft(predictions, dim=1)
        target_fft = torch.fft.fft(targets, dim=1)
        
        # 计算幅值频谱
        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)
        
        # 计算频域损失 (L2 范数)
        mse = torch.nn.MSELoss()
        loss = mse(pred_magnitude, target_magnitude)
        return loss


class CrossCorrelationLoss(nn.Module):
    def __init__(self):
        super(CrossCorrelationLoss, self).__init__()

    def forward(self, preds, labels):
        """
        preds: 预测信号 (batch_size, seq_len)
        labels: 标签信号 (batch_size, seq_len)
        """
        # 去均值
        preds_mean = preds - preds.mean(dim=1, keepdim=True)  # 对每个样本去均值
        labels_mean = labels - labels.mean(dim=1, keepdim=True)
        
        # 分子：计算内积
        numerator = torch.sum(preds_mean * labels_mean, dim=1)
        
        # 分母：归一化因子 (L2 范数乘积)
        denominator = torch.sqrt(torch.sum(preds_mean**2, dim=1)) * torch.sqrt(torch.sum(labels_mean**2, dim=1))
        
        # 防止分母为零
        denominator = denominator + 1e-8
        
        # 计算归一化相关系数
        correlation = numerator / denominator  # 每个样本的相关性
        
        # 损失：1 - 平均相关系数
        loss = 1 - correlation.mean()  # 损失是 1 减去平均相关性
        return loss
