import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from evaluation.metrics import calculate_metrics, save_metrics_to_file
# 导入 PhysLLM 模型
from neural_methods.model.PhysLLMModules.InputProcessor import RPPGInputProcessor
from neural_methods.model.Phys.PhysLLM import PhysLLM
from neural_methods.model.PhysLLMModules.facexformer import FaceXFormer
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator


class PhysLLMTrainer(BaseTrainer):
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
        self.num_train_batches = len(data_loader["train"])
        self.model_folder = f"{self.config.MODEL.FACE_MODEL}_{self.config.MODEL.ENV_MODEL}_{self.config.MODEL.VIDEO_ENC}"

        self.input_processor = RPPGInputProcessor(
            configs=config,
            face_model_type=config.MODEL.FACE_MODEL,
            env_model_type=config.MODEL.ENV_MODEL,
            video_encoder_type=config.MODEL.VIDEO_ENC
        )

        # 实例化 PhysLLM 模型
        print("====== PhysLLM Loading =====")
        self.model = PhysLLM(config, )
        print(self.model)
        print("====================================================================")
        print("layers in the model:")
        for name, param in self.model.named_parameters():
            print(f"Layer: {name}, Requires Grad: {param.requires_grad}")
        print("====== Loading Finished =====")
        print("After initialization:", next(self.model.parameters()).device)
        print("Model parameter count after initialization:", sum(p.numel() for p in self.model.parameters()))
        self.model = self.model.to('cuda')  # 将模型移到 GPU
        print("After moving to GPU:", next(self.model.parameters()).device)
        print("Model parameter count after moving to GPU:", sum(p.numel() for p in self.model.parameters()))
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))
        print("After DataParallel:", next(self.model.parameters()).device)  # 打印模型所在设备
        print("Model parameter count after DataParallel:", sum(p.numel() for p in self.model.parameters()))

        # 输出参数数量
        total_params_ip, trainable_params_ip = self.count_parameters(self.input_processor)
        total_params_model, trainable_params_model = self.count_parameters(self.model)
        print(f"Input Processor - Total: {total_params_ip}, Trainable: {trainable_params_ip}")
        print(f"Model - Total: {total_params_model}, Trainable: {trainable_params_model}")
        # 定义损失函数和优化器
        self.criterion = torch.nn.MSELoss()  # 根据需要选择合适的损失函数
        # TODO
        lora_params = [p for n, p in self.model.named_parameters() if "lora" in n]
        other_params = [p for n, p in self.model.named_parameters() if "lora" not in n]
        self.optimizer = torch.optim.AdamW(
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

    def train(self, data_loader):
        """Training routine for PhysLLM model."""
        if data_loader["train"] is None:
            raise ValueError("No data for training.")

        # 初始化 GradScaler
        # scaler = GradScaler()
        accelerator = Accelerator()
        self.model, self.optimizer, data_loader["train"] = accelerator.prepare(self.model, self.optimizer, data_loader["train"])
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            self.current_epoch = epoch + 1
            print(f"\n==== Training Epoch: {epoch + 1} ====")
            self.model.train()
            epoch_losses = []
            running_loss = 0.0

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                # 获取输入数据和标签
                facial_video = batch[0].float().to(self.device)  # 面部视频数据
                labels = batch[1].float().to(self.device)  # rPPG 信号标签

                self.optimizer.zero_grad()

                # 前向传播
                # with autocast():
                processed_data = self.input_processor(facial_video)
                predictions, _ = self.model(processed_data)
                # 计算损失
                loss = self.criterion(predictions, labels)
                # 反向传播和优化
                # accelerator.backward(loss)
                loss.backward()
                # scaler.scale(loss).backward()
                self.optimizer.step()
                # scaler.step(self.optimizer)
                # scaler.update()

                epoch_losses.append(loss.item())
                running_loss += loss.item()

                # 每隔一定步数，更新一次平均损失
                avg_loss = running_loss / (idx + 1) / facial_video.shape[0]
                tbar.set_postfix(loss=f"{avg_loss:.4f}")
                # if idx % 100 == 99:
                    # print(f"Epoch [{epoch}/{self.max_epoch_num}], Step [{idx+1}/{len(data_loader['train'])}], Loss: {np.mean(epoch_losses[-100:]):.4f}")

            # 学习率调整
            # lrs.append(self.scheduler.get_last_lr())
            mean_training_losses.append(np.mean(epoch_losses))
            # self.scheduler.step()

            # 保存模型
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
            vbar = tqdm(data_loader["valid"], ncols=80)
            for idx, batch in enumerate(vbar):
                data = batch[0].float().to(self.device)
                labels = batch[1].float().to(self.device)

                # 前向传播
                predictions = self.model(data)

                # 计算损失
                loss = self.criterion(predictions, labels)
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
                model_path = os.path.join(folder_path, self.model_file_name + f'.pth')
                self.load_model(model_path)
                self.model.to(self.device)
                print(f"Testing uses last epoch model: {model_path}")
            else:
                folder_path = os.path.join(self.model_dir, self.model_folder)
                model_path = os.path.join(folder_path, self.model_file_name + f'.pth')
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

                # 前向传播
                processed_data = self.input_processor(facial_video)
                pred_ppg_test, _ = self.model(processed_data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    baseline_ppg_test = _.cpu()

                for idx in range(batch_size):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                        # baselines[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]
                    # baselines[subj_index][sort_index] = baseline_ppg_test[idx]

        # 计算评价指标
        # if not self.rppg_beseline_tested and self.model.module.input_type == 'rPPG_sequence':
        #     baseline_metrics_dict = calculate_metrics(baselines, labels, self.config)
        #     save_metrics_to_file(baseline_metrics_dict, self.config, 0, "baseline")
        #     self.rppg_beseline_tested = True
        PhysLLM_metrics_dict = calculate_metrics(predictions, labels, self.config)
        comment = "self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)"
        # comment = "self.scheduler =optim.lr_scheduler.StepLR(self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)"
        save_metrics_to_file(PhysLLM_metrics_dict, self.config, self.current_epoch, comment)


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
        model_path = os.path.join(folder_path, self.model_file_name + f'.pth')
        
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
            model_path = os.path.join(folder_path, self.model_file_name + f'.pth')
            
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