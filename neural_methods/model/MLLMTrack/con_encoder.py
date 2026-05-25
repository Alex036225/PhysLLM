import json
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from config import get_config
import os
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
)


class ContentProcessor(nn.Module):
    def __init__(self, configs):
        super(ContentProcessor, self).__init__()
        if configs.TRAIN.DATA.PREPROCESS.PROMPT_GENERATION.IS_PREPROCESS or configs.TEST.DATA.PREPROCESS.PROMPT_GENERATION.IS_PREPROCESS:
            llm_layers = 32
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 加载视觉编码器和语言模型
            if configs.MODEL.MLLM == 'LLaVA':
                try:
                    # 尝试加载 LLaVA 的 Processor
                    self.processor = AutoProcessor.from_pretrained(model_id)
                except EnvironmentError:
                    print("LLaVA processor not found locally, attempting to download from Hugging Face...")
                    try:
                        self.processor = AutoProcessor.from_pretrained(model_id, force_download=True)
                    except Exception as e:
                        raise RuntimeError(f"Failed to download LLaVA processor: {e}")

                try:
                    # 尝试加载 LLaVA 的模型
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        model_id,
                        # torch_dtype="float32",
                        torch_dtype=torch.float16,
                    ).to(self.device)
                except EnvironmentError:
                    print("LLaVA model not found locally, attempting to download from Hugging Face...")
                    try:
                        self.model = LlavaForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        force_download=True  # 强制从 Hugging Face 下载
                    )
                    except Exception as e:
                        raise RuntimeError(f"Failed to download LLaVA model: {e}")

                try:
                    # 尝试加载 LLaVA 的模型
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                except EnvironmentError:
                    print("LLaVA tokenizer not found locally, attempting to download from Hugging Face...")
                    try:
                        self.tokenizer =  AutoTokenizer.from_pretrained(
                        model_id,
                        force_download=True  # 强制从 Hugging Face 下载
                    ).to(self.device)
                    except Exception as e:
                        raise RuntimeError(f"Failed to download LLaVA tokenizer: {e}")
            # 任务prompt
            text = (
                "Describe the key features of the given facial image that are relevant for rPPG signal prediction:"
                "Identify and describe the appearance of key facial regions (forehead, cheeks, nose bridge)."
                "Summarize the distribution and variation of skin tones across these regions."
                "Highlight any artifacts such as uneven lighting, shadows, or reflections, and their potential impact on signal quality."
                "Provide insights into visible skin texture and vascular patterns that could affect rPPG extraction.")
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image"},
                    ],
                },
            ]
            self.prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            self.prompt_dict = {}
            self.do_prompt_preprocess(configs)
        else:
            with open(os.path.join(configs.TRAIN.DATA.PREPROCESS.PROMPT_GENERATION.PREPROCESS_PATH,'LLaVA_prompt.json'), 'r', encoding='utf-8') as json_file:
                self.prompt_dict = json.load(json_file)
            if configs.TRAIN.DATA.DATASET != configs.TEST.DATA.DATASET:
                with open(os.path.join(configs.TEST.DATA.PREPROCESS.PROMPT_GENERATION.PREPROCESS_PATH,
                                       'LLaVA_prompt.json'), 'r', encoding='utf-8') as json_file:
                    test_prompt = json.load(json_file)
                self.prompt_dict.update(test_prompt)


    def forward(self, data, strength, filename, chunk_id, dataset_name):
        B,_,_,_,_  = data.shape
        description = []
        for i in range(B):
            cur_file = dataset_name + filename[i] + "_input" + chunk_id[i] + ".npy"
            output_str = self.prompt_dict[cur_file]
            if strength[i] == 0:
                sunlight_describe = "We did not add any additional lighting intensity."
            else:
                sunlight_describe = " We add a %d-degree artificial lighting intensity to the video."%strength[i]
            # output_str += sunlight_describe
            # print(output_str)
            description.append(output_str)
        return description # 返回文本[list]


    def do_prompt_preprocess(self, configs):
        train_path = configs.TRAIN.DATA.CACHED_PATH #   ../../../..Raw../
        test_path = configs.TEST.DATA.CACHED_PATH
        train_dataset = configs.TRAIN.DATA.DATASET  # UBFC-rPPG / PURE / ...
        test_dataset = configs.TEST.DATA.DATASET

        if configs.TRAIN.DATA.PREPROCESS.PROMPT_GENERATION.IS_PREPROCESS:
            # 创建train数据集的prompt
            train_files = [f for f in os.listdir(train_path) if 'input' in f]
            for npy_file in tqdm(train_files, ncols=80):
                cur_item = train_dataset + npy_file.split(os.sep)[-1]
                if cur_item not in self.prompt_dict:
                    video = np.load(os.path.join(train_path, npy_file))
                    T, _, _, _ = video.shape
                    mid_frame = T // 2  # 选择视频中的中间帧
                    face = video[mid_frame, :, :, :]
                    face_tensor = torch.from_numpy(face).to('cuda')
                    description = self.get_prompt_llm(face_tensor)
                    # print(description)
                    self.prompt_dict[cur_item] = description
                    print(len(self.prompt_dict))
            with open(os.path.join(configs.TRAIN.DATA.PREPROCESS.PROMPT_GENERATION.PREPROCESS_PATH,'LLaVA_prompt.json'), 'w', encoding='utf-8') as json_file:
                json.dump(self.prompt_dict, json_file, ensure_ascii=False, indent=4)
        else:
            with open(os.path.join(configs.TRAIN.DATA.PREPROCESS.PROMPT_GENERATION.PREPROCESS_PATH,'LLaVA_prompt.json'), 'r', encoding='utf-8') as json_file:
                self.prompt_dict = json.load(json_file)

        # 创建test数据集的prompt
        if configs.TEST.DATA.PREPROCESS.PROMPT_GENERATION.IS_PREPROCESS and train_dataset != test_dataset:
            test_prompt = {}
            test_files = [f for f in os.listdir(test_path) if 'input' in f]
            for npy_file in tqdm(test_files, ncols=80):
                cur_item = test_dataset + npy_file.split(os.sep)[-1]
                if cur_item not in self.prompt_dict:
                    video = np.load(os.path.join(test_path, npy_file))
                    T, _, _, _ = video.shape
                    mid_frame = T // 2  # 选择视频中的中间帧
                    face = video[mid_frame, :, :, :]
                    face_tensor = torch.from_numpy(face).to('cuda')
                    description = self.get_prompt_llm(face_tensor)
                    # 如果不存在，则创建新的键值对
                    self.prompt_dict[cur_item] = description
                    test_prompt[cur_item] = description
                    print(len(self.prompt_dict))
            with open(os.path.join(configs.TEST.DATA.PREPROCESS.PROMPT_GENERATION.PREPROCESS_PATH,
                                   'LLaVA_prompt.json'), 'w', encoding='utf-8') as json_file:
                json.dump(test_prompt, json_file, ensure_ascii=False, indent=4)
        elif (not configs.TEST.DATA.PREPROCESS.PROMPT_GENERATION.IS_PREPROCESS) and train_dataset != test_dataset:
            with open(os.path.join(configs.TEST.DATA.PREPROCESS.PROMPT_GENERATION.PREPROCESS_PATH,'LLaVA_prompt.json'), 'r', encoding='utf-8') as json_file:
                test_prompt = json.load(json_file)
            self.prompt_dict.update(test_prompt)



    def get_prompt_llm(self, face):
        # data = torch.clamp(data, 0, 1)  # 强制归一化
        inputs = self.processor(images=face, text=self.prompt, return_tensors='pt').to(0, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        describe = self.processor.decode(output[0][:], skip_special_tokens=True)
        start_index = describe.find("ASSISTANT")
        output_str = describe[start_index:].strip()
        return output_str  # 返回文本[list]


    # def before_forward(self, data, strength):
    #     """
    #     1. 先将图像和文本输入 LLaVA 生成描述。
    #     2. 使用 BERT 模型将生成的描述嵌入为向量。
    #     """
    #     # 强制归一化 (后续更改)
    #     # def normalize_frame(data):
    #     #     # 计算每个视频每一帧的最小值和最大值
    #     #     min_val = data.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    #     #     max_val = data.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    #     #
    #     #     # 应用归一化公式
    #     #     normalized_data = (data - min_val) / (max_val - min_val)
    #     #
    #     #     return normalized_data
    #     # print(data)
    #     # data = normalize_frame(data)
    #     """
    #     如果这里输入的是diffNormalize的视频，会出现问题，需要把其值映射到0~1之间，但是如果进行强行映射，最后的结果如下：
    #     Please discribe the environment and the face of the person. ASSISTANT:
    #     The image is a black and white photo of a person's face. The face is the main focus of the photo, and there is no other visible information about the environment.
    #     The photo is in black and white, which adds a classic and timeless feel to the image.
    #
    #     """
    #     # data = torch.clamp(data, 0, 1)  # 强制归一化
    #     # 任务prompt
    #     text_input = "Please describe the environment and the face of the person."
    #
    #     B, C, T, H, W = data.shape
    #     mid_frame = T // 2  # 选择视频中的中间帧
    #     # face = data[:, :, mid_frame, :, :]  # 选择中间帧，形状为 [B, C, H, W]
    #     # face = data[:, :, mid_frame, :, :].unsqueeze(0)  # 添加 batch 维度
    #     # print("中间帧大小:",face.shape)  # 应该输出 [B, C, H, W]
    #     text = (
    #         "Describe the key features of the given facial image that are relevant for rPPG signal prediction:"
    #         "Identify and describe the appearance of key facial regions (forehead, cheeks, nose bridge)."
    #         "Summarize the distribution and variation of skin tones across these regions."
    #         "Highlight any artifacts such as uneven lighting, shadows, or reflections, and their potential impact on signal quality."
    #         "Provide insights into visible skin texture and vascular patterns that could affect rPPG extraction.")
    #     conversation = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": text},
    #                 {"type": "image"},
    #             ],
    #         },
    #     ]
    #     prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
    #     # 使用 LLaVA 处理器处理图像和文本数据
    #     # inputs = self.processor(images=face, text=text_input, return_tensors="pt")
    #     # print("处理后的图像大小:",inputs['pixel_values'].shape)  # 应该输出 [B, C, H, W]
    #     # batch?
    #     description = []
    #     for i in range(B):
    #         face = data[i, :, mid_frame, :, :]
    #         inputs = self.processor(images=face, text=prompt, return_tensors='pt').to(0, torch.float16)
    #         output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
    #         describe = self.processor.decode(output[0][:], skip_special_tokens=True)
    #         start_index = describe.find("ASSISTANT")
    #         output_str = describe[start_index:].strip()
    #         if strength[i] == 0:
    #             sunlight_describe = "We did not add any additional lighting intensity."
    #         else:
    #             sunlight_describe = " We add a %d-degree artificial lighting intensity to the video."%strength[i]
    #         output_str += sunlight_describe
    #         print(output_str)
    #         description.append(output_str)
    #     return description # 返回文本[list]




