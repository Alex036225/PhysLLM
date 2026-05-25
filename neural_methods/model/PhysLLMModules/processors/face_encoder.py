from neural_methods.model.PhysLLMModules.facexformer import FaceXFormer
from .abstract_base import EncoderBase, resolve_checkpoint_path
import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

class FaceXFormerEncoder(EncoderBase):
    def __init__(self, configs) -> None:
        super().__init__()
        checkpoint_path = resolve_checkpoint_path(
            configs, "FACE_ENCODER", "PHYSLLM_FACE_ENCODER_CKPT"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model = FaceXFormer().float()
        state_dict = checkpoint.get('state_dict_backbone', checkpoint)
        self.model.load_state_dict(state_dict)
        # 将模型设置为评估模式并冻结参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.cuda()

        self.attributes = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
            'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
            'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
            'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
            'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
            'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
            'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
            ]
        self.attr_descriptions = {
            '5_o_Clock_Shadow': 'has a 5 o\'clock shadow',
            'Arched_Eyebrows': 'has arched eyebrows',
            'Attractive': 'is attractive',
            'Bags_Under_Eyes': 'has bags under the eyes',
            'Bald': 'is bald',
            'Bangs': 'has bangs',
            'Big_Lips': 'has big lips',
            'Big_Nose': 'has a big nose',
            'Black_Hair': 'has black hair',
            'Blond_Hair': 'has blond hair',
            'Blurry': 'the image is blurry',
            'Brown_Hair': 'has brown hair',
            'Bushy_Eyebrows': 'has bushy eyebrows',
            'Chubby': 'is chubby',
            'Double_Chin': 'has a double chin',
            'Eyeglasses': 'is wearing eyeglasses',
            'Goatee': 'has a goatee',
            'Gray_Hair': 'has gray hair',
            'Heavy_Makeup': 'is wearing heavy makeup',
            'High_Cheekbones': 'has high cheekbones',
            'Male': 'is male',
            'Mouth_Slightly_Open': 'has mouth slightly open',
            'Mustache': 'has a mustache',
            'Narrow_Eyes': 'has narrow eyes',
            'No_Beard': 'has no beard',
            'Oval_Face': 'has an oval face',
            'Pale_Skin': 'has pale skin',
            'Pointy_Nose': 'has a pointy nose',
            'Receding_Hairline': 'has a receding hairline',
            'Rosy_Cheeks': 'has rosy cheeks',
            'Sideburns': 'has sideburns',
            'Smiling': 'is smiling',
            'Straight_Hair': 'has straight hair',
            'Wavy_Hair': 'has wavy hair',
            'Wearing_Earrings': 'is wearing earrings',
            'Wearing_Hat': 'is wearing a hat',
            'Wearing_Lipstick': 'is wearing lipstick',
            'Wearing_Necklace': 'is wearing a necklace',
            'Wearing_Necktie': 'is wearing a necktie',
            'Young': 'is young'
            }
    
    def encode(self, x):
        B, C, T, H, W = x.shape
        mid_frame = T // 2
        face = x[:, :, mid_frame, ...]
        face = TF.resize(face, size=[224, 224], interpolation=InterpolationMode.BICUBIC)
        face = TF.normalize(face, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data = {'image': face, 'label': {"segmentation":torch.zeros([224,224]), "lnm_seg": torch.zeros([5, 2]),"landmark": torch.zeros([68, 2]), "headpose": torch.zeros([3]), "attribute": torch.zeros([40]), "a_g_e": torch.zeros([3]), 'visibility': torch.zeros([29])}, 'task': torch.tensor([3])}
        labels, task = data['label'], data['task']
        for k in labels.keys():
            labels[k] = labels[k].unsqueeze(0).to(device=x.device)
        task = task.to(device=x.device)
        facial_description = []
        for b in range(B): 
            landmark_output, headpose_output, facial_attributes, visibility_output, age_output, gender_output, race_output, seg_output = self.model(face[b].unsqueeze(0), labels, task)
            facial_description_ = self.face_attribute2text(facial_attributes)
            facial_description.append(facial_description_)
        
        return facial_description
    
    def face_attribute2text(self, facial_attributes):
        """
        process facial_model output as CelebA dataset
        """
        probs = torch.sigmoid(facial_attributes[0])
        preds = (probs >= 0.5).float()
        pred = preds.tolist()
        description_parts = []

        is_male = bool(pred[self.attributes.index('Male')])
        if is_male:
            pronoun = 'He'
            possessive_pronoun = 'his'
        else:
            pronoun = 'She'
            possessive_pronoun = 'her'
        # Essential descriptors
        # Essential descriptors
        if is_male:
            description_parts.append('a man')
        else:
            description_parts.append('a woman')

        # Age
        if pred[self.attributes.index('Young')]:
            description_parts.append('who is young')
        else:
            description_parts.append('who is middle-aged or older')

        # Hair color
        hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
        hair_color = None
        for color in hair_colors:
            if pred[self.attributes.index(color)]:
                hair_color = self.attr_descriptions[color]
                break
        if hair_color:
            description_parts.append(hair_color)

        # Bald
        if pred[self.attributes.index('Bald')]:
            description_parts.append('is bald')

        # Hair style
        hair_styles = []
        if pred[self.attributes.index('Bangs')]:
            hair_styles.append('has bangs')
        if pred[self.attributes.index('Receding_Hairline')]:
            hair_styles.append('has a receding hairline')
        if pred[self.attributes.index('Straight_Hair')]:
            hair_styles.append('has straight hair')
        if pred[self.attributes.index('Wavy_Hair')]:
            hair_styles.append('has wavy hair')

        if hair_styles:
            description_parts.extend(hair_styles)

        # Facial hair
        facial_hair = []
        if pred[self.attributes.index('Mustache')]:
            facial_hair.append('has a mustache')
        if pred[self.attributes.index('Goatee')]:
            facial_hair.append('has a goatee')
        if pred[self.attributes.index('Sideburns')]:
            facial_hair.append('has sideburns')
        if pred[self.attributes.index('5_o_Clock_Shadow')]:
            facial_hair.append('has a 5 o\'clock shadow')

        if facial_hair:
            description_parts.extend(facial_hair)
        else:
            if pred[self.attributes.index('No_Beard')]:
                description_parts.append('has no beard')

        # Accessories
        accessories = []
        if pred[self.attributes.index('Eyeglasses')]:
            accessories.append('is wearing eyeglasses')
        if pred[self.attributes.index('Wearing_Hat')]:
            accessories.append('is wearing a hat')
        if pred[self.attributes.index('Wearing_Earrings')]:
            accessories.append('is wearing earrings')
        if pred[self.attributes.index('Wearing_Necklace')]:
            accessories.append('is wearing a necklace')
        if pred[self.attributes.index('Wearing_Necktie')]:
            accessories.append('is wearing a necktie')

        if accessories:
            description_parts.extend(accessories)

        # Makeup
        makeup = []
        if pred[self.attributes.index('Wearing_Lipstick')]:
            makeup.append('is wearing lipstick')
        if pred[self.attributes.index('Heavy_Makeup')]:
            makeup.append('is wearing heavy makeup')

        if makeup:
            description_parts.extend(makeup)

        # Facial features
        features = []
        if pred[self.attributes.index('High_Cheekbones')]:
            features.append('has high cheekbones')
        if pred[self.attributes.index('Chubby')]:
            features.append('is chubby')
        if pred[self.attributes.index('Double_Chin')]:
            features.append('has a double chin')
        if pred[self.attributes.index('Pointy_Nose')]:
            features.append('has a pointy nose')
        if pred[self.attributes.index('Big_Nose')]:
            features.append('has a big nose')
        if pred[self.attributes.index('Big_Lips')]:
            features.append('has big lips')
        if pred[self.attributes.index('Rosy_Cheeks')]:
            features.append('has rosy cheeks')
        if pred[self.attributes.index('Pale_Skin')]:
            features.append('has pale skin')
        if pred[self.attributes.index('Bags_Under_Eyes')]:
            features.append('has bags under the eyes')
        if pred[self.attributes.index('Bushy_Eyebrows')]:
            features.append('has bushy eyebrows')
        if pred[self.attributes.index('Arched_Eyebrows')]:
            features.append('has arched eyebrows')
        if pred[self.attributes.index('Narrow_Eyes')]:
            features.append('has narrow eyes')
        if pred[self.attributes.index('Oval_Face')]:
            features.append('has an oval face')

        if features:
            description_parts.extend(features)

        # Expression
        if pred[self.attributes.index('Smiling')]:
            description_parts.append('is smiling')
        if pred[self.attributes.index('Mouth_Slightly_Open')]:
            description_parts.append('has mouth slightly open')

        # Overall attractiveness
        if pred[self.attributes.index('Attractive')]:
            description_parts.append('is attractive')

        # Blurry image
        if pred[self.attributes.index('Blurry')]:
            description_parts.append('the image is blurry')

        # Compose the final description
        description = f"This is {', '.join(description_parts)}."

        return description.capitalize()
