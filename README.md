# Scaling Up Personalized Image Aesthetic Assessment via Task Vector Customization [ECCV 2024]

>[Scaling Up Personalized Image Aesthetic Assessment via Task Vector Customization](https://arxiv.org/abs/2407.07176) [ECCV 2024]
>
>Jooyeol Yun and Jaegul Choo
>
>[Project Page](https://yeolj00.github.io/personal-projects/personalized-aesthetics/)
>

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA 11.0 or higher (for GPU acceleration)

### Installation

#### Clone this repository:

```bash
git clone https://github.com/YeolJ00/Personalized-Aesthetics.git
cd Personalized-Aesthetics
```

#### Download Aesthetic Assessment Models

__[2025/03/10]__ New links for pre-trained models are out! 
We have found a new way to safely host the files and now support permanent share links. 
Please use the link below to download the pre-trained models. 
You no longer have to submit a request. 


> Download link: [Link](https://gofile.me/6WpIS/pU1qTO1Dc) 
>
> Password: yeoljoo


__[2024/10/22]__ Pre-trained model checkpoints are now available! 
Unfortunately, due to security constraints, we are hosting the files on a private cloud service that does not support permanent share links. 
If you would like access, please submit a request through [this Microsoft form](https://forms.office.com/r/vyxtBDcdcZ), and we will send you a temporary download link.

~~*Pre-trained model checkpoints will be available soon. Stay tuned for updates!*~~

### Inference

You can easily use our six pre-trained aesthetic scoring models to assess image aesthetics. Below is a Python example using one of the models:

```python
from PIL import Image
import torch

import utils.parser as parser
from models.iaa import MultiModalIAA
from dataset import DEFAULT_TRANSFORM

opt = parser.parse_args()
device = torch.device(opt.device)

# Load models
model = MultiModalIAA(opt, device)

# choose any model
model_path = "./checkpoints/clip_L_3fc_aes.pth"

print("Loading checkpoint from {}".format(model_path))
state_dict = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state_dict=state_dict, strict=True)

model.to(device)
model.eval()

# open image
img_path = "./assets/nice_image.jpg"
img = Image.open(img_path).convert('RGB')
img = DEFAULT_TRANSFORM(img).unsqueeze(0).to(device)

# forward
pred = model({'img': img}).squeeze(0)
template = torch.arange(1, 11, dtype=torch.float32).to(device)
score = pred@template

print("Predicted score: ", f"{score.item():.2f}")
```

### Training

For training and evaluating on PIAA datasets, please refer to the shell script we provide.

```bash
bash train_piaa.sh
```

## Citation
```bibtex
@inproceedings{yun2024scaling,
  title={Scaling Up Personalized Image Aesthetic Assessment via Task Vector Customization},
  author={Yun, Jooyeol and Choo, Jaegul},
  booktitle={ECCV},
  year={2024},
  url={https://arxiv.org/abs/2407.07176}
}
```
