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

*Pre-trained model checkpoints will be available soon. Stay tuned for updates!*
<!-- ```bash
git lfs
``` -->
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
  author={Jooyeol Yun and Jaegul Choo},
  booktitle={ECCV},
  year={2024},
  url={https://arxiv.org/abs/2407.07176}
}
```