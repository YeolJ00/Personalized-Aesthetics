from PIL import Image
import torch

import utils.parser as parser
from models.iaa import MultiModalIAA
from dataset import DEFAULT_TRANSFORM

opt = parser.parse_args()

device = torch.device(opt.device)

# Load models
model = MultiModalIAA(opt, device)

# model_path = "./work_dir/clip_L_3fc_ava.pth"
model_path = "./work_dir/clip_L_3fc_aes.pth"
# model_path = "./work_dir/clip_L_3fc_tad.pth"
# model_path = "./work_dir/clip_L_3fc_para.pth"
# model_path = "./work_dir/clip_L_3fc_koniq.pth"
# model_path = "./work_dir/clip_L_3fc_spaq.pth"

print("Loading checkpoint from {}".format(model_path))
state_dict = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state_dict=state_dict, strict=True)

model.to(device)
model.eval()

# open image
img_path = "./assets/nice_image.jpg"
# img_path = "./assets/not_so_nice_image.jpg"
img = Image.open(img_path).convert('RGB')
img = DEFAULT_TRANSFORM(img).unsqueeze(0).to(device)

# forward
pred = model({'img': img}).squeeze(0)
template = torch.arange(1, 11, dtype=torch.float32).to(device)
score = pred@template

print("Predicted score: ", f"{score.item():.2f}")