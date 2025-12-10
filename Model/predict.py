import torch
from PIL.ImagePalette import random
import  random

from fairscale.internal.containers import from_np

from newzfnet import  ZFNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img2 = Image.open(os.path.join(folder, filename))
        img2 = img2.resize((224, 224))  # 你可以根据需要调整图片大小
        img2 = np.array(img2)
        images.append(img2)

    images = np.array(images)
    return images


# 指定包含图片的文件夹路径
folder_path = r"C:\Users\shys\Desktop\zm\dl\Data\test\1"
photo_extensions = ('.jpg', '.jpeg', '.png', '.gif')

# 使用列表推导式提取所有照片的绝对路径
photo_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(folder_path)
    for file in files
    if file.lower().endswith(photo_extensions)
]
# 加载图片数据集
image_dataset = load_images_from_folder(folder_path)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
for i in range(565):
    # load image
    #ss=random.randint(1,2712)


    img = image_dataset[i]
    img = cv2.resize(img, (224, 224))
    plt.imshow(img)

    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # load model weights
    # create model
    model = ZFNet(num_classes=2)
    model_weight_path = "./CNN.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)

    predict_cla = torch.argmax(predict).numpy()
    print(i,class_indict[str(predict_cla)],"accurate:" ,predict[predict_cla].item())
    plt.text(65, -5, f'Class: {class_indict[str(predict_cla)]}\nProb: {predict[predict_cla].item():.10f}', fontsize=12,
             color='white', bbox=dict(facecolor='blue', alpha=0.5))
    data = f'{photo_paths[i]}   Class: {class_indict[str(predict_cla)]}   Prob: {predict[predict_cla].item():.10f}\n'
    with open(r"C:\Users\shys\Desktop\CNN.txt", 'a', encoding='utf-8') as file:
        file.write(data)

    plt.show()

    i=i+1

