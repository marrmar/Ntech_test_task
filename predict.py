from modules.simple_net import SimpleCnn
import torch
from pathlib import Path
import cv2
from modules.predict_scripts import predict_one_sample
import json
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, required=True, help='path to images')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='checkpoints path')
    args = parser.parse_args()
    data_path = args.images_path
    checkpoint_path = args.checkpoint_path

    LABELS = {0: 'female', 1: 'male'}
    files = [*map(str, list(Path(data_path).rglob('*.jpg')))]

    model = SimpleCnn()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    predictions={}
    for path in tqdm(files):
        image = cv2.imread(path)
        predictions[path[path.rfind('\\')+1:]] = LABELS[predict_one_sample(model, image)]

    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)