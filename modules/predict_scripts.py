from torchvision import transforms
import cv2
import numpy as np


def predict_one_sample(model, image, resize_size=112):
    image = cv2.resize(image, (resize_size, resize_size))
    tensor_image = transforms.functional.to_tensor(image)
    norm_image = transforms.functional.normalize(tensor_image, mean=[0.3767, 0.4692, 0.6215],
                                                 std=[0.2218, 0.2273, 0.2501]).unsqueeze(0)

    outputs = np.squeeze(model(norm_image))
    pred = int(outputs > 0.)
    return pred
