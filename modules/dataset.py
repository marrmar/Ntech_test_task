from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np


class Male_Dataset(Dataset):
    def __init__(self, path, mode, transforms=None, resize_size=112):
        """
        Создает класс датасет.
        :param path: путь до файла с именами картинок
        :param mode: 'train', 'val', 'test'
        :param transforms: преобразования над картинками
        :param resize_size: пространственный размер картинок при входе в сеть
        """
        super().__init__()
        # загружаем файл с именами файлов
        self._files = []
        with open(path, 'r') as f:
            for line in f:
                self._files.append(line[:-1])

        self._transforms = transforms
        self._resize_size = resize_size

        self._mode = mode
        if self._mode not in ['train', 'val', 'test']:
            raise NameError(f"{self._mode} is not correct; correct modes: 'train', 'val', 'test'")

        self._len = len(self._files)

        self._label_encoder = {'male': 1, 'female': 0}

        if self._mode != 'test':
            self._labels = [self._label_encoder[Path(path).parent.name] for path in self._files]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        """
        :param index: номер
        :return: (картинку, ответ)
        """
        image = cv2.imread(self._files[index])
        image = self._prepare_sample(image, self._resize_size)
        image = np.array(image / 255, dtype='float32')

        if self._transforms:
            image = self._transforms(image)
        if self._mode == 'test':
            return image
        else:
            label = self._labels[index]
            return image, label

    def _prepare_sample(self, image, resize_size):
        """
        :param image: картинка
        :param resize_size: размер до которого надо преобразовать
        :return: преобразованная картинка
        """
        image = cv2.resize(image, (resize_size, resize_size))
        return np.array(image)
