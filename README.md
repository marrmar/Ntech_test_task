# Тестовое задание

В папке problem_1 находится решение первой задачи, дальше речь пойдет про вторую.

Задание: Предсказывать пол человека по фотографии.

За основу взята сетка DenseNet [отсюда](https://github.com/andreasveit/densenet-pytorch). Побавлен 4 блок(как для densenet на imagenet). В каждом Dense блоке по 4 свертки(точнее по 8, так как перд каждой стоит свертка 1x1 для уменьшения количества вычислений). Между dense блоками стоят transition блоки с уменьшением количества каналов в 2 раза(светка 1x1) и average polling(2x2). Перед dense блоками свертка 3x3 и average polling(2x2) для уменьшения пространственной размерности.

DenseNet выбран за качетво работы и относительную легкость(что немаловажно при обучении не на мощной машине).

Картинки подавались в сеть размером 112х112.

Тренировалась с Adam, lr=0.001. Сеть поддреживает включение MultiStepLR c DROP_AFTER_EPOCH параметром.

В качестве аугументации использовался только горизонтальный flipс вероятснотью 0.5 и нормализация, потому что датасет достаточно большой.

Отслеживание экспериментов возможно в tensorboard.

## Установить зависимости:

~~~
pip install -r requirements.txt
~~~

## Для тренировки:

### Сгенировать тренировочную и валидационную выборки

~~~
make_val.py [-h] --path PATH [--percentage PERCENTAGE]
~~~


### Запустить тренировку

~~~
train.py [-h] --path PATH --epochs EPOCHS [--batch_size BATCH_SIZE]
               [--resize_size RESIZE_SIZE] [--lr LR]
               [--start_from_checkpoint START_FROM_CHECKPOINT]
               [--weights_only WEIGHTS_ONLY]
               [--drop_after_epoch DROP_AFTER_EPOCH]
               [--checkpoint_path CHECKPOINT_PATH]
~~~

## Предсказать для папки с картинками:
~~~
predict.py [-h] --images_path IMAGES_PATH --checkpoint_path
                 CHECKPOINT_PATH
~~~

Дотренировать до логического завершения на собственном ноутбуке не получается в связи с малой мощностью, но на последней эпохе accuracy на валидационной выборке: **0.9684**. И кажется, может тренироваться дальше.

## Скачать чекпойнт
[Можно здесь](
https://drive.google.com/file/d/1cXFTlJKrPN7FaEDOHUH9FouwysUw7ZkF/view?usp=sharing)

