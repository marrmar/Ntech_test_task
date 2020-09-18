# Тестовое задание

В папке problem_1 находится решение первой задачи, дальше речь пойдет про вторую.

Задание: Предсказывать пол человека по фотографии.

## Установить зависимости:

~~~
pip install -r requirements.txt
~~~

## Чтобы запустить тренировку надо:

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

## Предсказать для папки с картинками
~~~
predict.py [-h] --images_path IMAGES_PATH --checkpoint_path
                 CHECKPOINT_PATH
~~~

Дотренировать до логического завершения на собственном ноутбуке не получается в связи с малой мощностью, но на последней эпохе accuracy на валидационной выборке: **0.9684**. И кажется, может тренироваться дальше.

## Скачать чекпойнт
[Можно здесь](
https://drive.google.com/file/d/1cXFTlJKrPN7FaEDOHUH9FouwysUw7ZkF/view?usp=sharing)

