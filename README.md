# big_data2

Ссылка на датасет: https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales/data

Для запуска, необходимо склонировать репозиторий и запустить run.sh скрипт:

 - git clone https://github.com/Den-Shab/big_data2.git
 - cd big_data2
 - положить файл data.csv в ./big_data2
 - ./run.sh

Данный скрипт автоматически поднимает контейнер, устанавливает необходимые зависимости, а также запускает все 4 тест-кейса и сохраняет результаты работы программы

# Результаты работы
Данные замеры производились на части датасета, состоящей из 1300000 строк.

Из результатов, представленных ниже, следуют вполне логичные выводы: самым медленным оказался подход с 1 DataNode-ой и без оптимизаций, небольшое ускорение дало добавление оптимизаций. Переход же к 3 DataNode-ам оказал больший эффект, и время сократилось значительнее. При этом, подход без оптимизаций оказался эффективнее.

![image](https://github.com/user-attachments/assets/702f4c22-5c25-401f-8522-e244344aba60)
![image](https://github.com/user-attachments/assets/f11b89e8-3406-43e0-b5c9-b8ed5fda9c83)

Следующий эксперимент производился на части датасета, состоящей из 130000 строк.

В целом, результаты замеров времени оказались схожими с первым экспериментом, и лишь по памяти появились различия, связанные, как раз таки, с объемом входных данных.

![image](https://github.com/user-attachments/assets/8b7b57e5-f13e-4e4a-b675-0c9e5fb2dfa8)
![memory_usage](https://github.com/user-attachments/assets/0a9b847a-a07b-4f28-bcce-816c31ab3b26)


