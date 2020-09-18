from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path to data')
    parser.add_argument('--percentage', type=float, default=0.2, help='percentage of val data')
    args = parser.parse_args()
    path = args.path

    percentage = args.percentage
    male_files = list(Path(path + '\\' + 'male').rglob('??????.jpg'))
    female_files = list(Path(path + '\\'+ 'female').rglob('??????.jpg'))
    male_files_train, male_files_val = train_test_split(male_files, test_size=percentage, random_state=42, shuffle=True)
    female_files_train, female_files_val = train_test_split(female_files, test_size=percentage, random_state=42, shuffle=True)
    train_files = male_files_train + female_files_train
    val_files = male_files_val + female_files_val

    with open('train_files.txt', 'w') as t:
     for path in train_files:
      t.write(str(path) + '\n')

    with open('val_files.txt', 'w') as v:
     for path in val_files:
      v.write(str(path) + '\n')

