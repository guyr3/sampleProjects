from sklearn.model_selection import train_test_split

"""
File names for the training and test sets
Convention file names for the training, validation and test sets after splitting the data
"""
TRAINING_FILE = 'nlp2-a4.train'
TEST_FILE = 'nlp2-a4.test'

TRAINING_SRC = 'data/src-train.txt'
TRAINING_TGT = 'data/tgt-train.txt'
VALID_SRC = 'data/src-val.txt'
VALID_TGT = 'data/tgt-val.txt'
TEST_SRC = 'data/src-test.txt'


"""
Read through the training file, split to training and validation sets, save into separate files in format expected by OpenNMT
"""
def process_training_data():
    with open(TRAINING_FILE, mode='r', encoding='utf-8') as f:
        X, y = [], []
        for line in f:
            orthographic, phonemic = line.strip().split('\t')
            X.append(' '.join(orthographic))
            y.append(phonemic)

    # Splitting to training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1)

    """
    Saving into 4 different files
    """

    # Training X
    with open(TRAINING_SRC, 'wt', encoding='utf-8') as f:
        for data_instance in X_train:
            print(data_instance, file=f)

    # Validation X
    with open(VALID_SRC, 'wt', encoding='utf-8') as f:
        for data_instance in X_validation:
            print(data_instance, file=f)

    # Training y
    with open(TRAINING_TGT, 'wt', encoding='utf-8') as f:
        for data_instance in y_train:
            print(data_instance, file=f)

    # Validation y
    with open(VALID_TGT, 'wt', encoding='utf-8') as f:
        for data_instance in y_validation:
            print(data_instance, file=f)


"""
Read the test file, save it with the expected format into 'src-test.txt'
"""
def process_test_data():
    with open(TEST_FILE, mode='r', encoding='utf-8') as f:
        with open(TEST_SRC, mode='wt', encoding='utf-8') as t:
            for line in f:
                orthographic = ' '.join(line.strip())
                print(orthographic, file=t)