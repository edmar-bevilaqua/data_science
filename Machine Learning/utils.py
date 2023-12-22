from sklearn.model_selection import train_test_split

# This function will use Sklearn's train_test_split method 2 times to separate the data into training, test and validation sets.

def get_dataset_splits(X, y, ratios, seed=123):
    ratio_train, ratio_valid, ratio_test = ratios

    # first, split between train+val and test splits
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=ratio_test, random_state=seed, stratify=y,
    )

    # next, split train and validation splits
    ratio_valid_adjusted = ratio_valid / (1 - ratio_test)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, test_size=ratio_valid_adjusted,
        random_state=seed, stratify=y_trainval,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
