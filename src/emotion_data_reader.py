import pickle


def run(pickle_file):
    with open(pickle_file, "rb") as file:
        # fi = file.read()
        save = pickle.load(file)
        train_images = save["train_images"]
        train_labels = save["train_labels"]
        validation_images = save["validation_images"]
        validation_labels = save["validation_labels"]
        test_images = save["test_images"]
        test_labels = save["test_labels"]

    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


if __name__ == "__main__":
    rs = run('./data/emotion_detection/emotion.bin')
