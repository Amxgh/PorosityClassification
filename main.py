import os

def main():

    os.system("python scripts/data_preprocessing.py")
    os.system("python scripts/train_model.py")
    os.system("python scripts/evaluate_model.py")


if __name__ == "__main__":
    main()