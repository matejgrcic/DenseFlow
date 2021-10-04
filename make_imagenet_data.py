import pickle
import numpy as np
from PIL import Image

def process_train(img_size):
    img_size2 = img_size * img_size

    dest = f"train_{img_size}x{img_size}"

    total = 0
    for i in range(1, 11):
        pickle_off = open(f"./../raw/train_data_batch_{i}","rb")
        emp = pickle.load(pickle_off)
        x = emp['data']

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))

        for j in range(x.shape[0]):
            total += 1
            name = f"./{dest}/{total}.png"
            Image.fromarray(x[j], "RGB").save(name)

    print(f"> Processed {total} training images!")


def process_valid(img_size):
    img_size2 = img_size * img_size

    dest = f"valid_{img_size}x{img_size}"

    total = 0
    pickle_off = open(f"./../raw/val_data","rb")
    emp = pickle.load(pickle_off)
    x = emp['data']

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))

    for j in range(x.shape[0]):
        total += 1
        name = f"./{dest}/{total}.png"
        Image.fromarray(x[j], "RGB").save(name)

    print(f"> Processed {total} validation images!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=32, required=True)
    args = parser.parse_args()

    process_train(args.img_size)
    process_valid(args.img_size)



