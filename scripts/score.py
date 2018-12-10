import click
import numpy as np
from deblurgan.utils import RESHAPE, list_image_files, load_image
from skimage.measure import compare_ssim, compare_psnr


def SSIM(X, Y, mean=True, multichannel=True):
    data_range = max(X.max() - X.min(), Y.max() - Y.min())
    scores = [compare_ssim(x, y, data_range=data_range, multichannel=multichannel) for x, y in zip(X, Y)]
    return scores if not mean else np.mean(scores)


def PSNR(X, Y, mean=True):
    data_range = max(X.max() - X.min(), Y.max() - Y.min())
    scores = np.mean([compare_psnr(x, y, data_range=data_range) for x, y in zip(X, Y)])
    return scores if not mean else np.mean(scores)


def evaluate_images(input_dir, mean=True):
    sharps, fakes = [], []
    image_files = list_image_files(input_dir)
    for path in image_files:
        image = np.array(load_image(path))
        _, image_y, image_x, image = np.split(image, np.arange(0, RESHAPE[0] * 3, RESHAPE[0], np.int32), 1)
        sharps.append(image_y)
        fakes.append(image)
    sharps = np.array(sharps)
    fakes = np.array(fakes)
    return len(image_files), SSIM(sharps, fakes, mean), PSNR(sharps, fakes, mean)


@click.command()
@click.option('--input_dir', help='Test images to evalute')
def evaluate_command(input_dir):
    n, ssim, psnr = evaluate_images(input_dir)
    print("n: {}, SSIM: {}, PSNR: {}\n".format(n, ssim, psnr))

if __name__ == "__main__":
    evaluate_command()
