import os
import click
import numpy as np
from deblurgan.utils import RESHAPE, list_image_files, load_image
from skimage.measure import compare_ssim, compare_psnr


def SSIM(X, Y, multichannel=True):
    data_range = max(X.max() - X.min(), Y.max() - Y.min())
    scores = [compare_ssim(x, y, data_range=data_range, multichannel=multichannel) for x, y in zip(X, Y)]
    return scores, np.mean(scores)


def PSNR(X, Y):
    data_range = max(X.max() - X.min(), Y.max() - Y.min())
    scores = [compare_psnr(x, y, data_range=data_range) for x, y in zip(X, Y)]
    return scores, np.mean(scores)


def evaluate_images(input_dir):
    sharps, fakes = [], []
    image_files = list_image_files(input_dir)
    for path in image_files:
        image = np.array(load_image(path))
        _, image_y, image_x, image = np.split(image, np.arange(0, RESHAPE[0] * 3, RESHAPE[0], np.int32), 1)
        sharps.append(image_y)
        fakes.append(image)
    sharps = np.array(sharps)
    fakes = np.array(fakes)
    return image_files, [SSIM(sharps, fakes), PSNR(sharps, fakes)]


@click.command()
@click.option('--input_dir', help='Test images to evalute')
@click.option('--output_path', default='log/score.txt', help='Path to save resut scores')
def evaluate_command(input_dir, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    images, (ssim, mean_ssim), (psnr, mean_psnr) = evaluate_images(input_dir)
    print("Total: {}, SSIM: {}, PSNR: {}\n".format(len(images), mean_ssim, mean_psnr))
    with open(output_path, 'w') as f:
        f.write('\n'.join([
            '[{}] SSIM: {}, PSNR: {}'.format(os.path.basename(image), ssim, psnr)
            for image, ssim, psnr in zip(images, ssim, psnr)
        ]))
        f.write("\nTotal: {}, SSIM: {}, PSNR: {}\n".format(len(images), mean_ssim, mean_psnr))

if __name__ == "__main__":
    evaluate_command()
