import os
import click
import numpy as np
from deblurgan.utils import RESHAPE, list_image_files, load_image
from skimage.measure import compare_ssim, compare_psnr


def SSIM(X, Y, multichannel=True):
    data_range = max(X.max() - X.min(), Y.max() - Y.min())
    scores = [
        compare_ssim(x, y, data_range=data_range, multichannel=multichannel)
        for x, y in zip(X, Y)
    ]
    return np.mean(scores), scores


def SSIMs(X, Ys, multichannel=True):
    return [SSIM(X, Y, multichannel) for Y in Ys]


def PSNR(X, Y):
    data_range = max(X.max() - X.min(), Y.max() - Y.min())
    scores = [
        compare_psnr(x, y, data_range=data_range)
        for x, y in zip(X, Y)
    ]
    return np.mean(scores), scores,


def PSNRs(X, Ys):
    return [PSNR(X, Y) for Y in Ys]


def evaluate_images(input_dir, num):
    sharps, blur_fake_groups = [], []
    image_files = list_image_files(input_dir)

    for path in image_files:
        image = np.array(load_image(path))
        images = np.split(image, np.arange(0, RESHAPE[0] * num, RESHAPE[0], np.int32), 1)
        image_y, blur_fake_images = images[1], images[2:]
        sharps.append(image_y)
        blur_fake_groups.append(blur_fake_images)

    sharps = np.array(sharps)
    blur_fake_groups = np.array(blur_fake_groups)

    return image_files, SSIMs(sharps, blur_fake_groups.swapaxes(0, 1)), PSNRs(sharps, blur_fake_groups.swapaxes(0, 1))


@click.command()
@click.option('--input_dir', help='Test images to evalute')
@click.option('--output_path', default='log/score.txt', help='Path to save resut scores')
@click.option('--num', default=3, help='Number of images in a row')
def evaluate_command(input_dir, output_path, num):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    images, num_ssim_info, num_psnr_info = evaluate_images(input_dir, num)

    num_mean_ssim = [ssim_info[0] for ssim_info in num_ssim_info]
    num_full_ssim = np.array([ssim_info[1] for ssim_info in num_ssim_info]).T
    num_mean_psnr = [psnr_info[0] for psnr_info in num_psnr_info]
    num_full_psnr = np.array([psnr_info[1] for psnr_info in num_psnr_info]).T

    message_mean_scores = "Total: {}, SSIM: {}, PSNR: {}".format(
        len(images), num_mean_ssim, num_mean_psnr
    )
    print(message_mean_scores)
    with open(output_path, 'w') as f:
        f.write('[Image] SSIM: [{0} Image], PSNR: [{0} Image]\n'.format(num-1))
        f.write('\n'.join([
            '[{}] SSIM: {}, PSNR: {}'.format(
                os.path.basename(image), num_ssim, num_psnr
            )
            for image, num_ssim, num_psnr in zip(
                images, num_full_ssim, num_full_psnr
            )
        ]))
        f.write("\n{}\n".format(message_mean_scores))

if __name__ == "__main__":
    evaluate_command()
