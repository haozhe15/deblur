import os
import numpy as np
from PIL import Image
import click

from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image


def deblur(image_path, output_dir):
    data = {
        'A_paths': [image_path],
        'A': np.array([preprocess_image(load_image(image_path))])
    }
    x_test = data['A']
    g = generator_model()
    g.load_weights('generator.h5')
    generated_images = g.predict(x=x_test)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)

    for i in range(generated_images.shape[0]):
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save(os.path.join(output_dir, os.path.basename(image_path)))


@click.command()
@click.option('--image_path', help='Image to deblur')
@click.option('--output_dir', default='samples', help='Path to deblurred images.')
def deblur_command(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return deblur(image_path, output_dir)


if __name__ == "__main__":
    deblur_command()
