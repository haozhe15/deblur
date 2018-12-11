import os
import numpy as np
from PIL import Image
import click

from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image
# from patch import load_images

# input_dir = 'datasets/images/test', 'datasets/DIV2K/test'
# output_dir = 'myresults'

def test(batch_size, input_dir, output_dir, generator_weights):
    data = load_images(input_dir, batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    g.load_weights(generator_weights)
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save(os.path.join(output_dir, 'results{}.png'.format(i)))


@click.command()
@click.option('--batch_size', default=4, help='Number of images to process')
@click.option('--input_dir', required=True, help='Path to input images')
@click.option('--output_dir', required=True, help='Path to output images')
@click.option('--generator_weights', default='generator.h5', help='Path to generator weights')
def test_command(batch_size, input_dir, output_dir, generator_weights):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return test(batch_size, input_dir, output_dir, generator_weights)


if __name__ == "__main__":
    test_command()
