import os
import numpy as np
from PIL import Image
import click

from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image
from patch import load_images

# Compare different models
# input_dir = 'datasets/images/test', 'datasets/DIV2K/test'
# output_dir = 'myresults/compare'

def compare(batch_size, input_dir, output_dir):
    data = load_images(input_dir, batch_size)
    y_test, x_test = data['B'], data['A']
    weights = ['generator.h5', 'weights/Deblur_nonlock/generator_3_136.h5','weights/Deblur_lock/generator_3_210.h5']
    # weights = ['generator.h5', 'weights/DIV2K_1/generator_3_374.h5','weights/DIV2K_2/generator_3_507.h5']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generated = []
    for weight in weights:
        g = generator_model()
        g.load_weights(weight)
        generated_images = g.predict(x=x_test, batch_size=batch_size)
        generated.append([deprocess_image(img) for img in generated_images])
    generated = np.array(generated)
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img_0 = generated[0, i, :, :, :] # original
        img_1 = generated[1, i, :, :, :] # trainsfer learning
        img_2 = generated[2, i, :, :, :] # trainsfer learning with locked parameters

        # combine imgs and store
        output = np.concatenate((y, x, img_0, img_1, img_2), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save(os.path.join(output_dir, 'results{}.png'.format(i)))

        # store img seperately
        # im = Image.fromarray(img_0.astype(np.uint8))
        # im.save(os.path.join(output_dir, 'b0/results{}.png'.format(i)))
        # im = Image.fromarray(img_1.astype(np.uint8))
        # im.save(os.path.join(output_dir, 'b1/results{}.png'.format(i)))
        # im = Image.fromarray(img_2.astype(np.uint8))
        # im.save(os.path.join(output_dir, 'b2/results{}.png'.format(i)))


@click.command()
@click.option('--batch_size', default=100, help='Number of images to process')
@click.option('--input_dir', required=True, help='Path to input images')
@click.option('--output_dir', required=True, help='Path to output images')
def compare_command(batch_size, input_dir, output_dir):
    return compare(batch_size, input_dir, output_dir)


if __name__ == "__main__":
    compare_command()
