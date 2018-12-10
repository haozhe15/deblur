
python scripts/train.py --n_images=512 --input_dir 'datasets/DIV2K/train' --weights_dir 'weights/DIV2K_1' --generator_weights 'generator.h5' 
python scripts/train.py --n_images=512 --input_dir 'datasets/DIV2K/train' --weights_dir 'weights/DIV2K_2' --generator_weights 'generator.h5' --use_transfer True

python scripts/test.py --batch_size=10 --input_dir 'datasets/DIV2K/test' --output_dir 'myresults/DIV2K_2' --generator_weights 'weights/DIV2K_2/generator_0_559.h5'
