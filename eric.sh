
python scripts/train.py --n_images=512 --input_dir 'datasets/DIV2K/train' --weights_dir 'weights/DIV2K_1' --generator_weights 'generator.h5'
python scripts/train.py --n_images=512 --input_dir 'datasets/DIV2K/train' --weights_dir 'weights/DIV2K_2' --generator_weights 'generator.h5' --use_transfer True

python scripts/test.py --batch_size=10 --input_dir 'datasets/DIV2K/test' --output_dir 'myresults/DIV2K_0' --generator_weights 'generator.h5'
python scripts/test.py --batch_size=10 --input_dir 'datasets/DIV2K/test' --output_dir 'myresults/DIV2K_1' --generator_weights 'weights/DIV2K_1/generator_3_374.h5'
python scripts/test.py --batch_size=10 --input_dir 'datasets/DIV2K/test' --output_dir 'myresults/DIV2K_2' --generator_weights 'weights/DIV2K_2/generator_3_507.h5'

python scripts/compare.py --batch_size=1 --input_dir 'datasets/DIV2K/test' --output_dir 'myresults/DIV2K' 
python scripts/compare.py --batch_size=20 --input_dir 'datasets/images/test' --output_dir 'myresults/GOPRO' 
