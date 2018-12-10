
export PYTHONPATH=`pwd`
python scripts/train.py --n_images=512 --input_dir '../data/DeepVideo+CERTH' --weights_dir 'weights/Deblur_nonlock' --generator_weights 'generator.h5'
python scripts/train.py --n_images=512 --input_dir '../data/DeepVideo+CERTH' --weights_dir 'weights/Deblur_lock' --generator_weights 'generator.h5' --use_transfer True
