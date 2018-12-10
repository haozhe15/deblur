export PYTHONPATH='deblur'
python ~/deblur/scripts/train.py --n_images=512 --input_dir '~/DeblurTrain/' --weights_dir 'weights/Deblur_nonlock' --generator_weights 'generator.h5'
python ~/deblur/scripts/train.py --n_images=512 --input_dir '~/DeblurTrain/' --weights_dir 'weights/Deblur_lock' --generator_weights 'generator.h5' --use_transfer True
