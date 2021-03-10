# Machine Learning for Dogs in Pytorch

Download data.

```
bash download_data.sh
```

Install dependencies.

- Install pytorch using instructions here: https://pytorch.org/

```
pip install pytorch-ignite
pip install scipy
pip install tqdm
```

Train using Resnet18. Should achieve about 45% accuracy on train data after 5 epochs.

```
python main.py \
--train_config '{"lr": 0.002, "max_epochs": 5}' \
--train_data_config '{"batch_size": 128, "num_workers": 4, "metadata_path": "./data/train_list.mat", "images_folder": "./data/Images"}' \
--eval_data_config '{"batch_size": 128, "num_workers": 4, "metadata_path": "./data/test_list.mat", "images_folder": "./data/Images"}' \
--model_config '{"name": "resnet"}' \
--progress \
--cuda
```

To fit larger batch sizes, can use gradient accumulation:

```
# This command accumulates gradients over 4 steps, so only batch size 32 is input to the model,
# and batch size of 128 is used to update the model.
python main.py \
--train_config '{"lr": 0.002, "max_epochs": 5, "accum_steps": 4}' \
--train_data_config '{"batch_size": 32, "num_workers": 4, "metadata_path": "./data/train_list.mat", "images_folder": "./data/Images"}' \
--eval_data_config '{"batch_size": 32, "num_workers": 4, "metadata_path": "./data/test_list.mat", "images_folder": "./data/Images"}' \
--model_config '{"name": "resnet"}' \
--progress \
--cuda
```

## Useful Resources

- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [Example classifier for dogs data.](https://github.com/zrsmithson/Stanford-dogs)
- [Pytorch Ignite](https://pytorch.org/ignite/)
