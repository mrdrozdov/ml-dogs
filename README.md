# Machine Learning for Dogs in Pytorch

Download data.

```
bash download_data.sh
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

## Useful Resources

- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [Example classifier for dogs data.](https://github.com/zrsmithson/Stanford-dogs)
- [Pytorch Ignite](https://pytorch.org/ignite/)
