#!/usr/bin/env bash

mkdir -p data
cd data

wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar xvf images.tar

wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
tar xvf annotation.tar

wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
tar xvf lists.tar
