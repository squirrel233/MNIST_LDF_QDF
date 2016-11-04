%--------------------------------------------------------------------------
% 16/10/8
% load raw data, and save as .mat file
%--------------------------------------------------------------------------
train_images = loadMNISTImages('train-images.idx3-ubyte');
save MNIST-train-images train_images;
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
save MNIST-train-labels train_labels;

test_images = loadMNISTImages('t10k-images.idx3-ubyte');
save MNIST-test10k-images test_images;
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
save MNIST-test10k-labels test_labels;