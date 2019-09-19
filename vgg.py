# Running the following code to load parameters from the VGG model...
# Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. 
# The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

# we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. 
#This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers).
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
