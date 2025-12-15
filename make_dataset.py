import os

# Here we create a directory for storing the input data
input_data_path = './data/input/COLOR/instances'
if not os.path.exists(input_data_path):
    os.makedirs(input_data_path)

# #Here we download the input data
# ! wget https://mat.tepper.cmu.edu/COLOR/instances/instances.tar -P ./data/input/COLOR/

# #Here we extract the input data
# ! tar -xvf ./data/input/COLOR/instances.tar -C './data/input/COLOR/instances'
