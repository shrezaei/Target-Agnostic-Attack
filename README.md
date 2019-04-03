# Target-Agnostic Brute Force Attack on Deep Models
This is the open source repository of target-agnostic attack on deep neural networks. The paper is accessible here:

# Quick Start
You need to put all the re-training data into the data folder. For face recognition case study, re-training data is available [here](http://vis-www.cs.umass.edu/lfw/#deepfunnel-anchor). Then, modify the loadData.py according to your re-training data folder structure.

For face recognition case study, the pre-trained model of VGG face is loaded from Keras package. For other case studies, you need to provide the pre-train model in the re-train.py and attack.py files.

Do the following step to re-train and launch the target-agnostic attack:
1. Run the loadData.py to load all re-training data into a numpy format.
2. Run the re-train.py to re-train a model using transfer learning on the loaded data.
3. Run the attack.py to launch the target-agnostic attack on the re-trained model.
