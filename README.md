# Facial Expression Recognition
Used Convolutional neural networks (CNN) for facial expression recognition . The goal is to classify each facial image into one of the seven facial emotion categories considered .


## Data :
We trained and tested our models on the data set from the [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge), which comprises 48-by-48-pixel grayscale images of human faces,each labeled with one of 7 emotion categories:<strong> anger, disgust, fear, happiness, sadness, surprise, and neutral </strong>.
<br><br>
 Image set of 35,887 examples, with training-set : dev-set: test-set as <strong> 80 : 10 : 10 </strong>.

## Dependencies
 Python 2.7, sklearn, numpy, Keras.

## Library Used:
  <ul>
	  <li> Keras </li>
	  <li> Sklearn </li>
	  <li> numpy </li>
  </ul>

## Getting started

  To run the code -

  1. Download FER2013 dataset from [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and extract in the main folder.

  2. To run deep CNN model. Open terminal and navigate to the project folder and run cnn_major.py file
    <pre>
    python cnn_major.py
    </pre>
    No need to train the model , already trained weights saved in model4layer_2_2_pool.h5 file.

  3. Want to train model yourself ?<br>
      Just change the statement
      <pre>
        is_model_saved = True
        // to
        is_model_saved = False
      </pre>


#####  Shallow CNN Model

Code Link -  [cnn_major_shallow](https://github.com/rishabh30/Facial-Expression-Recognition/blob/master/cnn_major_shallow.py)<br>
Model Structure-  [Link](https://github.com/rishabh30/Facial-Expression-Recognition/blob/master/model_2layer_2_2_pool.json)<br>
Saved model trained weights - [Link](https://github.com/rishabh30/Facial-Expression-Recognition/blob/master/model_2layer_2_2_pool.h5)


#####  Deep CNN Model

Code Link -  [cnn_major](https://github.com/rishabh30/Facial-Expression-Recognition/blob/master/cnn_major.py)<br>
Model Structure-  [Link](https://github.com/rishabh30/Facial-Expression-Recognition/blob/master/model_4layer_2_2_pool.json)<br>
Saved model trained weights - [Link](https://github.com/rishabh30/Facial-Expression-Recognition/blob/master/model_4layer_2_2_pool.h5)


## Model Training:

<h3> Shallow Convolutional Neural Network </h3>

First we built a shallow CNN. This network had two convolutional layers and one FC layer.<br>
<p>First convolutional layer, we had 32 3×3 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.</p>
<p>Second convolutional layer, we had 64 3×3 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.</p>
<p>In the FC layer, we had a hidden layer with 512 neurons and Softmax as the loss function.</p>


<h3> Deep Convolutional Neural Networks </h3>

To improve accuracy we used deeper CNN . This network had 4 convolutional layers and with 2 FC layer.

## Model Evaluation:

Model predicts softmax output for 7 label for an image
<pre>[  4.99624775e-07   3.69855790e-08   9.91190791e-01   8.15907307e-03  2.62175627e-06   9.97206644e-06   1.02341000e-03]
</pre>
which is converted to <br> <strong>  [ 2 ] </strong> <br>  label having highest value .
For evaluation , categorial accuracy is used .<br>

Some Experiment are done by changing number of layers and changing hyper-parameters.<br>

#### Accuracy Achieved :
<h4>
Shallow CNN -- 56.31% <br><br>
Deep-CNN    -- 65.55%
</h4>

## References

1. [*"Dataset: Facial Emotion Recognition (FER2013)"*](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) ICML 2013 Workshop in Challenges in Representation Learning, June 21 in Atlanta, GA.

2. [*"Convolutional Neural Networks for Facial Expression Recognition"*](https://arxiv.org/abs/1704.06756) Convolutional Neural Networks for Facial Expression Recognition Shima Alizadeh, Azar Fazel

3. [*"Andrej Karpathy's Convolutional Neural Networks (CNNs / ConvNets)"*](http://cs231n.github.io/convolutional-networks/) Convolutional Neural Networks for Visual Recognition (CS231n), Stanford University.

## License

Licensed under [MIT License](LICENSE)
