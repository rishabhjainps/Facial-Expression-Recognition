# Facial Expression Recognition
We have developed convolutional neural networks (CNN) for a facial expression recognition task. The goal is to classify each facial image into one of the seven facial emotion categories considered .


## Data :
We trained and tested our models on the data set from the [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge), which comprises 48-by-48-pixel grayscale images of human faces,each labeled with one of 7 emotion categories: anger, disgust, fear, happiness, sadness, surprise, and neutral. We used a image set of 35,887 examples, with training set : dev set: test set as 80:10:10.

## Library Used:
   <ul>
	<li> Keras </li> 
	<li> Sklearn </li>
   </ul>

## Model Training:
	
<h3> Shallow Convolutional Neural Network </h3>
First we built a shallow CNN. This network had two convolutional layers and one FC layer.<br>
<p>First convolutional layer, we had 32 3×3 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.</p>
<p>Second convolutional layer, we had 64 3×3 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.</p>
<p>In the FC layer, we had a hidden layer with 512 neurons and Softmax as the loss function.</p>


<h3> Deep Convolutional Neural Networks </h3>
To improve accuracy we used deeper CNN . This network had 4 convolutional layers and with 2 FC layer.

<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #  
=================================================================
conv2d_1 (Conv2D)            (None, 48, 48, 64)        640      
_________________________________________________________________
batch_normalization_1 (Batch (None, 48, 48, 64)        256      
_________________________________________________________________
activation_1 (Activation)    (None, 48, 48, 64)        0        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 24, 24, 64)        0        
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 64)        0        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 128)       204928   
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 24, 128)       512      
_________________________________________________________________
activation_2 (Activation)    (None, 24, 24, 128)       0        
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 128)       0        
_________________________________________________________________
dropout_2 (Dropout)          (None, 12, 12, 128)       0        
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 512)       590336   
_________________________________________________________________
batch_normalization_3 (Batch (None, 12, 12, 512)       2048     
_________________________________________________________________
activation_3 (Activation)    (None, 12, 12, 512)       0        
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 512)         0        
_________________________________________________________________
dropout_3 (Dropout)          (None, 6, 6, 512)         0        
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 512)         2359808  
_________________________________________________________________
batch_normalization_4 (Batch (None, 6, 6, 512)         2048     
_________________________________________________________________
activation_4 (Activation)    (None, 6, 6, 512)         0        
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 3, 3, 512)         0        
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 3, 512)         0        
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0        
_________________________________________________________________
dense_1 (Dense)              (None, 256)               1179904  
_________________________________________________________________
batch_normalization_5 (Batch (None, 256)               1024     
_________________________________________________________________
activation_5 (Activation)    (None, 256)               0        
_________________________________________________________________
dropout_5 (Dropout)          (None, 256)               0        
_________________________________________________________________
dense_2 (Dense)              (None, 512)               131584   
_________________________________________________________________
batch_normalization_6 (Batch (None, 512)               2048     
_________________________________________________________________
activation_6 (Activation)    (None, 512)               0        
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0        
_________________________________________________________________
dense_3 (Dense)              (None, 7)                 3591     
=================================================================


</pre>
## Model Evaluation:
<pre>
Model predicts possibility for 7 label for an image 
[  4.99624775e-07   3.69855790e-08   9.91190791e-01   8.15907307e-03  2.62175627e-06   9.97206644e-06   1.02341000e-03]
which is converted to  [2]  label having highest probability .</pre>
<br>
For evaluation , categorial accuracy is used .<br>
Experiment by changing number of layers and parameter is done .<br>
For Shallow CNN we achieved 56.31% test set accuracy.<br>
The best performance came in Deep-CNN achieving a test set accuracy of 65.55%.

