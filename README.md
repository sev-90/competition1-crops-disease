# competition1-crops-disease

# crops disease type prediction
Many African crops are suffering from the wheat rust as a devastating plant disease. This issue can significantly affect the life of farmers and the food security in the continent. The goal of this project is to train a machine learning algorithm to classify the crops to healthy, stem rust, and leaf rust. Relying on the state-of-the-art fastai learning library, convolutional neural network is trained to achieve the score of 0.3758 loss. 

## Task
In this competition, the goal is to create an image recognition model with the best accuracy to classify the crop imagery data into three classes namely, ‘healthy_wheat’, ‘leaf_rust’ and ‘stem_rust’.  The imagery dataset were collected from different sources such as in-field images collected in Ethiopia and Tanzania, and public images of Google [1]. 

## Strategy
The approach is to use convolutional neural networks for classification using the fastai library. The whole structure of our classifier consists of 7 components, namely, data inspection, data cleaning, data augmentation, transfer learning, progressive resizing, model re-training and prediction. The following sections will discuss each component in detail.

## Data inspection
Before getting into the model training, it is necessary to inspect the data carefully. We are given 876 labelled training data samples and 610 unlabelled test data samples. The train data consists of 142 images of healthy wheat, 358 images of crops with leaf rust, and 376 images of crops with stem rust. These images are of a variety of sizes, thus, we unify all the images to a unique size and normalize them with imagenet statistics available as imagenet_stats in fastai library. We used the show_batch method to observe a sample of our inputs.

## Data cleaning
When glancing through the original training data provided, we discovered that some of the pictures are labelled wrongly, for example in Fig.2, this picture should be labelled as ‘leaf_rust’ but its assigned label is ‘stem_rust’. The amount of mislabelled pictures are not significantly large, but given the size of training data is considerably small, as compared to normal training sample size of neural network, the mislabelled pictures might have significant adverse effect on the classifier. Hence, data cleaning is necessary. 

To do so, we first use the original training dataset to build a CNN. We only run the training for a few epoches (4 epoches) such that the general features of all classes are learnt but the CNN is not well fitted on all training data. Next, we use this trained CNN to classify on all the training data and eliminate data with top 200 loss. This is implemented using ImageCleaner widgets in fastai. Given the dataset size is considerably small, using the ImageCleaner widget is feasible. Since we are training with a small dataset, to maintain as many samples as possible, we relabel samples if possible and only delete samples which are hard to classify by human.

## Data augmentation
Given the small dataset size, we consider data augmentation necessary to prevent overfitting and have better generalization of the model. The consideration we have when applying data augmentation is that if the picture can be classified to its original class after transforming using data augmentation. Under this consideration, we transform the images with any combination within the following 4 effects: flipping images horizontally, rotating images by maximum degree of 30, zoom out by a maximum factor of 0.5 and applying lighting to the images with a maximum factor of 0.2. 

## Transfer learning
Transfer learning is a popular technique in computer vision especially when dealing with small dataset. Pre-trained networks based on millions of images are capable of extracting some common features of images in the convolutional layers. The training data can be used to train the top layers such that the neural networks can be tuned for this task. In this case, we used Resnet34 as the pre-trained model. 

## Progressive image resizing
Progressive resizing is a method to boost the trained CNN accuracy. This technique was first adopted by Jeremy Howard who could achieve a top 10% score in the Planet kaggle competition, and he presents this method at his course “ Practical Deep learning for Coders [2].” In this approach, the aim is to train the CNN with sequentially resizing images. Thus, following this approach, we tried different progressive resizing settings as (64, 128,224) and (128, 224), and (128, 256). Due to overfitting issue resulting from the first two settings, and the best accuracy we got from the third setting, we used (128, 256) to train the model. First, we trained our model with the image size of 128*128, and then, using the weights of the trained model, we kept training the CNN with the larger image size of 256 * 256. Therefore, the final large-scale model embeds the layers and weights estimated by smaller-scale model. 

## Model re-training
After progressive resizing, we have a model whose input size is 224 x 224. We then use this model to make predictions on the training data, identify the 120 samples with highest value loss and add these samples to the training data and train the model with previous weights again. The reason behind this is that these samples are the samples that have significant loss, and it is very likely that they look ambiguous to the model, for example, stem_rust sample and leaf_rust sample can look similar as shown in fig.3 which is a sample from leaf_rust but similar to stem_rust.

## Performance 
The structure above constructs the classifier and performs hyperparameter tuning on the number of epoch and learning rate. The accuracy of the trained model is monitored with the metrics of error rate and accuracy. Finally, we predicted the test samples using the trained weights. The best result we could achieve on the public board is 0.3758 loss.




References
    1. https://zindi.africa/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease/data

    2. https://course.fast.ai/