# Plankton Recognition
In this challenge, we deal with an image dataset of piece of oceanic ecosystems. This challenge aims at developing solid approaches to plankton image classification. Beyond simply producing a well-performing model for making predictions, in this challenge we would like you to start developing your skills as a machine learning scientist. In this regard, the notebook should be structured in such a way as to explore the following tasks that are expected to be carried out whenever undertaking such a project. 

# Dataset
You can download the dataset [here](https://drive.google.com/file/d/1kkv9wH0IuPbODt0gdEx_a1aXSnQ6ZNc5/view?usp=sharing)

There are totally 243610 samples that is categorized into 39 classes.

![](https://github.com/tuananh0305/Plankton_Recognition_Challenge/blob/master/images/classes_distribution.png)

## 1. Data preparation
* **Find missing labeled samples**: 
We remove all the samples having no label from the dataset. There are 1003 samples like that. After removing, the dataset remains 242607 samples.

* **Draw distribution of image sizes**:
In the image classification, inputs images should have the same size. Thus, to find the most appropriate dimension of image to resize, we draw two distributions of images size in terms of width and height. These distributions can give us more insight of "the population" of imges size, and then possibly choose the most suitable dimension of images to resize.

* **Draw distribution of classes**: 
  * The important step is to discover the distribution of classes in the dataset. It can help us detect imbalance problem and how its level is. In Plankton dataset, we can clearly see that there is a serious imbalance among 39 classes. In particular, the majority class has 138439 samples while the minority class only has 12 samples.
  * The serious imbalance dataset can result in the bias for the classifier because the loss will be dominated by the majority classes. The accuracy can still be high if the model is trained with an imbalanced dataset. However, it will not reflect much meaning due to the high accuracy mainly comes from the majority classes, not from minority classes.
  * Thus, we suggest two methods to deal with imbalanced problems of dataset. The first one is to try resampling the dataset. The approach of this method is: we generate instances from the under-represented classes by modifying them by flipping both vertically and horizontally, Roting, Zooming, Scaling, Cropping. This process is called "Data Augumentaion". Simutaneously, we also delete instances from the over-represented classes to ensure the balance characteristics of the dataset. The ratio for generating and deleting depends on the intitial proportion of images in each class. The second method is 'class_weight' method. The class_weight is a dictionary of weights that defines each class and the weighting to apply in the calculation of loss function when fitting the model.

* **Resize images and do Pixel Normalization**
we resize all the input images to 32x32 and scale pixel values to the range 0-1 to centering and even standardizing the values.

* **Data Augmentation**
Data Augumentation is the process for up/down sampling instances from the original dataset to narrow the difference between over-represented classes and under-presented classes. For under-represented classes, we try flipping, roting, zooming, cropping from intial images to create more images. For over-represented classes, we randomly delete images. The ratio for generating or deleting images should depend on the intial proportion of images in each class.

We catogorized the classes into 5 levels of imbalance. It will help us easily to scale up/down the number of images in each class to ensure the balance characteristic.

  * For 'VeryLowFreqClasses' group that the number of sample less than 100, we generate the data by 200 times
  * For 'LowFreqClasses' group that the number of sample n (n >= 100 and n < 1000), we scale up the number of image by 20 times
  * For 'MediumFreqClasses' group that the number of sample n (n >= 1000 and n < 10000), we scale up the number of image by 2 times
  * For 'HighFreqClasses' group, we keep unchange the number of images;
  * In veryHighFreqClasses, we do the down-sampling to reduce the number of images downto 20000 per class.

After the augumentation process, we can see the new distribution of 39 classes in the figure below. The dataset now becomes more balanced.

![](https://github.com/tuananh0305/Plankton_Recognition_Challenge/blob/master/images/classes_distribution_after_data_augmentation.png)

## 2. Model Selection
* To clearly see the effect of the augumentation technique on the imbalanced classification problem; firstly we will train our model with the originally imbalanced dataset and use the right metrics to evaluate the performances of classifiers; after that, we will train our model with the augment dataset above and also use the same metrics to evaluate the performances of classifiers again.

* Actually, when working with an imbalanced dataset, the minority classes are typically of the most interest. This means that the model's ability in correctly predicting the label of probability for the minority classes is more important than the majority classes. That is why we need to choose of metrics that give us more insight into the accuracy of the model rather than traditional classification accuracy.

* Confusion metric is use to describe the performance of a classification model on a set of test data for which the true values are known.

We focus on trying different CNN models because it is the best method for image classification. We start with a simple model, then try to train with more complicated models:

* Train with simple model (LeNet5) with (1) the original imbalance dataset, (2) the dataset after data augment, (3) the class weighted method
* Improve the LeNet5 model by adding 'dropout' layer and using 'relu' activation
* Apply transfer learning with complicated pre-trained model (VGG16) for our dataset
* Construct a new model which is suitable for the scope of our dataset.

## 3. Performance Evaluation
As mentioned above, **we not only focus on accuracy and weighted average f1-score, we also strongly want to improve macro average f1-score.**

The dataset is split into train set, validation set and test set to make sure that we performance evaluation on the unseen dataset. The 'stratify' is used in 'train_test_split' function to ensure the representative characteristic for validation set and test set. Data augumentation is only applied on training set.

After training our model with the best hyperparameters, we test the performance on test set and got the results:

|      |Accuracy | Macro average F1 score | Weighted average F1 score |
|------|------|------|------|
| Our best model on test set |0.74|0.58|0.75|


