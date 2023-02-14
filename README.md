# Food-11 Image Classification

Food-11 Image Classification is an innovative computer vision project that seeks to recognize and categorize 11 different types of popular food items. From Apple Pie to Sushi, the project will use deep learning algorithms to accurately classify images of food and provide a comprehensive understanding of each food item's visual characteristics. With this model, we aim to gain a better understanding of the distribution of various food items, contribute to the development of food-related technologies, and provide a fun and engaging way for people to learn about different types of cuisine. Whether you're a foodie or just looking for some culinary inspiration, this project promises to bring the world of food to life.

## Requirements

The following packages are required to run the notebook:

- matplotlib
- numpy
- tensorflow
- pandas
- splitfolders
- opencv-python
- seaborn
- scikit-learn
- squarify

You can install these packages using the following pip command:

```bash
  pip install matplotlib numpy tensorflow pandas splitfolders opencv-python seaborn scikit-learn squarify
```

## Usage

Here's how you can use the code in this project:

1. Clone or download the repository to your local machine.
 
2. Download the dataset from Kaggle (link provided below).

3. Open the Jupyter Notebook file in the repository.

4. Run the cells in the notebook to train the model and make predictions.

5. You can modify the code to fit your needs, such as changing the model architecture, training parameters, and prediction logic.

Note: Make sure to activate the virtual environment each time you work on the project to have the required packages installed.

## Dataset

The dataset used in this project is the Food-11 Image Classification dataset and can be obtained from [Kaggle](https://www.kaggle.com/datasets/imbikramsaha/food11).

The dataset contains 9900 images and is divided into 11 categories: "Apple Pie", "Cheesecake", "Chicken Curry", "French Fries", "Fried Rice", "Hamburger", 'Hot Dog', 'Ice Cream', 'Omelette', 'Pizza', 'Sushi'.

Note: Please review and abide by the terms of use and license information for the dataset before using it in your own projects.

## Model

The model used in this project is based on EfficientNetV2B0, with additional layers added for the food image classification task. The model architecture is as follows:

- A preprocessing layer for data augmentation
- The EfficientNetV2B0 model, without the top layer, as a base model
- A dropout layer with a rate of 0.3 to prevent overfitting
- A GlobalAveragePooling2D layer to reduce the spatial dimensions
- Another dropout layer with a rate of 0.4
- A dense layer with 11 neurons to produce the output
- An activation layer with softmax activation function to convert the output into probabilities of each class

The model was compiled with categorical crossentropy loss, Adam optimizer, and accuracy metric.

## Training Process

The model was trained for 25 epochs using the `fit` method from the Keras API. The training data was passed to the method as an argument, along with the number of steps per epoch and the validation data. 

Three callbacks were added to monitor the training process: model checkpoint, learning rate reduction, and a CSV logger. 

After the initial training phase with the layers frozen, the model was fine-tuned by unfreezing all of the layers of the base model. The layers were set to be trainable and the model was trained again using the same process as before. 

## Evaluation

The model was evaluated on a test dataset which was 10% of the original dataset. The model achieved an accuracy score of 90.71% on the test dataset. A classification report and a confusion matrix were also generated to provide a more comprehensive understanding of the model's performance. The classification report displayed the precision, recall and f1-score for each of the 6 chessman categories. The confusion matrix provided a graphical representation of the number of correct and incorrect predictions made by the model. Both the classification report and the confusion matrix were plotted to visualize the results and to gain further insights into the model's performance.

## Future Work

- Training the model with more data or different datasets to improve the accuracy score.
- Experimenting with other image classification models such as ResNet, InceptionV3, and Xception to compare their performance.
- Incorporating additional layers to enhance the model's ability to identify food images.
- Exploring the use of Generative Adversarial Networks to generate new images of food items to add to the dataset.

# License

MIT

## Author

[Umar Saeed](https://www.linkedin.com/in/umar-saeed-16863a21b/)


