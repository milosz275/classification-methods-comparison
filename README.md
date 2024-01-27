# Classification methods comparison
R Project aiming to compare k-NN, SVM and Random Forest classification methods

## Introduction
This report deals with the analysis of the R Project titled "Comparison of Classification Methods." It provides a comprehensive overview of the script's functionality, outlining the datasets and classification algorithms employed to classify plant/animal species based on additional data. The project's execution involved the implementation of a function designed to handle valid sets comprising features for the classification of objects into specific subgroups. Furthermore, data preprocessing was carried out to render them suitable for analysis.

The utilized algorithms include 1-NN, k-NN, and SVM, applied to datasets such as iris, penguins, and Hawks. These datasets were partitioned into training and testing sets at an 80% to 20% ratio. The report compares the effectiveness of each algorithm individually and explores variations of k-NN and SVM, involving adjustments to parameters during the model training process.

## Datasets
### Iris Dataset
This is a classic dataset widely used in the field of machine learning and statistics, describing irises categorized by species. It includes measurements of sepal length, sepal width, petal length, and petal width. The dataset consists of 150 observations, with 50 for each species: setosa, versicolor, and virginica.

### Penguins Dataset
The penguins dataset contains information about penguins, categorized by species, island of inhabitance, bill length, bill width, flipper length, body mass, sex, and the year of measurement. Rows with NA (not applicable) values were removed, constituting 3.20% of the original dataset (11 out of 344), which is less than the 5% threshold. The dataset comprises a total of 333 penguin records.

### Hawks Dataset
The hawks dataset includes information about hawks, categorized by species, month of measurement, day of measurement, age, wing length, weight, culmen length, hallux length, and tail length. Columns that did not contribute to the analysis were removed, followed by the elimination of records with NA values, accounting for 1.87% of the original dataset (17 out of 908), which is less than the 5% threshold. The dataset comprises a total of 891 observations of hawks.

## Algorithms
The following algorithms, 1-NN and k-NN, are treated as separate entities in this project, even though they represent a single algorithm. In the case of k-NN, the determination of the parameter k was implemented.

### 1-NN Algorithm
The 1-NN (1-nearest neighbor) algorithm is a simple and intuitive machine learning algorithm, but it can be computationally expensive. It involves finding the nearest case among all training cases, which may result in high computational costs. Subsequently, it assigns the label of the found nearest neighbor to the test case.

### k-NN Algorithm
The k-NN (k-nearest neighbors) algorithm is a straightforward algorithm mainly used for classification and, less frequently, for regression. For a test case, it identifies the k nearest training cases. After finding the k nearest training cases, it assigns the label to the test case that most frequently occurred among the nearest training cases.

### SVM Algorithm
The SVM (Support Vector Machines) algorithm is utilized for machine learning in both regression and classification. It maximizes the prediction accuracy of the model without overfitting to the training data. It is a recommended algorithm for analyzing large datasets. The default kernel function applied is the radial basis function (RBF) kernel.

### Random Forest Algorithm
The Random Forest algorithm is a machine learning algorithm used for both classification and regression. It relies on multiple decision trees, and during the construction of these trees, they receive random features (subset of data with replacement). In classification, data pass through all decision trees, and based on the decisions from the Random Forest trees, a majority vote determines the final classification.

## Research results
For each dataset, a comparison of the accuracies of the algorithms was conducted. Legend:

- Model_1nn: k-NN algorithm for k = 1
- Model_knn: k-NN algorithm for k from 1 to 100, defaulting to the best accuracy value
- Model_svm: SVM algorithm for degree = 1, scale = 1, and C = 1
- Model_svm_2: SVM algorithm for degree = 2, scale = 2, and C = 2
- Model_svm_3: SVM algorithm for degree = 3, scale = 3, and C = 3
- Model_RF: Random Forest algorithm

### Results – Iris dataset
#### Accuracies of models in the iris dataset
This comparison illustrates the accuracies of the 1-NN, k-NN, SVM, and Random Forest models in classifying iris species. In this case, SVM proved to be the most effective in classification.

![](/src/acc-iris.png)

#### k-NN accuracies in the iris dataset
In the following comparison, the accuracy of the k-NN algorithm in classifying iris species is shown, varying with algorithm parameters for the iris dataset. Performance was examined for k from 1 to 100. No distinguishable trend was observed regarding the k parameter, but for larger k values, accuracy decreases.

![](/src/knn-iris.png)

#### SVM accuracies in the iris dataset
In the last comparison, the accuracy variation of the SVM algorithm in classifying iris species is shown, depending on algorithm parameters for the iris dataset. A decrease in performance is noticeable when parameters are set to 3.

![](/src/svm-iris.png)

### Results – Penguins dataset
#### Accuracies of models in the penguins dataset
This comparison illustrates the accuracies of the 1-NN, k-NN, SVM, and Random Forest models in classifying penguin species. A significant advantage is observed for the Random Forest and SVM algorithms.

![](/src/acc-penguins.png)

#### k-NN accuracies in the penguins dataset
In the following comparison, the accuracy of the k-NN algorithm in classifying penguin species is shown, varying with algorithm parameters for the penguins dataset. Performance was examined for k from 1 to 100. Initial k values are the most efficient.

![](/src/knn-penguins.png)

#### SVM accuracies in the penguins dataset
In the last comparison, the accuracy variation of the SVM algorithm in classifying penguin species is shown, depending on algorithm parameters for the penguins dataset. The SVM algorithm demonstrates its best efficiency with parameters set to 1.

![](/src/svm-penguins.png)

### Results – Hawks dataset
#### Accuracies of models in the Hawks dataset
This comparison illustrates the accuracies of the 1-NN, k-NN, and SVM models in classifying hawk species. All algorithms perform similarly well.

![](/src/acc-Hawks.png)

#### k-NN accuracies in the Hawks dataset
In the following comparison, the accuracy of the k-NN algorithm in classifying hawk species is shown, varying with algorithm parameters for the Hawks dataset. Performance was examined for k from 1 to 100. Initial k values are the most efficient, with a noticeable decline at k = 2.

![](/src/knn-Hawks.png)

#### SVM accuracies in the Hawks dataset
In the last comparison, the accuracy variation of the SVM algorithm in classifying hawk species is shown, depending on algorithm parameters for the Hawks dataset. In this case, similar accuracy is observed for all three combinations of SVM algorithm parameters.

![](/src/svm-hawks.png)

## Summary
The direct observation suggests that, on average, the most effective algorithm is SVM, achieving a classification accuracy of 97% on the test datasets. The Random Forest algorithm performed slightly less well but still competently. Conversely, both 1-NN and k-NN algorithms, where accuracy was taken as the maximum over the range of k values from 1 to 100, exhibited relatively lower performance.

![](/src/algorithms-overall.png)

## References
1. Iris dataset classification https://github.com/dataprofessor/code/blob/master/iris/iris-classification.R
2. K parameter calculation https://daviddalpiaz.github.io/r4sl/k-nearest-neighbors.html
