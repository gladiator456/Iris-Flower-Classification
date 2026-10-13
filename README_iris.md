# Iris Flower Classification

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

A first end-to-end machine-learning workflow on the classic Iris dataset: explore the data, split it, train a classifier, evaluate it, and predict the species of a new flower. Everything is in one notebook, [`irisflower.ipynb`](irisflower.ipynb).

## The task

Given four measurements of an iris flower (sepal length, sepal width, petal length, petal width, all in cm), predict which of three species it is: *Iris-setosa*, *Iris-versicolor* or *Iris-virginica*. The dataset has 150 samples, 50 per species, with no missing values.

## Workflow

1. **Explore:** `info()` and `describe()` summaries, a seaborn pair plot coloured by species, and a correlation heatmap of the four features.
2. **Split:** 80% train and 20% test (120 / 30 samples), with `random_state=42`.
3. **Train:** `LogisticRegression(max_iter=200)` from scikit-learn.
4. **Evaluate:** accuracy, a per-class classification report and a confusion matrix on the test set.
5. **Predict:** classify a new flower from its raw measurements.

## Results

| Species | Precision | Recall | F1 | Test samples |
| --- | --- | --- | --- | --- |
| Iris-setosa | 1.00 | 1.00 | 1.00 | 10 |
| Iris-versicolor | 1.00 | 1.00 | 1.00 | 9 |
| Iris-virginica | 1.00 | 1.00 | 1.00 | 11 |

The model scored **100% accuracy** on the 30 held-out flowers, and the confusion matrix is purely diagonal.

Read this with care: Iris is a small, well-separated dataset (setosa in particular is linearly separable), and 30 test samples is too few to tell good models apart. The score shows that the pipeline works, and it should not be taken as a sign of a strong model.

## Run it

```bash
git clone https://github.com/gladiator456/Iris-Flower-Classification.git
cd Iris-Flower-Classification
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook irisflower.ipynb
```

The notebook reads the data from a file named `irisflower.zip` in the same folder. The dataset is not stored in this repo, so add your own copy: a CSV with the columns `sepal_length, sepal_width, petal_length, petal_width, species` (species labelled `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`), zipped under that name.

## Next steps

- Use k-fold cross-validation instead of a single split, to get a more reliable estimate.
- Compare against k-NN, a decision tree, SVM and random forest.
- Report a confusion matrix plot and per-feature importance.

## Author

Anuj Saxena
