# Classification Reports

Dataset: `data\oranges_vs_grapefruit.csv`

Target column: `name`

Best model: **Support Vector Machine**

| model                  |   accuracy |   train_rows |   test_rows |
|:-----------------------|-----------:|-------------:|------------:|
| Support Vector Machine |     0.9395 |         8000 |        2000 |
| Naive Bayes            |     0.926  |         8000 |        2000 |
| Decision Tree          |     0.921  |         8000 |        2000 |

## Decision Tree

Accuracy: 0.9210

```
              precision    recall  f1-score   support

  grapefruit       0.91      0.94      0.92      1000
      orange       0.93      0.91      0.92      1000

    accuracy                           0.92      2000
   macro avg       0.92      0.92      0.92      2000
weighted avg       0.92      0.92      0.92      2000

```

## Naive Bayes

Accuracy: 0.9260

```
              precision    recall  f1-score   support

  grapefruit       0.92      0.93      0.93      1000
      orange       0.93      0.92      0.93      1000

    accuracy                           0.93      2000
   macro avg       0.93      0.93      0.93      2000
weighted avg       0.93      0.93      0.93      2000

```

## Support Vector Machine

Accuracy: 0.9395

```
              precision    recall  f1-score   support

  grapefruit       0.93      0.95      0.94      1000
      orange       0.95      0.93      0.94      1000

    accuracy                           0.94      2000
   macro avg       0.94      0.94      0.94      2000
weighted avg       0.94      0.94      0.94      2000

```
