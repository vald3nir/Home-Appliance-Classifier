# Home Appliance Classifiers

Machine learning algorithms used to identify certain home appliances based on electricity consumption

## Data acquisition

### Sensor Node

![sensor-node](https://user-images.githubusercontent.com/23506996/154678026-78fc2a6c-b8c1-4a3e-aa02-fe6c66340325.png)

### Specification of collected data

| Attribute | Type |
| --- | --- |
| Current Waveform | Integer Array |
| Real Power | Float |
| Apparent Power | Float |
| Power Factor | Float 

### Dataset

![datasets](https://user-images.githubusercontent.com/23506996/154680476-5dd93464-dae5-4a0a-8d89-fded4e539e19.png)

Fetures:
 - Real electrical power
 - Harmonic components from 60Hz to 1320Hz

## Classifier tests

```python
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

```python
# Reading the base dataset
dataset = pd.read_csv('data/dataset.csv')
```

```python
# Splitting data sets for training
inputs = dataset.iloc[:, 1:].values
outputs = dataset.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=.3)
```

```python
# Standardizing the dataset
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
```

```python
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
```

```python
print(metrics.classification_report(y_test, y_predict, zero_division=0))
```

                           precision    recall  f1-score   support
    
                Air Fryer       1.00      1.00      1.00        34
                Batedeira       1.00      1.00      1.00        38
                Cafeteira       1.00      1.00      1.00        31
     Espremedor de Frutas       1.00      1.00      1.00        29
           Ferro Eletrico       0.97      1.00      0.99        36
                Furadeira       1.00      1.00      1.00        34
                  Gelagua       1.00      1.00      1.00        29
    Lampada Incandescente       1.00      1.00      1.00       159
           Liquidificador       1.00      1.00      1.00       119
         Maquina de Lavar       1.00      1.00      1.00        53
              Micro-ondas       1.00      1.00      1.00        52
        Prancha de Cabelo       1.00      1.00      1.00        48
             Sanduicheira       1.00      1.00      1.00        94
        Secador de Cabelo       1.00      0.99      0.99        67
                       TV       1.00      1.00      1.00        57
               Ventilador       1.00      1.00      1.00        83
    
                 accuracy                           1.00       963
                macro avg       1.00      1.00      1.00       963
             weighted avg       1.00      1.00      1.00       963

```python
print(metrics.confusion_matrix(y_test, y_predict))
```

    [[ 34   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0  38   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0  31   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0  29   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0  36   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0  34   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0  29   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0 159   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0 119   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  53   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  52   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0  48   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0  94   0   0   0]
     [  0   0   0   0   1   0   0   0   0   0   0   0   0  66   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  57   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  83]]

```python
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
```

```python
print(metrics.classification_report(y_test, y_predict, zero_division=0))
```

                           precision    recall  f1-score   support
    
                Air Fryer       1.00      1.00      1.00        34
                Batedeira       0.93      1.00      0.96        38
                Cafeteira       0.94      0.97      0.95        31
     Espremedor de Frutas       1.00      1.00      1.00        29
           Ferro Eletrico       0.94      0.94      0.94        36
                Furadeira       1.00      1.00      1.00        34
                  Gelagua       0.97      1.00      0.98        29
    Lampada Incandescente       1.00      1.00      1.00       159
           Liquidificador       1.00      0.98      0.99       119
         Maquina de Lavar       0.98      1.00      0.99        53
              Micro-ondas       1.00      1.00      1.00        52
        Prancha de Cabelo       1.00      0.96      0.98        48
             Sanduicheira       1.00      0.98      0.99        94
        Secador de Cabelo       0.97      0.97      0.97        67
                       TV       1.00      1.00      1.00        57
               Ventilador       1.00      1.00      1.00        83
    
                 accuracy                           0.99       963
                macro avg       0.98      0.99      0.99       963
             weighted avg       0.99      0.99      0.99       963

```python
print(metrics.confusion_matrix(y_test, y_predict))
```

    [[ 34   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0  38   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0  30   0   0   0   0   0   0   1   0   0   0   0   0   0]
     [  0   0   0  29   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0  34   0   0   0   0   0   0   0   0   2   0   0]
     [  0   0   0   0   0  34   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0  29   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0 159   0   0   0   0   0   0   0   0]
     [  0   1   0   0   0   0   1   0 117   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  53   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  52   0   0   0   0   0]
     [  0   2   0   0   0   0   0   0   0   0   0  46   0   0   0   0]
     [  0   0   2   0   0   0   0   0   0   0   0   0  92   0   0   0]
     [  0   0   0   0   2   0   0   0   0   0   0   0   0  65   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  57   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  83]]

```python
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
```

```python
print(metrics.classification_report(y_test, y_predict, zero_division=0))
```

                           precision    recall  f1-score   support
    
                Air Fryer       1.00      0.88      0.94        34
                Batedeira       0.83      0.50      0.62        38
                Cafeteira       0.76      0.90      0.82        31
     Espremedor de Frutas       0.74      1.00      0.85        29
           Ferro Eletrico       0.78      1.00      0.88        36
                Furadeira       0.89      0.74      0.81        34
                  Gelagua       0.74      0.69      0.71        29
    Lampada Incandescente       0.71      0.82      0.76       159
           Liquidificador       1.00      0.91      0.95       119
         Maquina de Lavar       0.76      0.83      0.79        53
              Micro-ondas       1.00      1.00      1.00        52
        Prancha de Cabelo       0.62      0.60      0.61        48
             Sanduicheira       0.99      0.93      0.96        94
        Secador de Cabelo       0.98      0.94      0.96        67
                       TV       1.00      0.88      0.93        57
               Ventilador       0.51      0.51      0.51        83
    
                 accuracy                           0.82       963
                macro avg       0.83      0.82      0.82       963
             weighted avg       0.83      0.82      0.82       963

```python
print(metrics.confusion_matrix(y_test, y_predict))
```

    [[ 30   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0]
     [  0  19   0   0   0   0   4   6   0   0   0   3   0   0   0   6]
     [  0   0  28   0   0   0   0   0   0   3   0   0   0   0   0   0]
     [  0   0   0  29   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0  36   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   4   0  25   0   0   0   5   0   0   0   0   0   0]
     [  0   1   0   0   0   0  20   3   0   0   0   1   0   0   0   4]
     [  0   0   0   0   0   0   0 130   0   0   0   3   0   0   0  26]
     [  0   0   0   5   0   3   0   0 108   3   0   0   0   0   0   0]
     [  0   0   9   0   0   0   0   0   0  44   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  52   0   0   0   0   0]
     [  0   2   0   0   0   0   1  11   0   0   0  29   0   0   0   5]
     [  0   0   0   0   3   0   0   0   0   3   0   0  87   1   0   0]
     [  0   0   0   0   3   0   0   0   0   0   0   0   1  63   0   0]
     [  0   0   0   1   0   0   1   0   0   0   0   5   0   0  50   0]
     [  0   1   0   0   0   0   1  33   0   0   0   6   0   0   0  42]]

```python
classifier = SVC(kernel='linear', C=5.0)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
```

```python
print(metrics.classification_report(y_test, y_predict, zero_division=0))
```

                           precision    recall  f1-score   support
    
                Air Fryer       1.00      1.00      1.00        34
                Batedeira       0.97      0.97      0.97        38
                Cafeteira       1.00      0.94      0.97        31
     Espremedor de Frutas       0.97      1.00      0.98        29
           Ferro Eletrico       0.86      0.86      0.86        36
                Furadeira       1.00      1.00      1.00        34
                  Gelagua       1.00      1.00      1.00        29
    Lampada Incandescente       0.99      1.00      1.00       159
           Liquidificador       1.00      0.99      1.00       119
         Maquina de Lavar       0.96      1.00      0.98        53
              Micro-ondas       1.00      1.00      1.00        52
        Prancha de Cabelo       0.98      0.98      0.98        48
             Sanduicheira       1.00      1.00      1.00        94
        Secador de Cabelo       0.93      0.93      0.93        67
                       TV       1.00      1.00      1.00        57
               Ventilador       1.00      0.99      0.99        83
    
                 accuracy                           0.98       963
                macro avg       0.98      0.98      0.98       963
             weighted avg       0.98      0.98      0.98       963

```python
print(metrics.confusion_matrix(y_test, y_predict))
```

    [[ 34   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0  37   0   0   0   0   0   0   0   0   0   1   0   0   0   0]
     [  0   0  29   0   0   0   0   0   0   2   0   0   0   0   0   0]
     [  0   0   0  29   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0  31   0   0   0   0   0   0   0   0   5   0   0]
     [  0   0   0   0   0  34   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0  29   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0 159   0   0   0   0   0   0   0   0]
     [  0   0   0   1   0   0   0   0 118   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  53   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  52   0   0   0   0   0]
     [  0   1   0   0   0   0   0   0   0   0   0  47   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0  94   0   0   0]
     [  0   0   0   0   5   0   0   0   0   0   0   0   0  62   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  57   0]
     [  0   0   0   0   0   0   0   1   0   0   0   0   0   0   0  82]]

```python
classifier = MLPClassifier(hidden_layer_sizes=(20, 20, 20))
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
```

    /home/dev/.local/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py:692: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(

```python
print(metrics.classification_report(y_test, y_predict, zero_division=0))
```

                           precision    recall  f1-score   support
    
                Air Fryer       1.00      1.00      1.00        34
                Batedeira       0.97      0.76      0.85        38
                Cafeteira       1.00      1.00      1.00        31
     Espremedor de Frutas       0.97      1.00      0.98        29
           Ferro Eletrico       0.75      1.00      0.86        36
                Furadeira       0.91      0.91      0.91        34
                  Gelagua       0.72      0.97      0.82        29
    Lampada Incandescente       0.98      0.99      0.98       159
           Liquidificador       0.97      0.97      0.97       119
         Maquina de Lavar       1.00      0.96      0.98        53
              Micro-ondas       1.00      1.00      1.00        52
        Prancha de Cabelo       0.94      0.94      0.94        48
             Sanduicheira       0.98      1.00      0.99        94
        Secador de Cabelo       1.00      0.82      0.90        67
                       TV       1.00      1.00      1.00        57
               Ventilador       0.99      0.95      0.97        83
    
                 accuracy                           0.96       963
                macro avg       0.95      0.95      0.95       963
             weighted avg       0.96      0.96      0.96       963

```python
print(metrics.confusion_matrix(y_test, y_predict))
```

    [[ 34   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0  29   0   0   0   0   9   0   0   0   0   0   0   0   0   0]
     [  0   0  31   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0  29   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0  36   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0  31   0   0   3   0   0   0   0   0   0   0]
     [  0   1   0   0   0   0  28   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0 157   0   0   0   2   0   0   0   0]
     [  0   0   0   1   0   3   0   0 115   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  51   0   0   2   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  52   0   0   0   0   0]
     [  0   0   0   0   0   0   2   0   0   0   0  45   0   0   0   1]
     [  0   0   0   0   0   0   0   0   0   0   0   0  94   0   0   0]
     [  0   0   0   0  12   0   0   0   0   0   0   0   0  55   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  57   0]
     [  0   0   0   0   0   0   0   3   0   0   0   1   0   0   0  79]]

```python
# Splitting data sets for training
inputs = dataset.iloc[:, 1:3].values
outputs = dataset.iloc[:, 0].values
```

```python
pca = PCA(n_components=2)
components = pca.fit_transform(StandardScaler().fit_transform(inputs))
targets = set(outputs)
```

```python
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.set_title('Components PCA', fontsize=20)
ax.set_xlabel('Firs Component', fontsize=15)
ax.set_ylabel('Second  Component', fontsize=15)

final_frame = pd.DataFrame(data=components, columns=['component_1', 'component_2'])
final_df = pd.concat([final_frame, dataset[['home_appliance']]], axis=1)

colors = ["lime", "grey", "maroon", "red", "m", "seagreen", "coral", "orange", "blue",
          "pink", "c", "k", "darkblue", "y"]

for target, color in zip(targets, colors):
    indices_to_keep = final_df['home_appliance'] == target
    x = final_df.loc[indices_to_keep, 'component_1']
    y = final_df.loc[indices_to_keep, 'component_2']
    ax.scatter(x, y, c=color, s=50)
    
ax.legend(targets)
ax.grid()
plt.show()
```

![image](https://user-images.githubusercontent.com/23506996/154677104-f9845d3e-d46a-43d4-89c7-9d7e400263e2.png)
