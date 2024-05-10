import csv
import pandas as pd
import numpy as np
import seaborn as sns
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService
from IPython.display import clear_output
from qiskit.circuit.library import EfficientSU2

features=[]
labels=np.array([])

with open(r'adelie.csv', newline='') as csvfile:
    adelie = csv.DictReader(csvfile)
    for pingvin in adelie:
        pom_array = [int(pingvin['Culmen_Length(mm)']),int(pingvin['Body_Mass(g)'])]
        features.append(pom_array)
        labels=np.append(labels,int(pingvin['Sex']))

print("Features: ",features)
print("Labels: ",labels)
featureNames=['Culmen Length (mm)','Body Mass(g)']
normalizedFeatures = MinMaxScaler().fit_transform(features)
print("Normalized Features: ",normalizedFeatures)

df = pd.DataFrame(features, columns=featureNames)
df["class"] = pd.Series(labels)

sns.pairplot(df, hue="class", palette="tab10")
plt.show()

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    normalizedFeatures, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)

svc = SVC()
_ = svc.fit(train_features, train_labels)

train_score_c4 = svc.score(train_features, train_labels)
test_score_c4 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")

num_features = normalizedFeatures.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", style="clifford", fold=20)
plt.show()

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
plt.show()

optimizer = COBYLA(maxiter=40)

service = QiskitRuntimeService(channel="ibm_quantum",token="4e30e9f296b475eb6739cd26c0793a94a95e29f6e3e929488cf15e1954c35fc0f418db7bb84762a7c3bef49f55028ac65ce25194d3466bcc52d56e581e127499")
backend = service.get_backend("ibmq_qasm_simulator")

sampler = Sampler(backend=backend)

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)


vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start
plt.show()

print(f"Training time: {round(elapsed)} seconds")


train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")
