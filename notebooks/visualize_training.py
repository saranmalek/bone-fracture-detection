import json
import matplotlib.pyplot as plt

with open("models/training_history.json") as f:
    history = json.load(f)

plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])

plt.show()