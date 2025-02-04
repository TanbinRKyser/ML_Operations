import numpy as np
import matplotlib.pyplot as plt
#%%
# Confusion Matrix Calculation
def compute_confusion_matrix(predictions, ground_truth):
    TP = np.sum((predictions == 1) & (ground_truth == 1))
    TN = np.sum((predictions == 0) & (ground_truth == 0))
    FP = np.sum((predictions == 1) & (ground_truth == 0))
    FN = np.sum((predictions == 0) & (ground_truth == 1))
    return TP, TN, FP, FN
#%%
# Metrics Calculation
def compute_accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)
#%%
def compute_sensitivity(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0
#%%
def compute_specificity(TN, FP):
    return TN / (TN + FP) if (TN + FP) != 0 else 0
#%%
def compute_mcc(TP, TN, FP, FN):
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return numerator / denominator if denominator != 0 else 0
#%%
# Bayes' Theorem
def bayes_theorem( prior, sensitivity, specificity ):
    false_positive_rate = 1 - specificity
    marginal_probability = (sensitivity * prior) + (false_positive_rate * (1 - prior))
    return (sensitivity * prior) / marginal_probability
#%%
# Data
predictions = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1 ])
ground_truth = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0 ])

# Confusion Matrix
TP, TN, FP, FN = compute_confusion_matrix( predictions, ground_truth )

# Metrics
accuracy = compute_accuracy( TP, TN, FP, FN )
sensitivity = compute_sensitivity( TP, FN )
specificity = compute_specificity( TN, FP)
mcc = compute_mcc( TP, TN, FP, FN ) 
#%%
# Print Metrics
print(f"Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"MCC: {mcc}")
#%%
# Given parameters
sensitivity = 0.95  # 95%
specificity = 0.99  # 99%
# Range of prior probabilities from 0 to 1
prior_probabilities = np.linspace(0, 1, 1000)

# Calculate posterior probabilities
posterior_probabilities = [bayes_theorem(prior, sensitivity, specificity) for prior in prior_probabilities]
#%%
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(prior_probabilities, posterior_probabilities, label='Posterior Probability $P(D|T)$', color='blue')
plt.xlabel('Prior Probability $P(D)$')
plt.ylabel('Posterior Probability $P(D|T)$')
plt.title('Posterior Probability as a Function of Prior Probability')
plt.grid(True)
plt.legend()
plt.show()
#%%
# Plotting on a semi-log scale
plt.figure(figsize=(8, 6))
plt.semilogx(prior_probabilities, posterior_probabilities, label='Posterior Probability $P(D|T)$', color='blue')
plt.xlabel('Prior Probability $P(D)$ (Log Scale)')
plt.ylabel('Posterior Probability $P(D|T)$')
plt.title('Posterior Probability for Low Prior Probabilities')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
