# Logistic Regression From Scratch
# Predicting student pass/fail

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # used by 2D plot legend

# ── Training data (90 students) ──────────────────────────────
# [attendance%, study_hours]
X = [
    # Original 18 samples
    [90, 8], [30, 2], [75, 6], [50, 3],
    [85, 7], [20, 1], [65, 5], [40, 2],
    [95, 9], [72, 5], [60, 4], [88, 6],
    [35, 1], [55, 2], [78, 7], [45, 3],
    [58, 6], [82, 3],  # intentional noise
    # Additional clear PASS (high attendance + solid study time)
    [92, 8], [87, 9], [93, 7], [80, 8], [77, 6], [83, 7], [91, 9],
    [70, 6], [86, 8], [79, 7], [84, 6], [96, 10], [73, 5], [89, 8],
    [76, 7], [94, 9], [81, 6], [74, 5], [71, 6], [97, 10],
    [82, 8], [98, 7], [69, 5], [87, 6], [92, 9], [73, 7], [80, 5], [77, 8],
    # Additional clear FAIL (low attendance + minimal study)
    [25, 1], [32, 2], [18, 1], [42, 2], [28, 1],
    [37, 2], [22, 1], [48, 3], [33, 2], [15, 1],
    [43, 2], [27, 1], [38, 3], [23, 1], [46, 2],
    [31, 2], [19, 1], [44, 3], [36, 2], [26, 1],
    [41, 2], [29, 2], [34, 1], [47, 3], [21, 1],
    [39, 2], [24, 1], [16, 1], [49, 3], [50, 2],
    # Borderline students (moderate attendance + moderate study)
    # Expanded to give the model a clearer signal in the fuzzy decision region
    [62, 5], [57, 4], [68, 5], [53, 3], [63, 4],
    [66, 5], [52, 3], [67, 6], [59, 4], [61, 5],
    [64, 5], [56, 3], [60, 5], [54, 4], [65, 6],
    [51, 3], [69, 5], [58, 4], [64, 3], [62, 6],
    [53, 5], [57, 5], [55, 3], [63, 5], [66, 4],
    # Additional noisy cases (counter-intuitive outcomes)
    [47, 8], [88, 2], [38, 7], [85, 1],
]
y = [
    # Original 18
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 0, 1, 0, 0, 1, 0,
    1, 0,
    # Additional clear PASS (28)
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    # Additional clear FAIL (30)
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    # Borderline (35): pass if study >= 5, fail if study <= 4
    1, 0, 1, 0, 0,
    1, 0, 1, 0, 1,
    1, 0, 1, 0, 1,
    0, 1, 0, 0, 1,
    1, 1, 0, 1, 0,
    # Noisy (4): low-att hard worker passes; high-att non-studier fails
    1, 0, 1, 0,
] # 1=pass, 0=fail

# ── Test data (50 students, unseen during training) ────────────
X_test = [
    # Clear PASS
    [91, 7], [86, 9], [78, 6], [93, 8], [74, 5],
    [83, 8], [71, 7], [96, 9], [82, 6], [88, 7],
    [75, 8], [90, 6], [79, 5], [84, 9], [70, 7],
    [95, 8], [76, 6], [85, 5], [72, 8], [89, 7],
    # Clear FAIL
    [22, 2], [44, 1], [17, 1], [38, 2], [29, 3],
    [33, 1], [46, 2], [26, 1], [41, 3], [19, 2],
    [35, 1], [48, 2], [24, 1], [43, 3], [31, 2],
    [27, 1], [37, 2], [20, 3], [45, 1], [32, 2],
    # Borderline
    [64, 5], [56, 4], [61, 3], [66, 4], [53, 5],
    [58, 3], [63, 6], [55, 4],
    # Noisy (low-att but hard worker; high-att but barely studies)
    [43, 9], [91, 1],
]
y_test = [
    # Clear PASS (20)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # Clear FAIL (20)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # Borderline (8): pass if study >= 5, fail otherwise
    1, 0, 0, 0, 1, 0, 1, 0,
    # Noisy (2)
    1, 0,
]

attendance_values = [student[0] for student in X]
study_hour_values = [student[1] for student in X]
attendance_min = min(attendance_values)
attendance_max = max(attendance_values)
study_hour_min = min(study_hour_values)
study_hour_max = max(study_hour_values)


def normalize_feature(value, minimum, maximum):
    return (value - minimum) / (maximum - minimum)


def normalize_student(attendance, study_hours):
    normalized_attendance = normalize_feature(attendance, attendance_min, attendance_max)
    normalized_study_hours = normalize_feature(study_hours, study_hour_min, study_hour_max)
    return normalized_attendance, normalized_study_hours


X_normalized = [list(normalize_student(student[0], student[1])) for student in X]

# ── Sigmoid function ───────────────────────
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# ── Initialize weights randomly ────────────
w1 = 0.0
w2 = 0.0
b = 0.0
lr_initial = 0.1   # higher starting rate for faster early progress
lr_decay  = 0.001  # decay factor; lr = lr_initial / (1 + decay * epoch)
epochs = 8000

# ── Training loop ──────────────────────────
for epoch in range(epochs):
    # Step 1: Forward pass — compute predictions
    predictions = []

    for student in X_normalized:
        z = w1 * student[0] + w2 * student[1] + b
        p = sigmoid(z)
        predictions.append(p)

    # Step 2: Compute loss (Binary Cross-Entropy)
    loss = 0
    for i in range(len(y)):
        p = predictions[i]
        loss -= y[i] * math.log(p + 1e-9)
        loss -= (1 - y[i]) * math.log(1 - p + 1e-9)
    loss /= len(y)

    # Step 3: Compute gradients
    grad_w1 = 0
    grad_w2 = 0
    grad_b = 0
    N = len(y)
    for i in range(N):
        error = predictions[i] - y[i]
        grad_w1 += error * X_normalized[i][0]
        grad_w2 += error * X_normalized[i][1]
        grad_b += error
    grad_w1 /= N
    grad_w2 /= N
    grad_b /= N

    # Step 4: Update weights with decaying learning rate
    lr = lr_initial / (1 + lr_decay * epoch)
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
    b -= lr * grad_b

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}: Loss = {loss:.4f}  lr = {lr:.5f}")

# ── Training accuracy ──────────────────────
correct = 0
for i in range(len(y)):
    predicted_label = 1 if predictions[i] >= 0.5 else 0
    if predicted_label == y[i]:
        correct += 1
accuracy = correct / len(y)
print(f"\nTraining accuracy: {accuracy:.1%}  ({correct}/{len(y)} correct)")

confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
for i in range(len(y)):
    predicted_label = 1 if predictions[i] >= 0.5 else 0
    if y[i] == 1 and predicted_label == 1:
        confusion["tp"] += 1
    elif y[i] == 0 and predicted_label == 1:
        confusion["fp"] += 1
    elif y[i] == 0 and predicted_label == 0:
        confusion["tn"] += 1
    else:
        confusion["fn"] += 1
print(f"TP (True Positive)={confusion['tp']}  FP (False Positive)={confusion['fp']}  TN (True Negative)={confusion['tn']}  FN (False Negative)={confusion['fn']}")

# ── Prediction function ────────────────────
def predict_probability(attendance, study_hours):
    normalized_attendance, normalized_study_hours = normalize_student(attendance, study_hours)
    z = w1 * normalized_attendance + w2 * normalized_study_hours + b
    return sigmoid(z)


def predict(attendance, study_hours):
    probability = predict_probability(attendance, study_hours)
    label = "PASS" if probability >= 0.5 else "FAIL"
    print(f"P(pass) = {probability:.1%} -> {label}")

# ── Test set evaluation ───────────────────
print("\n-- Test Set Results ----------------------")
test_correct = 0
test_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
for i in range(len(y_test)):
    prob = predict_probability(X_test[i][0], X_test[i][1])
    predicted_label = 1 if prob >= 0.5 else 0
    if predicted_label == y_test[i]:
        test_correct += 1
    if y_test[i] == 1 and predicted_label == 1:
        test_confusion["tp"] += 1
    elif y_test[i] == 0 and predicted_label == 1:
        test_confusion["fp"] += 1
    elif y_test[i] == 0 and predicted_label == 0:
        test_confusion["tn"] += 1
    else:
        test_confusion["fn"] += 1
test_accuracy = test_correct / len(y_test)
print(f"Test accuracy:    {test_accuracy:.1%}  ({test_correct}/{len(y_test)} correct)")
print(f"TP (True Positive)={test_confusion['tp']}  FP (False Positive)={test_confusion['fp']}  TN (True Negative)={test_confusion['tn']}  FN (False Negative)={test_confusion['fn']}")

# ── 2D top-down decision boundary plot ────
def plot_model_2d():
    att_grid = np.linspace(attendance_min - 5, attendance_max + 5, 300)
    study_grid = np.linspace(study_hour_min - 1, study_hour_max + 1, 300)
    att_mesh, study_mesh = np.meshgrid(att_grid, study_grid)

    norm_att_mesh = normalize_feature(att_mesh, attendance_min, attendance_max)
    norm_study_mesh = normalize_feature(study_mesh, study_hour_min, study_hour_max)
    z_values = w1 * norm_att_mesh + w2 * norm_study_mesh + b
    prob_surface = 1 / (1 + np.exp(-z_values))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Filled probability background
    contour_filled = ax.contourf(att_mesh, study_mesh, prob_surface, levels=50, cmap="viridis", alpha=0.75)
    fig.colorbar(contour_filled, ax=ax, label="Predicted pass probability")

    # Decision boundary line (P = 0.5)
    ax.contour(att_mesh, study_mesh, prob_surface, levels=[0.5], colors="white", linewidths=2)

    # Training points — shape encodes true label, colour encodes predicted probability
    training_probabilities = [predict_probability(s[0], s[1]) for s in X]
    noisy_indices = {16, 17}  # indices of the two intentionally noisy cases

    for index, student in enumerate(X):
        marker = "o" if y[index] == 1 else "X"
        edgecolor = "red" if index in noisy_indices else "black"
        linewidth = 2 if index in noisy_indices else 0.8
        ax.scatter(
            student[0],
            student[1],
            c=training_probabilities[index],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=110,
            marker=marker,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=3,
        )

    # Annotate noisy cases
    ax.annotate("Noise: pass\n(low att, passes)", xy=(58, 6), xytext=(62, 6.5),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax.annotate("Noise: fail\n(high att, fails)", xy=(82, 3), xytext=(70, 2.2),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="red"), color="red")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="True label: PASS", markerfacecolor="gray", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="X", color="w", label="True label: FAIL", markerfacecolor="gray", markeredgecolor="black", markersize=9),
        Line2D([0], [0], color="white", label="Decision boundary (P=0.5)", linewidth=2),
        Line2D([0], [0], marker="o", color="w", label="Noisy case", markerfacecolor="gray", markeredgecolor="red", markersize=9, markeredgewidth=2),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.85)

    ax.set_title("Logistic Regression — 2D Top-Down View")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Study Hours")
    plt.tight_layout()
    plt.show()


# ── Try it out ─────────────────────────────
predict(80, 7) # Should be PASS
predict(25, 2) # Should be FAIL
predict(55, 4) # Borderline case
plot_model_2d()