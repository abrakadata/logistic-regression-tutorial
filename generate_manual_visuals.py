import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


OUT_DIR = r"c:\Users\awild\Desktop\MDC Projects and Homework\Extra Curricular\images\manual_visuals"


def chapter1_foundations_visuals():
    # 1) AI vs ML vs Deep Learning nested map
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")

    rectangles = [
        (0.08, 0.12, 0.84, 0.76, "AI", "#DCEBFF"),
        (0.22, 0.24, 0.56, 0.52, "Machine Learning", "#B8D9FF"),
        (0.35, 0.36, 0.30, 0.28, "Deep Learning", "#7FB8FF"),
    ]

    for x, y, w, h, label, color in rectangles:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#2C5282", linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12, weight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter1_ai_ml_dl_map.png", dpi=160)
    plt.close()

    # 2) Basic ML workflow
    steps = [
        "Define",
        "Collect",
        "Clean",
        "Split",
        "Train",
        "Evaluate",
        "Improve",
        "Use",
    ]

    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")

    for i, step in enumerate(steps):
        x = 0.05 + i * 0.11
        ax.text(
            x,
            0.5,
            step,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#E9F7EF", "edgecolor": "#2F855A"},
            transform=ax.transAxes,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 0.055, 0.5),
                xytext=(x + 0.085, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#2F855A"},
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter1_workflow.png", dpi=160)
    plt.close()


def chapter2_data_basics_visuals():
    # 1) Dataset anatomy table
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    cell_text = [
        ["hours_studied", "attendance", "passed"],
        ["2", "60", "0"],
        ["5", "80", "1"],
        ["7", "95", "1"],
    ]

    table = ax.table(cellText=cell_text, loc="center", cellLoc="center")
    table.scale(1, 1.8)

    ax.text(0.28, 0.82, "Features", ha="center", va="bottom", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.76, 0.82, "Label", ha="center", va="bottom", fontsize=11, weight="bold", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter2_dataset_anatomy.png", dpi=160)
    plt.close()

    # 2) Safe split vs leakage risk
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")

    left_boxes = [
        (0.08, 0.62, "Raw Data"),
        (0.25, 0.62, "Split First"),
        (0.42, 0.62, "Fit Prep on Train"),
        (0.63, 0.62, "Evaluate on Test"),
    ]
    right_boxes = [
        (0.08, 0.20, "Raw Data"),
        (0.25, 0.20, "Prep Full Data"),
        (0.46, 0.20, "Split Later"),
        (0.63, 0.20, "Fake Good Score"),
    ]

    for x, y, label in left_boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#E6FFFA", "edgecolor": "#2C7A7B"},
            transform=ax.transAxes,
        )
    for x, y, label in right_boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF5F5", "edgecolor": "#C53030"},
            transform=ax.transAxes,
        )

    for i in range(len(left_boxes) - 1):
        ax.annotate(
            "",
            xy=(left_boxes[i][0] + 0.07, left_boxes[i][1]),
            xytext=(left_boxes[i + 1][0] - 0.07, left_boxes[i + 1][1]),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#2C7A7B"},
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )
    for i in range(len(right_boxes) - 1):
        ax.annotate(
            "",
            xy=(right_boxes[i][0] + 0.07, right_boxes[i][1]),
            xytext=(right_boxes[i + 1][0] - 0.07, right_boxes[i + 1][1]),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#C53030"},
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter2_split_leakage.png", dpi=160)
    plt.close()


def chapter3_regression_plot():
    data = pd.DataFrame({
        "size_sqft": [600, 750, 800, 900, 1100, 1200, 1400, 1600, 1800, 2000],
        "bedrooms": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        "price": [150, 180, 200, 220, 260, 280, 320, 360, 390, 430],
    })

    x = data["size_sqft"].values.reshape(-1, 1)
    y = data["price"].values

    model = LinearRegression()
    model.fit(x, y)

    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure(figsize=(7, 4))
    plt.scatter(x, y, label="Actual points")
    plt.plot(x_line, y_line, label="Best-fit line")
    plt.xlabel("size_sqft")
    plt.ylabel("price (thousands)")
    plt.title("Regression: Data Points and Best-Fit Line")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter3_regression_line.png", dpi=160)
    plt.close()


def chapter4_confusion_matrix_heatmap():
    data = pd.DataFrame({
        "hours_studied": [1, 2, 3, 4, 5, 6, 2, 7, 8, 3, 5, 6],
        "attendance": [45, 55, 60, 70, 75, 80, 50, 88, 92, 65, 78, 85],
        "passed": [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    })

    X = data[["hours_studied", "attendance"]]
    y = data["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter4_confusion_matrix.png", dpi=160)
    plt.close()


def chapter5_tree_and_importance():
    data = pd.DataFrame({
        "tenure_months": [1, 2, 3, 6, 8, 12, 18, 24, 30, 36, 4, 10, 20, 28, 40],
        "monthly_bill": [95, 88, 92, 70, 65, 55, 60, 58, 62, 59, 90, 72, 64, 61, 57],
        "support_calls": [5, 4, 4, 2, 1, 1, 2, 1, 1, 0, 5, 2, 1, 1, 0],
        "month_to_month": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        "churn": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    })

    X = data[["tenure_months", "monthly_bill", "support_calls", "month_to_month"]]
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)

    forest = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    forest.fit(X_train, y_train)

    plt.figure(figsize=(12, 6))
    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=["No Churn", "Churn"],
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter5_decision_tree.png", dpi=160)
    plt.close()

    importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(7, 4))
    importances.plot(kind="bar")
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter5_feature_importance.png", dpi=160)
    plt.close()


def chapter6_cluster_and_elbow():
    data = pd.DataFrame({
        "monthly_spend": [20, 25, 30, 120, 130, 140, 55, 60, 65, 200, 210, 220],
        "visits_per_month": [1, 1, 2, 8, 9, 9, 4, 5, 4, 12, 13, 12],
        "avg_items_per_order": [1, 2, 2, 6, 7, 6, 3, 3, 4, 9, 10, 9],
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data["cluster"] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(data["monthly_spend"], data["visits_per_month"], c=data["cluster"])
    plt.xlabel("monthly_spend")
    plt.ylabel("visits_per_month")
    plt.title("Customer Clusters (2D View)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter6_cluster_scatter.png", dpi=160)
    plt.close()

    ks = range(1, 8)
    inertias = []

    for k in ks:
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        m.fit(X_scaled)
        inertias.append(m.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter6_elbow_plot.png", dpi=160)
    plt.close()


def chapter7_improvement_visuals():
    data = pd.DataFrame({
        "tenure_months": [1, 2, 3, 6, 8, 12, 18, 24, 30, 36, 4, 10, 20, 28, 40, 5, 7, 14, 22, 34],
        "monthly_bill": [95, 88, 92, 70, 65, 55, 60, 58, 62, 59, 90, 72, 64, 61, 57, 85, 80, 68, 63, 58],
        "support_calls": [5, 4, 4, 2, 1, 1, 2, 1, 1, 0, 5, 2, 1, 1, 0, 4, 3, 2, 1, 1],
        "month_to_month": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        "churn": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    })

    X = data[["tenure_months", "monthly_bill", "support_calls", "month_to_month"]]
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    baseline = RandomForestClassifier(random_state=42)
    baseline.fit(X_train, y_train)
    base_preds = baseline.predict(X_test)
    base_f1 = f1_score(y_test, base_preds)

    cv_scores = cross_val_score(
        RandomForestClassifier(random_state=42),
        X_train,
        y_train,
        cv=5,
        scoring="f1",
    )

    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [2, 3, 4, None],
        "min_samples_leaf": [1, 2, 3],
    }

    search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_preds = search.best_estimator_.predict(X_test)
    tuned_f1 = f1_score(y_test, tuned_preds)

    labels = ["Baseline F1", "CV Mean F1", "Tuned Test F1"]
    values = [base_f1, cv_scores.mean(), tuned_f1]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Quality After Improvement Steps")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter7_score_comparison.png", dpi=160)
    plt.close()

    depths = [1, 2, 3, 4, 5, None]
    means = []
    for d in depths:
        m = RandomForestClassifier(max_depth=d, random_state=42)
        s = cross_val_score(m, X_train, y_train, cv=5, scoring="f1")
        means.append(s.mean())

    depth_labels = [str(d) for d in depths]

    plt.figure(figsize=(6, 4))
    plt.plot(depth_labels, means, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("max_depth")
    plt.ylabel("CV mean F1")
    plt.title("Effect of max_depth on Cross-Validated F1")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter7_depth_curve.png", dpi=160)
    plt.close()


def chapter8_workflow_and_monitoring_visuals():
    # 1) Workflow map
    steps = [
        "Define Goal",
        "Prepare Data",
        "Train + Evaluate",
        "Save Model",
        "Serve Predictions",
        "Monitor + Retrain",
    ]

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")

    for i, step in enumerate(steps):
        x = 0.06 + i * 0.155
        ax.text(
            x,
            0.5,
            step,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#E8F0FE", "edgecolor": "#4A67A1"},
            transform=ax.transAxes,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 0.075, 0.5),
                xytext=(x + 0.105, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#4A67A1"},
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter8_workflow_map.png", dpi=160)
    plt.close()

    # 2) Monitoring trend plot
    weeks = ["W1", "W2", "W3", "W4", "W5", "W6"]
    f1_scores = [0.88, 0.87, 0.86, 0.83, 0.80, 0.78]
    alert_threshold = 0.82

    plt.figure(figsize=(6.5, 4))
    plt.plot(weeks, f1_scores, marker="o", label="Weekly F1")
    plt.axhline(alert_threshold, linestyle="--", label="Alert Threshold")
    plt.ylim(0.7, 0.92)
    plt.xlabel("Week")
    plt.ylabel("F1 Score")
    plt.title("Example Monitoring Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter8_monitoring_trend.png", dpi=160)
    plt.close()


def chapter9_responsible_ml_visuals():
    # 1) Fairness rate chart
    fairness_df = pd.DataFrame({
        "group": ["A", "B"],
        "positive_prediction_rate": [0.25, 0.75],
    })

    plt.figure(figsize=(6.5, 4))
    plt.bar(fairness_df["group"], fairness_df["positive_prediction_rate"])
    plt.ylim(0, 1)
    plt.xlabel("Group")
    plt.ylabel("Positive Prediction Rate")
    plt.title("Fairness Snapshot: Positive Prediction Rate by Group")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter9_fairness_rates.png", dpi=160)
    plt.close()

    # 2) Example feature importance chart
    feature_names = ["monthly_bill", "tenure_months", "support_calls", "month_to_month"]
    importance = [0.35, 0.25, 0.20, 0.20]

    plt.figure(figsize=(7, 4))
    plt.bar(feature_names, importance)
    plt.ylabel("Importance")
    plt.title("Global Feature Importance Example")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter9_feature_importance_example.png", dpi=160)
    plt.close()


def chapter10_capstone_visuals():
    # 1) Capstone lifecycle roadmap
    stages = [
        "Problem",
        "Data",
        "Baseline",
        "Improve",
        "Validate",
        "Report",
    ]

    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")

    for i, stage in enumerate(stages):
        x = 0.08 + i * 0.15
        ax.text(
            x,
            0.5,
            stage,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EAF7EF", "edgecolor": "#2D6A4F"},
            transform=ax.transAxes,
        )
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 0.075, 0.5),
                xytext=(x + 0.105, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#2D6A4F"},
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter10_capstone_roadmap.png", dpi=160)
    plt.close()

    # 2) Baseline vs tuned capstone performance summary
    labels = ["Baseline F1", "CV Mean F1", "Tuned Test F1"]
    values = [0.80, 0.83, 0.87]

    plt.figure(figsize=(6.2, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Capstone Model Performance Summary")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}\\chapter10_capstone_performance.png", dpi=160)
    plt.close()


def main():
    chapter1_foundations_visuals()
    chapter2_data_basics_visuals()
    chapter3_regression_plot()
    chapter4_confusion_matrix_heatmap()
    chapter5_tree_and_importance()
    chapter6_cluster_and_elbow()
    chapter7_improvement_visuals()
    chapter8_workflow_and_monitoring_visuals()
    chapter9_responsible_ml_visuals()
    chapter10_capstone_visuals()
    print("Generated manual visuals in:", OUT_DIR)


if __name__ == "__main__":
    main()
