import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "TG": {"type": "numerical", "min": 0.0, "max": 300.0, "default": 150.0},
    "ALT": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 40.0},
    "HDL-C": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 50.0},
    "Eosinophil": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 2.0},
    "Ures": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 25.0},
    "PA": {"type": "numerical", "min": 0.0, "max": 300.0, "default": 100.0},
    "RDW-SD": {"type": "numerical", "min": 30.0, "max": 60.0, "default": 45.0},
    "D-D": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 0.5},
    "APTT": {"type": "numerical", "min": 20.0, "max": 60.0, "default": 32.0},
    "GLB": {"type": "numerical", "min": 10.0, "max": 100.0, "default": 40.0},
    "CCP": {"type": "numerical", "min": 0.0, "max": 300.0, "default": 20.0},
    "AST": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 24.0},
    "ALB": {"type": "numerical", "min": 30.0, "max": 60.0, "default": 45.0},
    "UA": {"type": "numerical", "min": 0.0, "max": 15.0, "default": 5.0},
    "LCL-C": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 100.0},
    "Arthralgia": {"type": "categorical", "options": [0, 1, 2], "default": 0},
    "P-LCR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 23.0},
    "WBC": {"type": "numerical", "min": 3.0, "max": 15.0, "default": 6.0},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
            index=properties["options"].index(properties["default"])
        )
    feature_values.append(value)

# 转换为 DataFrame（保证特征顺序正确）
input_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(input_df)[0]
    predicted_proba = model.predict_proba(input_df)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果（使用 Matplotlib 绘制文本）
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

    # 显示所有类别概率
    st.subheader("Class probabilities:")
    proba_df = pd.DataFrame([predicted_proba], columns=[f"Class {i}" for i in range(len(predicted_proba))])
    st.table(proba_df)

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 二分类 / 多分类处理
    if isinstance(shap_values, list):  # 多分类
        shap_values_for_class = shap_values[predicted_class]
        expected_value = explainer.expected_value[predicted_class]
    else:  # 二分类
        shap_values_for_class = shap_values
        expected_value = explainer.expected_value

    # 生成 SHAP 力图
    shap_fig = shap.force_plot(
        expected_value,
        shap_values_for_class[0, :],
        input_df.iloc[0, :],
        matplotlib=True
    )

    st.pyplot(shap_fig)
