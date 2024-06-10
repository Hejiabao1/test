import os
import pandas as pd
import torch
import numpy as np

import streamlit as st
import base64
import matplotlib as plt
import matplotlib.pyplot as plt


@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def page_bg(jpg_file):
    """设置网页背景图片"""
    bin_str = get_base64_of_bin_file(jpg_file)
    page_bg_css = f"""
          <style>
          .stApp {{
              background-image: url("data:image/jpeg;base64,{bin_str}");
              background-size: 100% 120%;
              background-repeat: no-repeat;
              background-attachment: center;
          }}
          </style>
          """

    st.markdown(page_bg_css, unsafe_allow_html=True)


# 设置背景图片
page_bg(r"D:\Pytharm\py_Project\Depression_Detect_new\Depression_Detect\EEG_Depression\yiyuzheng.jpg")


# 加载图片
# 图片转Base64编码函数
def get_image_base64(image_filename):
    with open(image_filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


image_filename = "logo.png"  # 图片文件名
image_base64 = get_image_base64(image_filename)
img_tag = f'<img src="data:image/png;base64,{image_base64}" alt="侧边栏图片" style="width: 100%; max-width: 200px; display: block; margin: auto;">'
# 设置两列布局，第一列宽度设为图片合适的比例，第二列自动填充剩余空间
col1, col2 = st.columns([2, 9])  # 3 和 2 是列的比例，可以根据需要调整

with col1:  # 在第一列显示图片
    st.markdown(img_tag, unsafe_allow_html=True)

with col2:  # 在第二列显示标题
    st.title("患者心脑电波形展示平台")
    st.markdown(f'''
               <div style="padding-left: 20px;">
                   <span style="color:black; font-weight:bold; font-style:italic; font-size:24px;">------欢迎来到“心境守护，抑郁智检”青少年抑郁检测平台👏👏👏👏</span>
               </div>
               ''', unsafe_allow_html=True)

st.sidebar.title("🏠心境守护，抑郁智检")

# 分隔线增强视觉区分
st.sidebar.markdown("---")
st.sidebar.markdown(img_tag, unsafe_allow_html=True)


# 使用pandas读取CSV文件
@st.cache_data  # 这个装饰器可以帮助缓存数据，加快应用响应速度
def load_data(file_path):
    return pd.read_csv(file_path)



def load_data(uploaded_file):
    """加载并处理上传的CSV文件"""
    return pd.read_csv(uploaded_file)  # 假设索引是信号的连续编号或样本点



# 分隔线增强视觉区分
st.markdown("---")

st.header("ECG波形：")

# 文件上传组件
uploaded_file = st.file_uploader("上传你的CSV文件", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # 确认数据加载无误
    if not df.empty:
        # 允许用户选择用于绘制的列，默认选择第一列，即列名为'0'
        column_options = df.columns.tolist()  # 列名列表
        selected_column = st.selectbox("选择要显示的信号列", column_options, index=0)

        # 添加滑块选择信号的样本点范围
        sample_start = st.slider("信号起始样本点", min_value=0, max_value=len(df) - 1, value=0, step=1)
        sample_end = st.slider("信号结束样本点", min_value=sample_start, max_value=len(df) - 1, value=len(df) - 1,
                                     step=1)

        # 根据用户选择的范围筛选数据
        filtered_df = df.iloc[sample_start:sample_end + 1]

        # 绘制信号图
        plt.figure(figsize=(10, 5))
        plt.plot(filtered_df[selected_column])
        st.pyplot(plt)
    else:
        st.error("文件加载为空，请检查CSV文件内容。")
else:
    st.info("请上传一个ECG(.csv)文件。")


def load_eeg_data(uploaded_file):
    """加载并处理上传的EEG CSV文件"""
    return pd.read_csv(uploaded_file)  # 假设索引是信号的连续时间点或样本编号

 # 分隔线增强视觉区分
st.markdown("---")

# Streamlit应用开始
st.header("EEG波形：")

# 文件上传组件
uploaded_file = st.file_uploader("上传你的EEG CSV文件", type=["csv"])

if uploaded_file is not None:
    df = load_eeg_data(uploaded_file)

    # 确认数据加载无误
    if not df.empty:
        # 允许用户选择显示哪些通道，默认都选中
        show_channel_0 = st.checkbox('显示通道 0', value=True)
        show_channel_1 = st.checkbox('显示通道 1', value=True)

        # 添加滑块选择信号的样本点范围
        sample_start = st.slider("信号起始样本点", min_value=0, max_value=len(df) - 1, value=0, step=1)
        sample_end = st.slider("信号结束样本点", min_value=sample_start, max_value=len(df) - 1, value=len(df) - 1,
                                     step=1)

        # 根据用户选择的范围和通道筛选数据
        filtered_df = df.iloc[sample_start:sample_end + 1]

        # 准备绘图
        fig, ax = plt.subplots(figsize=(10, 5))

        if show_channel_0:
            ax.plot(filtered_df['0'], label='通道 0')
        if show_channel_1:
            ax.plot(filtered_df['1'], label='通道 1')

        ax.legend()

        st.pyplot(fig)
    else:
        st.error("文件加载为空，请检查CSV文件内容。")
else:
    st.info("请上传一个EEG(.csv)文件。")

st.write("(⚠️：一次只能上传一个文件！)")