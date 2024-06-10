import os
import pandas as pd
import torch
import numpy as np
from dataset import BME, MyDataset, dataset_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from model.CNN_LSTM import Net # 确保这指向正确的SleepNet定义文件
import streamlit as st
import time
import base64
import matplotlib as plt


def binary_accuracy(preds, y):
    """
    计算二分类任务的准确率。
    """
    rounded_preds = torch.round(torch.sigmoid(preds))#首先将预测值应用 sigmoid 函数，然后对结果进行四舍五入。
    correct = (rounded_preds == y).float()#将四舍五入后的预测值与真实标签进行比较，得到一个布尔张量，然后将其转换为浮点张量。
    acc = correct.sum() / len(correct)#计算准确预测的数量并除以总样本数量，得到准确率。
    return acc.item()#返回准确率的数值部分

# def classify_and_count(predictions):
#     """
#     对模型输出的logits进行二分类，并统计0和1的数量。
#     """
#     # 假设您使用0.5作为阈值进行分类
#     binary_predictions = (predictions > 0.5).astype(int)
#     # 确保binary_predictions是一维的且元素为非负整数
#     if binary_predictions.ndim > 1:
#         binary_predictions = binary_predictions.flatten()
#
#     # 确保所有元素为非负整数
#     binary_predictions = np.where(binary_predictions >= 0, binary_predictions, 0).astype(int)
#     counts = np.bincount(binary_predictions)
#
#     if counts[0] >= counts[1]:  # 如果0的数量大于等于1的数量
#         return 0
#     else:
#         return 1
def classify_and_count(predictions):
    """
    对模型输出的logits进行二分类，并统计0和1的数量。
    """
    # 假设您使用0.5作为阈值进行分类
    binary_predictions = (predictions > 0.5).astype(int)
    # 确保binary_predictions是一维的且元素为非负整数
    if binary_predictions.ndim > 1:
        binary_predictions = binary_predictions.flatten()

    # 确保所有元素为非负整数，这一步在转换为int后其实已经保证了非负，可以考虑去掉
    binary_predictions = np.where(binary_predictions >= 0, binary_predictions, 0).astype(int)

    counts = np.bincount(binary_predictions)

    # 检查counts长度，如果只有一个类别
    if len(counts) == 1:
        # 直接返回该类别，因为binary_predictions中只有一种值，我们直接取其值即可
        unique_value = np.unique(binary_predictions)[0]
        return unique_value

    # 如果有两类，则按原逻辑判断
    if counts[0] >= counts[1]:
        return 0
    else:
        return 1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += binary_accuracy(outputs, labels.unsqueeze(1).float()) * inputs.size(0)

            outputs = torch.round(torch.sigmoid(outputs)).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            # 将预测结果和真实标签添加到列表中

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(labels, outputs)
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    # 计算敏感性（Se）和特异性（Sp）
    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)

    # 计算 F1 分数
    F1 = f1_score(labels, outputs)

    print(f"混淆矩阵:\n{cm}")
    print(f"敏感性: {Se}")
    print(f"特异性: {Sp}")
    print(f"F1 分数: {F1}")

    print(f'test Average loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc,Se,Sp,F1

def save_results_to_csv(predictions, filename='predict_results_3.csv', mode='a'):
    """
    将测试结果保存到CSV文件中，如果文件存在则追加。
    """
    # 生成序号列表，从1开始，长度与predictions相同
    ids = list(range(1, len(predictions) + 1))
    # 创建一个DataFrame，包含ID和Results两列
    df = pd.DataFrame({'ID': ids, 'Results': predictions})
    # 检查文件是否存在，不存在则创建
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode='w')
    else:
        # 如果文件存在，则以追加模式写入
        df.to_csv(filename, index=False, header=False, mode=mode)


def test_my_net(patient_index,network):
    # 实例化模型
    hidden_size = 32
    seq_len = 750
    is_bidirectional = True
    network = "LSTM"  # 或其他您在训练时使用的网络类型
    model = Net(hidden_size, seq_len, is_bidirectional, network)

    # 加载预训练模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置模型为评估模式

    # 示例：创建假数据进行测试，实际应用中应该使用真实测试集
    # 加载数据集
    bme = BME(time_len=3, time_step=2)
    bme.processing()
    # train_val_dataset = bme.load()
    # trainset, restset = dataset_split(train_val_dataset, val_rate=0.4)
    # valset, testset = dataset_split(restset, val_rate=0.25)
    testset = bme.load_test_data

    # 假设我们创建一个batch的测试数据，形状应符合模型输入要求
    batchsize = 32  # 选择一个适合测试的batch size
    # 处理数据集部分，实例化数据集，构建DataLoader
    # train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)  # [1, 2, 750]
    # val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    # test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    # test_data_indices = list(range(0, 15))  # 示例：使用前15个数据，根据实际情况调整
    test_data_indices = [patient_index]
    test_samples = bme.load_test_data(sub_index=test_data_indices)  # 调用方法并传入索引范围
    # testset = torch.utils.data.TensorDataset(torch.tensor(test_samples), torch.zeros(len(test_samples)))  # 假设标签全为0，仅作示例
    testset = torch.utils.data.TensorDataset(test_samples.clone().detach(),
                                             torch.zeros(len(test_samples), dtype=torch.float32).detach())
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    # 通过模型进行预测，这次使用真实的测试数据
    all_predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)  # 使用sigmoid将logits转化为概率
            # 假设 probabilities 是一个包含概率的张量，对于二分类问题
            rounded_probs = torch.round(probabilities)  # 四舍五入概率值
            predicted_classes = rounded_probs.detach().cpu().numpy().astype(int)  # 转换为numpy数组并确保是整数类型

            all_predictions.extend(predicted_classes)

    final_prediction = classify_and_count(np.array(all_predictions))
    st.write(f"最终预测样本的标签为: {final_prediction}")
    if final_prediction == 0:
        st.markdown(f'<span style="color:#8B0000; font-weight:bold; font-size:24px;">您的状态不太好，要好好对待自己噢😡</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color:green; font-weight:bold; font-size:24px;">您的身体很健康，请继续保持良好的状态噢😯</span>', unsafe_allow_html=True)
    print(f"Final Prediction (Based on Counts): {final_prediction}")
    # 保存预测结果到CSV
    save_results_to_csv([final_prediction])
    # _, _, Se, Sp, F1 = validate(model, test_loader, criterion=criterion, device=device)
    # print(f"Test Accuracy (Based on Validation): Not Directly Computed Here")  # 注意：这里的准确率需根据实际情况调整计算方式
    # print(f"Specificity: {Sp}")
    # print(f"Sensitivity: {Se}")
    # print(f"F1 Score: {F1}")



if __name__ == '__main__':
    # 假设您已经保存了模型权重到某个路径，例如'model.pth'
    model_path = r'D:\Pytharm\py_Project\Depression_Detect_new\Depression_Detect\EEG_Depression\all_best_model.pth'
    criterion = torch.nn.BCEWithLogitsLoss()
    # model = SleepNet(hidden_size=32, seq_len=750, is_bidirectional=True, network="LSTM")
    # 设置测试时使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    col1, col2 = st.columns([2,9])  # 3 和 2 是列的比例，可以根据需要调整

    with col1:  # 在第一列显示图片
        st.markdown(img_tag, unsafe_allow_html=True)

    with col2:  # 在第二列显示标题
        st.title("青少年抑郁症检测预防系统")
        st.markdown(f'''
                <div style="padding-left: 20px;">
                    <span style="color:black; font-weight:bold; font-style:italic; font-size:24px;">------欢迎来到“心境守护，抑郁智检”青少年抑郁检测平台👏👏</span>
                </div>
                ''', unsafe_allow_html=True)

    st.sidebar.title("🏠心境守护，抑郁智检")

    # 分隔线增强视觉区分
    st.sidebar.markdown("---")



    st.sidebar.markdown(img_tag, unsafe_allow_html=True)
    st.sidebar.header("参数选择：")
    # 在这里添加侧边栏的内容，例如：
    selected_option = st.sidebar.selectbox("请选择一个模型", ["长短期记忆网络（CNN-LSTM）", "CNN-GRU"])
    sex = st.sidebar.radio(
        label='请选择你的通道数',
        options=('单通道', '双通道','三通道'),
        index=0,
        format_func=str,
        help='选择心电，脑电，心脑电对应选择单通道，双通道和三通道'
    )
    cb = st.sidebar.checkbox('是否选择双向（默认为否）', value=False)
    selected_option2 = st.sidebar.selectbox("请选择测量模态", ["脑电单模态", "心电单模态", "心脑电双模态"])


    # 分隔线增强视觉区分
    st.markdown("---")

    st.header("参数预览（通过左边侧边栏的选项进行更改）")

    # 自定义CSS样式
    custom_style = """
    <style>
    .font-style {
        font-family: Arial, sans-serif; /* 更改字体 */
        font-size: 30px; /* 更改字体大小 */
        color: #000000; /* 更改字体颜色 */
    }
    </style>
    """

    # 应用自定义样式
    st.markdown(custom_style, unsafe_allow_html=True)




    if selected_option == "长短期记忆网络（CNN-LSTM）":
        selected_network = "LSTM"
        st.markdown("<p class='font-style'>1️⃣当前模型为：<strong>CNN-LSTM</strong></p>", unsafe_allow_html=True)
    elif selected_option == "CNN-GRU":
        selected_network = "GRU"
        st.markdown("<p class='font-style'>1️⃣当前模型为：<strong>CNN-GRU</strong></p>", unsafe_allow_html=True)
    else:
        st.error("ERROR")

    if sex == '单通道':
        st.markdown("<p class='font-style'>2️⃣当前通道为：<strong>单通道</strong></p>", unsafe_allow_html=True)
    elif sex == '双通道':
        st.markdown("<p class='font-style'>2️⃣当前通道为：<strong>双通道</strong></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='font-style'>2️⃣当前通道为：<strong>三通道</strong></p>", unsafe_allow_html=True)

    if cb:
        st.markdown("<p class='font-style'>3️⃣双向选择：<strong>是</strong></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='font-style'>3️⃣双向选择：<strong>否</strong></p>", unsafe_allow_html=True)

    if selected_option2 == "脑电单模态":
        selected_network = "脑电单模态"
        st.markdown("<p class='font-style'>4️⃣当前模态为：<strong>脑电单模态</strong></p>", unsafe_allow_html=True)
    elif selected_option2 == "心电单模态":
        selected_network = "心电单模态"
        st.markdown("<p class='font-style'>4️⃣当前模态为：<strong>心电单模态️</strong></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='font-style'>4️⃣当前模态为：<strong>心脑电双模态️</strong></p>", unsafe_allow_html=True)

    # 分隔线增强视觉区分
    st.markdown("---")

    st.header("检测区域（注：请选择正确的文件格式）")
    col1, col2 = st.columns(2)  # 使用列布局进行调整
    # 文件上传与检测区域

    # 心电文件上传
    ecg_uploaded_file = None
    with col1:
        uploaded_file1 = st.file_uploader("上传心电文件", type=["csv"])
        if uploaded_file1 is not None:
            if uploaded_file1.name != "ecgsignals.csv":
                st.warning("请上传正确格式的心电文件。")
            else:
                ecg_uploaded_file = uploaded_file1



    # 脑电文件上传
    eeg_uploaded_file = None
    with col2:
        uploaded_file2 = st.file_uploader("上传脑电文件", type=["csv"])
        if uploaded_file2 is not None:
            if uploaded_file2.name != "eegsignals.csv":
                st.warning("请上传正确格式的脑电文件。")
            else:
                eeg_uploaded_file = uploaded_file2



    patient_index_input = st.number_input("输入患者编号", min_value=0, max_value=14, value=0)  # 新增输入框
    patient_index_input = int(patient_index_input)
    detect_button = st.button("点击检测")

    import matplotlib.pyplot as plt




    # 分隔线增强视觉区分
    st.markdown("---")
    # 结果展示区域
    st.header("⌛️检测结果")


    result_placeholder = st.empty()
    progress_bar = st.empty()


    def show_progress():
        """模拟进度条动画"""

        for i in range(21):
            progress_bar.progress(i / 20, '进度')
            time.sleep(0.5)

        with st.spinner('加载中...'):
            time.sleep(5)


    if detect_button and uploaded_file1 is not None and uploaded_file2 is not None:
        result_placeholder.empty()
        progress_bar.progress(0)
        try:
            show_progress()
            test_my_net(patient_index_input,selected_network)
            # 假设test_sleep_net是处理上传文件并返回结果的函数，需确保该函数已定义
            result_placeholder.success(f"操作成功，结果如下。\n\n")
            # 添加超链接
            link_text = "[<p class='font-style'>—>更多预防青少年抑郁症的知识，点击此处前往<—</p>](https://xueshu.baidu.com/s?wd=%E9%9D%92%E5%B0%91%E5%B9%B4%E6%8A%91%E9%83%81%E7%97%87%E9%A2%84%E9%98%B2&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_hit=1)"  # 请将https://www.example.com/depression-prevention替换为实际链接
            st.markdown(link_text, unsafe_allow_html=True)
        except Exception as e:
            result_placeholder.error(f"检测过程中出现错误: {e}")
    else:
        st.info("请确保成功上传文件并选择患者编号后点击按钮进行检测。")
