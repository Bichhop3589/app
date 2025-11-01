import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

#LOAD MODEL & ARTIFACTS

try:
    model_reg = joblib.load("model_reg.joblib")
    model_cls = joblib.load("model_cls.joblib")
    model_ord = joblib.load("model_ord.joblib")
    ordinal_enc = joblib.load("ordinal_enc.joblib")

    with open("final_columns.json", "r") as f:
        final_columns = json.load(f)
    with open("best_params.json", "r") as f:
        best_params = json.load(f)

# Load trọng số ensemble
    w_reg, w_cls, w_ord = best_params["w_reg"], best_params["w_cls"], best_params["w_ord"]
    cut1, cut2 = best_params["cutoff_1"], best_params["cutoff_2"]
# =================================================================
    
    
    # 1. Thay 'Bachelors' bằng kết quả .mode() của bạn
    EDUCATION_MODE = 'Bachelors' 
    
    # 2. Thay [min, max] bằng kết quả .quantile() của bạn
    AGE_BOUNDS = [25.0, 51.0] 
    SERVICE_YEARS_BOUNDS = [1.0, 15.0]
    TRAINING_COUNT_BOUNDS = [1.0, 2.0]
    
    # 3. Dán list region hiếm của bạn vào đây
    RARE_REGIONS_LIST = [
        'region_1', 'region_24', 'region_12', 'region_9', 'region_21', 'region_34', 'region_3', 'region_33', 'region_18'
    ]
except FileNotFoundError:
    st.error("LỖI: Không tìm thấy file model hoặc file .json.")
    st.stop()
except KeyError:
    st.error("LỖI: File 'best_params.json' bị lỗi.")
    st.stop()   
# =============================
# HÀM XỬ LÝ DỮ LIỆU ĐẦU VÀO
# =============================
def preprocess_raw(df_raw):
    df = df_raw.copy()

    # --- Đổi tên cột ---
    new_column_names = {
        'employee_id': 'employee_id',
        'department': 'department',
        'region': 'region',
        'education': 'education',
        'gender': 'gender',
        'recruitment_channel': 'recruitment_channel',
        'no_of_trainings': 'training_count',
        'age': 'age',
        'length_of_service': 'service_years',
        'KPIs_met_more_than_80': 'KPIs_met_more_than_80',
        'awards_won': 'awards_won',
        'avg_training_score': 'avg_training_score'
    }
    df = df.rename(columns=new_column_names)

    # --- Fill missing 
    df['education'] = df['education'].fillna(EDUCATION_MODE) # Dùng giá trị hardcode

    # --- Drop duplicate
    if 'employee_id' in df.columns:
        df = df.drop_duplicates(subset='employee_id', keep='first')

    # --- Winsorize 
    df['age'] = df['age'].clip(AGE_BOUNDS[0], AGE_BOUNDS[1]) # Dùng giá trị hardcode
    df['service_years'] = df['service_years'].clip(SERVICE_YEARS_BOUNDS[0], SERVICE_YEARS_BOUNDS[1])
    df['training_count'] = df['training_count'].clip(TRAINING_COUNT_BOUNDS[0], TRAINING_COUNT_BOUNDS[1])

    # --- Gộp region hiếm 
    df['region'] = df['region'].apply(lambda x: 'Other_Region' if x in RARE_REGIONS_LIST else x) # Dùng list hardcode

    # --- Feature engineering 
    df['age_at_joining'] = df['age'] - df['service_years']
    df['service_ratio'] = df['service_years'] / (df['age'] - 18).replace(0, np.nan)
    df.loc[df['age_at_joining'] < 18, 'age_at_joining'] = np.nan
    df.loc[df['service_ratio'] < 0, 'service_ratio'] = np.nan
    df['age_at_joining'] = df['age_at_joining'].fillna(0)
    df['service_ratio'] = df['service_ratio'].fillna(0)

    def map_experience_group(x):
        if 1 <= x <= 2: return 'Newbie'
        elif 3 <= x <= 5: return 'Junior'
        elif 6 <= x <= 9: return 'Mid'
        elif 10 <= x <= 14: return 'Senior'
        elif x >= 15: return 'Veteran'
        else: return np.nan
    df['experience_group'] = df['service_years'].apply(map_experience_group)
    df['experience_group'] = df['experience_group'].fillna('Newbie') 

    df['total_training_impact'] = df['training_count'] * df['avg_training_score']
    df['training_per_year'] = df['training_count'] / (df['service_years'] + 1)
    df['award_per_year'] = df['awards_won'] / (df['service_years'] + 1)
    df['talent_signal'] = df['avg_training_score'] * df['awards_won']
    df['KPI_and_Award'] = df['KPIs_met_more_than_80'] * df['awards_won']

    df['dept_region'] = df['department'] + '_' + df['region']
    df['dept_education'] = df['department'] + '_' + df['education']
    df['region_education'] = df['region'] + '_' + df['education']
    df['channel_dept'] = df['recruitment_channel'] + '_' + df['department']

    def map_department_group(dept):
        if dept in ['Sales & Marketing', 'Operations']: return 'High Performance'
        elif dept in ['Technology', 'Procurement', 'Analytics']: return 'Medium Performance'
        elif dept in ['HR', 'Finance', 'Legal', 'R&D']: return 'Low Performance'
        else: return 'Low Performance'
    df['department_group'] = df['department'].apply(map_department_group)

    region_group_map = {
        'region_2': 'High_Perf_Region', 'region_13': 'High_Perf_Region',
        'region_22': 'High_Perf_Region', 'region_7': 'High_Perf_Region',
        'region_29': 'High_Perf_Region', 'region_4': 'Medium_Perf_Region',
        'region_26': 'Medium_Perf_Region', 'region_17': 'Medium_Perf_Region',
        'region_11': 'Medium_Perf_Region', 'region_15': 'Medium_Perf_Region',
    }
    df['region_group'] = df['region'].map(region_group_map).fillna('Low_Perf_Region')

    df['KPI_miss_but_high_edu'] = ((df['KPIs_met_more_than_80']==0) & (df['education'].isin(['Bachelors','Masters']))).astype(int)
    df['High_Perf_Channel'] = (df['recruitment_channel'].isin(['other'])).astype(int)
    df['not_sales_low_potential'] = ((df['department']!='Sales & Marketing') & (df['KPIs_met_more_than_80']==0) & (df['awards_won']==0)).astype(int)
    df['doer_not_learner'] = ((df['KPIs_met_more_than_80']==1) & (df['avg_training_score']<65)).astype(int)
    df['learner_not_doer'] = ((df['avg_training_score']>64) & (df['KPIs_met_more_than_80']==0)).astype(int)
    df['high_risk_zone'] = ((df['service_years']>5) & (df['KPIs_met_more_than_80']==0)).astype(int)

    drop_cols = ['age_at_joining','service_years','awards_won','talent_signal',
                 'KPI_and_Award','High_Perf_Channel','KPI_miss_but_high_edu',
                 'training_count','employee_id','award_per_year']
    df = df.drop(columns=drop_cols, errors='ignore')

    ordinal_features = ['education', 'experience_group']
    for i, col in enumerate(ordinal_features):
        known_categories = ordinal_enc.categories_[i]
        # Xử lý unseen: gán về giá trị đầu tiên 
        df[col] = df[col].apply(lambda x: x if x in known_categories else known_categories[0]) 
        
    df[ordinal_features] = ordinal_enc.transform(df[ordinal_features])

    df = pd.get_dummies(df)
    
    final_df = pd.DataFrame(columns=final_columns)
    final_df = pd.concat([final_df, df])
    final_df = final_df.fillna(0)
    final_df = final_df[final_columns]
    
    return final_df

# =============================
#  PREDICT FUNCTION 

def predict_raw(df_raw):
    try:
        X = preprocess_raw(df_raw)
    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý dữ liệu: {e}")
        return None

    y_reg = model_reg.predict(X)
    y_cls = model_cls.predict_proba(X) @ np.array([1,2,3])
    y_ord = model_ord.predict_proba(X) @ np.array([1,2,3])
    y_blend = w_reg*y_reg + w_cls*y_cls + w_ord*y_ord
    preds = np.full_like(y_blend, 2)
    preds[y_blend < cut1] = 1
    preds[y_blend >= cut2] = 3
    labels = pd.Series(preds).map({1:'Low', 2:'Medium', 3:'High'})
    return labels

# =============================
#  STREAMLIT UI 
# --- Cấu hình trang (Page Config) ---
st.set_page_config(
    page_title="HR Performance Predictor", 
    page_icon="🚀",
    layout="wide" 
)

# --- Tiêu đề ---
st.title("HR Performance Predictor (Ensemble Model)")
st.write("Dự đoán hiệu suất nhân viên dựa trên mô hình Ensemble.")

tab1, tab2 = st.tabs(["📝 Nhập thông tin nhân viên", "📁 Tải lên file CSV"])

# --- Code cho Tab 1: Dự đoán cho 1 người
with tab1:
    st.header("Nhập thông tin nhân viên cần dự đoán:")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Thông tin cơ bản")
            age = st.number_input("Age (Tuổi)", min_value=18, max_value=70, value=30)
            gender = st.selectbox("Gender (Giới tính)", ['f', 'm'], key='manual_gender')
            education = st.selectbox("Education (Học vấn)", 
                                     ["Masters & above", "Bachelors", "Below Secondary"], key='manual_edu')
        with c2:
            st.subheader("Thông tin công việc")
            department = st.selectbox("Department (Phòng ban)", 
                                      ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 
                                       'Procurement', 'Finance', 'HR', 'Legal', 'R&D'], key='manual_dept')
            region = st.text_input("Region (Vùng)", value="region_7", key='manual_region',
                                   help="Phải gõ đúng tên region, ví dụ: region_7, region_22")
            recruitment_channel = st.selectbox("Recruitment Channel (Kênh tuyển dụng)", 
                                               ['sourcing', 'other', 'referred'], key='manual_channel')
            service_years = st.number_input("Length of Service (Thâm niên)", min_value=0, max_value=40, value=5)

    with st.container(border=True):
        st.subheader("Thông tin hiệu suất")
        c3, c4, c5 = st.columns(3)
        with c3:
            training_count = c3.number_input("No. of Trainings (Số lần training)", min_value=0, max_value=20, value=1)
        with c4:
            avg_training_score = c4.number_input("Avg Training Score (Điểm training TB)", min_value=0, max_value=100, value=75)
        with c5:
            awards_won = c5.number_input("Awards Won (Số giải thưởng)", min_value=0, max_value=10, value=0)
            KPIs_met_more_than_80 = c5.selectbox("KPIs Met > 80% (Đạt KPI)", [0, 1], key='manual_kpi')

    st.divider() 

    if st.button("Dự đoán ", type="primary"): 
        manual_data = {
            'department': [department], 'region': [region], 'education': [education],
            'gender': [gender], 'recruitment_channel': [recruitment_channel],
            'no_of_trainings': [training_count], 'age': [age],
            'length_of_service': [service_years],
            'KPIs_met_more_than_80': [KPIs_met_more_than_80],
            'awards_won': [awards_won], 'avg_training_score': [avg_training_score],
            'employee_id': ['manual_test']
        }
        df_manual = pd.DataFrame(manual_data)
        
        st.write("📋 Dữ liệu đầu vào:")
        st.dataframe(df_manual)
        
        with st.spinner("Đang dự đoán..."):
            prediction = predict_raw(df_manual)
        
        if prediction is not None:
            st.subheader("Kết quả dự đoán:")
            result = prediction.iloc[0]
            if result == "High":
                st.success(f"**Performance Dự đoán: {result}** 🚀")
            elif result == "Medium":
                st.info(f"**Performance Dự đoán: {result}** 👍")
            else:
                st.warning(f"**Performance Dự đoán: {result}** ⚠️")

# --- Code cho Tab 2: Tải file CSV ---
with tab2:
    st.header("Tải lên file CSV để dự đoán cho nhiều nhân viên")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Chọn file CSV (Kéo thả hoặc nhấn để chọn)", 
            type=["csv"],
            label_visibility="collapsed" 
        )
        
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}")
            st.stop()
            
        required_cols = ['department', 'region', 'education', 'gender', 
                         'recruitment_channel', 'no_of_trainings', 'age', 
                         'length_of_service', 'KPIs_met_more_than_80', 
                         'awards_won', 'avg_training_score', 'employee_id']
        
        if not all(col in df_input.columns for col in required_cols):
            st.error(f"LỖI: File CSV thiếu các cột bắt buộc. "
                     f"Các cột cần có: {', '.join(required_cols)}")
        else:
            st.success("Tải file thành công! Dưới đây là 5 dòng đầu tiên:")
            st.dataframe(df_input.head())
            
            st.divider()
            if st.button("Dự đoán (từ File)", type="primary"):
                with st.spinner("Đang xử lý và dự đoán... (có thể mất vài giây)"):
                    preds = predict_raw(df_input)
                
                if preds is not None:
                    st.subheader("🎯 Kết quả dự đoán:")
                    df_output = df_input[['employee_id']].copy()
                    df_output["Predicted_Performance"] = preds
                    st.dataframe(df_output)
                    
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv_output = convert_df_to_csv(df_output)
                    
                    st.download_button(
                        label="Tải kết quả CSV",
                        data=csv_output,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )