import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import base64

def add_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/jpg;base64,{image_file});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Read and encode the image file
with open("car-insurance.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Set the background
add_background(encoded_string)


def preprocess_data(df):

    imputer = SimpleImputer(strategy='mean')
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
    
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    df_non_numeric_imputed = df_non_numeric.apply(lambda col: col.fillna(col.mode()[0]), axis=0)
    
    df_imputed = pd.concat([df_numeric_imputed, df_non_numeric_imputed], axis=1)
    

    label_encoders = {}
    for column in df_imputed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_imputed[column] = le.fit_transform(df_imputed[column])
        label_encoders[column] = le
        
    return df_imputed, label_encoders


st.title('Automobile Insurance Claim Prediction')
st.sidebar.header('Upload Training Dataset')
uploaded_train_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", key='train_file')

if uploaded_train_file is not None:
    train_data = pd.read_csv(uploaded_train_file)
    st.write("Training Dataset")
    st.write(train_data)
    
  
    target_column = st.sidebar.selectbox('Select the target column', train_data.columns, key='target_column')
    
    
    feature_columns = st.sidebar.multiselect('Select feature columns', train_data.columns.drop(target_column), key='feature_columns')
    
    if feature_columns and target_column:
     
        train_data, label_encoders = preprocess_data(train_data)
        
        
        st.subheader('Correlation Heatmap')
        corr = train_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(plt)
        
      
        X = train_data[feature_columns]
        y = train_data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

        
        model_choice = st.sidebar.selectbox('Choose Model', ('','Linear Regression', 'Decision Tree', 'Random Forest'))
        
        if model_choice:
            if model_choice == 'Linear Regression':
                model = LinearRegression()
            elif model_choice == 'Decision Tree':
                model = DecisionTreeRegressor()
            else:
                model = RandomForestRegressor()
            
            model.fit(X_train, y_train)
            
           
            y_pred = model.predict(X_test)
             
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f'{model_choice} Mean Squared Error: {mse}')
            st.write(f'{model_choice} R² Score: {r2}')
    

           
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, color='blue', label='Predicted Values', alpha=0.6)
            plt.scatter(y_test, y_test, color='red', label='Actual Values', alpha=0.6)

            m, c = np.polyfit(y_test, y_pred, 1)  
            plt.plot(y_test, m*y_test + c, color='green', label='Best Fit Line')

            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.legend()
            
            st.pyplot(plt)
            if 'models' not in st.session_state:
                st.session_state['models'] = {}
            st.session_state['models'][model_choice] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
        
        if st.sidebar.button('Find Best Model'):
            best_model_name = None
            best_mse = float('inf')
            best_r2 = float('-inf')

            for name, model_info in st.session_state['models'].items():
                if model_info['r2'] > best_r2:
                    best_r2 = model_info['r2']
                    best_mse = model_info['mse']
                    best_model_name = name

            best_model_info = st.session_state['models'][best_model_name]
            st.write(f"Best Model: {best_model_name} with R² Score: {best_r2} and MSE: {best_mse}")

            # Plotting the best model
            plt.figure(figsize=(10, 6))
            plt.scatter(best_model_info['y_test'], best_model_info['y_pred'], color='blue', label='Predicted Values', alpha=0.6)
            plt.scatter(best_model_info['y_test'], best_model_info['y_test'], color='red', label='Actual Values', alpha=0.6)
            m, c = np.polyfit(best_model_info['y_test'], best_model_info['y_pred'], 1)  
            plt.plot(best_model_info['y_test'], m*best_model_info['y_test'] + c, color='green', label='Best Fit Line')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted Values for Best Model: {best_model_name}')
            plt.legend()
            st.pyplot(plt)

            
           

        st.sidebar.header('Live Data Prediction')
        live_data_input = st.session_state.get('live_data_input', {}) 
        for col in feature_columns:
            live_data_input[col] = st.sidebar.text_input(f'Enter {col}', live_data_input.get(col, ''))
        
        st.session_state['live_data_input'] = live_data_input  
        
        if st.sidebar.button('Predict Live Data'):
            live_data_df = pd.DataFrame([live_data_input])
            
            for col in feature_columns:
                if col not in live_data_df:
                    live_data_df[col] = np.nan
            
            for col, le in label_encoders.items():
                if col in live_data_df:
                    try:
                        live_data_df[col] = le.transform([live_data_df[col].iloc[0]])
                    except ValueError:
                        live_data_df[col] = 0  
            
            live_data_df = live_data_df[feature_columns]  
            live_prediction = model.predict(live_data_df)[0]  
            st.write("Live Data Prediction")
            st.write(f"Prediction Value: {live_prediction}")
            
            claim_status = "Claim Made" if live_prediction >= 200 else "No Claim"
            st.write(f"Claim Status: {claim_status}")
            
            plt.figure(figsize=(10, 6))
            prediction_color = 'green' if live_prediction >= 200 else 'red'
            
            plt.bar(['Live Data Prediction'], [live_prediction], color=prediction_color)
            plt.xlabel('Prediction')
            plt.ylabel('Probability')
            plt.title(f'Live Data Prediction: {claim_status}')
            plt.ylim(0, 1)
            st.pyplot(plt)