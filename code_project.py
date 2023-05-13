import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib







# Load data from CSV file
# input_data = pd.read_csv('Customers.csv')
# input_data = input_data.drop(['CustomerID'], axis=1)
# input_data.Profession.fillna('mode', inplace=True)
# input_data['Profession'] = pd.factorize(input_data['Profession'])[0]
# input_data['Gender'] = pd.factorize(input_data['Gender'])[0]

# Scale the data
scaler = MinMaxScaler()

# Load the KMeans model
loaded_model = joblib.load('kmeans_model.pkl')

st.title('Cluster Analysis')
st.markdown("<h2 style='color: blue;'>Upload CSV file</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=["csv"])



if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    input_data.Profession.fillna('mode', inplace=True)
    newdata = input_data.drop(['CustomerID'], axis = 1)
    newdata['Gender'] = pd.factorize(newdata['Gender'])[0]
    newdata['Profession'] = pd.factorize(newdata['Profession'])[0]

    newdata=newdata[['Age','Annual Income ($)','Spending Score (1-100)']]
    newdata = scaler.fit_transform(newdata)
    
   
    # Perform clustering on the input data
    cluster_label = loaded_model.predict(newdata)

    input_data['ClusterID'] = cluster_label

    st.table(input_data.style.set_properties(**{'background-color': 'lightblue',
                                       'color': 'black',
                                       'text-align': 'center'}))
    


# input_data[['CustomerID', 'ClusterID']]