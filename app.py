import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load data
df = pd.read_csv('airbnb_prices.csv', index_col=False)
st.title('Airbnb Data Explorer')
st.write('Welcome to the Airbnb Data Explorer. This app allows you to explore Airbnb data from major cities around the world.')
st.write('Raw Data')
st.write(df)

# Set arrays
# Neighbourhood Group Mapping
neighbourhood_groupArr = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']

# Neighbourhood Mapping
neighbourhoodArr = ['Kensington', 'Midtown', 'Clinton Hill', 'East Harlem', 'Murray Hill', 'Bedford-Stuyvesant', 'Upper West Side', 'Chinatown', "Hell's Kitchen", 'West Village', 'Williamsburg', 'Fort Greene', 'Chelsea', 'Crown Heights', 'Park Slope', 'East Village', 'Harlem', 'Greenpoint', 'Bushwick', 'Lower East Side', 'South Slope', 'Prospect-Lefferts Gardens', 'Long Island City', 'Kips Bay', 'SoHo', 'Upper East Side', 'Washington Heights', 'Flatbush', 'Brooklyn Heights', 'Prospect Heights', 'Carroll Gardens', 'Gowanus', 'Cobble Hill', 'Flushing', 'Boerum Hill', 'Sunnyside', 'St. George', 'Highbridge', 'Financial District', 'Morningside Heights', 'Jamaica', 'Ridgewood', 'NoHo', 'Flatiron District', 'Roosevelt Island', 'Greenwich Village', 'Little Italy', 'East Flatbush', 'Tompkinsville', 'Astoria', 'Eastchester', 'Inwood', 'Kingsbridge', 'Two Bridges', 'Queens Village', 'Rockaway Beach', 'Forest Hills', 'Nolita', 'Windsor Terrace', 'University Heights', 'Gramercy', 'Allerton', 'East New York', 'Theater District', 'Sheepshead Bay', 'Fort Hamilton', 'Bensonhurst', 'Tribeca', 'Shore Acres', 'Sunset Park', 'Concourse', 'DUMBO', 'Ditmars Steinway', 'Elmhurst', 'Brighton Beach', 'Cypress Hills', 'St. Albans', 'Arrochar', 'Rego Park', 'Wakefield', 'Clifton', 'Bay Ridge', 'Graniteville', 'Spuyten Duyvil', 'Stapleton', 'Briarwood', 'Ozone Park', 'Columbia St', 'Jackson Heights', 'Longwood', 'Canarsie', 'Battery Park City', 'Civic Center', 'East Elmhurst', 'New Springville', 'Morris Heights', 'Woodside', 'Gravesend', 'Tottenville', 'Mariners Harbor', 'Flatlands', 'Concord', 'Bayside', 'Downtown Brooklyn', 'Port Morris', 'Fieldston', 'Kew Gardens', 'Midwood', 'College Point', 'Mount Eden', 'Vinegar Hill', 'City Island', 'Glendale', 'Red Hook', 'Arverne', 'Maspeth', 'Port Richmond', 'Williamsbridge', 'Soundview', 'Woodhaven', 'Stuyvesant Town', 'Parkchester', 'Middle Village', 'Dyker Heights', 'Bronxdale', 'Richmond Hill', 'Sea Gate', 'Riverdale', 'Kew Gardens Hills', 'Borough Park', 'Co-op City', 'Claremont Village', 'Whitestone', 'Fordham', 'Bayswater', 'Concourse Village', 'Navy Yard', 'Mott Haven', 'Eltingville', 'Brownsville', 'Bay Terrace', 'Mount Hope', 'Clason Point', 'Norwood', 'Lighthouse Hill', 'Springfield Gardens', 'Howard Beach', 'Jamaica Estates', 'Bellerose', 'Fresh Meadows', 'Morris Park', 'Tremont', 'Corona', 'Far Rockaway', 'Great Kills', 'Manhattan Beach', 'West Brighton', 'Marble Hill', 'Dongan Hills', 'East Morrisania', 'Hunts Point', 'Neponsit', 'Randall Manor', 'Throgs Neck', 'Todt Hill', 'Silver Lake', 'Laurelton', 'Grymes Hill', 'Pelham Gardens', 'Rosedale', 'Castleton Corners', 'Pelham Bay', 'New Brighton', 'Baychester', 'Melrose', 'Cambria Heights', 'Richmondtown', 'Woodlawn', 'Howland Hook', 'Schuylerville', 'Coney Island', "Prince's Bay", 'South Beach', 'Edgemere', 'Holliswood', 'Bath Beach', 'South Ozone Park', 'Midland Beach', 'Oakwood', 'Bergen Beach', 'Douglaston', 'Belmont', 'Grant City', 'Emerson Hill', 'Westerleigh', 'Jamaica Hills', 'Little Neck', 'Westchester Square', 'Rosebank', 'Unionport', 'Mill Basin', 'Hollis', 'Edenwald', 'Morrisania', 'Van Nest', 'Arden Heights', "Bull's Head", 'Olinville', 'North Riverdale', 'Belle Harbor', 'Rossville', 'Breezy Point', 'Castle Hill', 'Willowbrook', 'New Dorp Beach', 'West Farms', 'Huguenot', 'Bay Terrace, Staten Island']

# Room Type Mapping
room_typeArr = ['Private room', 'Entire home/apt', 'Shared room']


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('**:round_pushpin: Location**')
    neighbourhood = neighbourhoodArr.index(st.selectbox('Neighbourhood', neighbourhoodArr)) + 1
    neighbourhood_group = neighbourhood_groupArr.index(st.selectbox('Neighbourhood Group', neighbourhood_groupArr)) + 1
    latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, step=0.0001)
    longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, step=0.0001)
with col2:
    st.markdown('**:information_source: Additional Information**')
    noise = st.number_input('Noise (db)', min_value=0.0, max_value=100.0, step=0.01)
    room_type = room_typeArr.index(st.selectbox('Room Type', room_typeArr)) + 1
    minimum_nights = st.number_input('Minimum Nights', min_value=0, max_value=365, step=1)    
    floor = st.number_input('Floors', min_value=0, max_value=10, step=1)    
with col3:
    st.markdown('**:star2: Reviews**')
    number_of_reviews = st.number_input('Reviews', min_value=0, max_value=1000, step=1)
    reviews_per_month = st.number_input('Reviews per Month', min_value=0.0, max_value=60.0, step=0.1)
    last_review_time = st.date_input('Last Review Time')
    last_review_year = last_review_time.year
    last_review_month = last_review_time.month
    last_review_day = last_review_time.day
    reviews_per_day = reviews_per_month / 30

# ['neighbourhood_group', 'neighbourhood', 'room_type', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'floor', 'noise(dB)', 'last_review_year', 'last_review_month', 'last_review_day', 'reviews_per_day']


input_features = pd.DataFrame({
    'neighbourhood_group': [neighbourhood_group],
    'neighbourhood': [neighbourhood],
    'room_type': [room_type],
    'minimum_nights': [minimum_nights],
    'number_of_reviews': [number_of_reviews],
    'reviews_per_month': [reviews_per_month],
    'floor': [floor],
    'noise(dB)': [noise],
    'last_review_year': [last_review_year],
    'last_review_month': [last_review_month],
    'last_review_day': [last_review_day],
    'reviews_per_day': [reviews_per_day]
})

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]

st.write('Input Data for Prediction')
st.write(input_features)

if st.button('Predict Price'):
    
    prediction = predict(input_features) * 30.7
    st.success(f'The predicted price is ${prediction:.2f}')
