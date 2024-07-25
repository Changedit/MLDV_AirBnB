import streamlit as st
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('airbnb_prices.csv')
st.title('Airbnb Data Explorer')
st.write('Welcome to the Airbnb Data Explorer. This app allows you to explore Airbnb data from major cities around the world.')
st.write('Raw Data')
st.write(df)

# Set tuples
neighbourhoodTuple = tuple(df['neighbourhood'].unique())
neighbourhood_groupTuple = tuple(df['neighbourhood_group'].unique())
room_typeTuple = tuple(df['room_type'].unique())

col1, col2, col3 = st.columns(3)

with col1:
    neighbourhood = st.selectbox('Neighbourhood', neighbourhoodTuple)
    neighbourhood_group = st.selectbox('Neighbourhood Group', neighbourhood_groupTuple)
with col2:
    reviews = st.number_input('Reviews', min_value=0, max_value=1000, step=1)
    noise = st.number_input('Noise (db)', min_value=0, max_value=100, step=1)
with col3:
    room_type = st.selectbox('Room Type', room_typeTuple)
    minimum_nights = st.number_input('Minimum Nights', min_value=0, max_value=365, step=1)
floor = st.number_input('Floors', min_value=0, max_value=10, step=1)    
    
input_features = pd.DataFrame({
    'neighbourhood': [neighbourhood],
    'neighbourhood_group': [neighbourhood_group],
    'room_type': [room_type],
    'number_of_reviews': [reviews],
    'noise': [noise],
    'minimum_nights': [minimum_nights],
    'floor': [floor]
})

st.write('Input Features')
st.write(input_features)

# Add placeholder columns for the output
output_columns = [
    'noise' , 'floor', 'minimum_nights', 'number_of_reviews'
]

# Adding categorical variables
neighbourhood_groups = [
    'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'
]

neighbourhoods = [
    'Arden Heights', 'Arrochar', 'Arverne', 'Astoria', 'Bath Beach', 'Battery Park City', 
    'Bay Ridge', 'Bay Terrace', 'Bay Terrace, Staten Island', 'Baychester', 'Bayside', 
    'Bayswater', 'Bedford-Stuyvesant', 'Belle Harbor', 'Bellerose', 'Belmont', 
    'Bensonhurst', 'Bergen Beach', 'Boerum Hill', 'Borough Park', 'Breezy Point', 
    'Briarwood', 'Brighton Beach', 'Bronxdale', 'Brooklyn Heights', 'Brownsville', 
    'Bull\'s Head', 'Bushwick', 'Cambria Heights', 'Canarsie', 'Carroll Gardens', 
    'Castle Hill', 'Castleton Corners', 'Chelsea', 'Chinatown', 'City Island', 'Civic Center', 
    'Claremont Village', 'Clason Point', 'Clifton', 'Clinton Hill', 'Co-op City', 'Cobble Hill', 
    'College Point', 'Columbia St', 'Concord', 'Concourse', 'Concourse Village', 'Coney Island', 
    'Corona', 'Crown Heights', 'Cypress Hills', 'DUMBO', 'Ditmars Steinway', 'Dongan Hills', 
    'Douglaston', 'Downtown Brooklyn', 'Dyker Heights', 'East Elmhurst', 'East Flatbush', 
    'East Harlem', 'East Morrisania', 'East New York', 'East Village', 'Eastchester', 'Edenwald', 
    'Edgemere', 'Elmhurst', 'Eltingville', 'Emerson Hill', 'Far Rockaway', 'Fieldston', 
    'Financial District', 'Flatbush', 'Flatiron District', 'Flatlands', 'Flushing', 'Fordham', 
    'Forest Hills', 'Fort Greene', 'Fort Hamilton', 'Fresh Meadows', 'Glendale', 'Gowanus', 
    'Gramercy', 'Graniteville', 'Grant City', 'Gravesend', 'Great Kills', 'Greenpoint', 
    'Greenwich Village', 'Grymes Hill', 'Harlem', 'Hell\'s Kitchen', 'Highbridge', 'Hollis', 
    'Holliswood', 'Howard Beach', 'Howland Hook', 'Huguenot', 'Hunts Point', 'Inwood', 
    'Jackson Heights', 'Jamaica', 'Jamaica Estates', 'Jamaica Hills', 'Kensington', 
    'Kew Gardens', 'Kew Gardens Hills', 'Kingsbridge', 'Kips Bay', 'Laurelton', 'Lighthouse Hill', 
    'Little Italy', 'Little Neck', 'Long Island City', 'Longwood', 'Lower East Side', 
    'Manhattan Beach', 'Marble Hill', 'Mariners Harbor', 'Maspeth', 'Melrose', 'Middle Village', 
    'Midland Beach', 'Midtown', 'Midwood', 'Mill Basin', 'Morningside Heights', 'Morris Heights', 
    'Morris Park', 'Morrisania', 'Mott Haven', 'Mount Eden', 'Mount Hope', 'Murray Hill', 
    'Navy Yard', 'Neponsit', 'New Brighton', 'New Dorp Beach', 'New Springville', 'NoHo', 
    'Nolita', 'North Riverdale', 'Norwood', 'Oakwood', 'Olinville', 'Ozone Park', 'Park Slope', 
    'Parkchester', 'Pelham Bay', 'Pelham Gardens', 'Port Morris', 'Port Richmond', 'Prince\'s Bay', 
    'Prospect Heights', 'Prospect-Lefferts Gardens', 'Queens Village', 'Randall Manor', 
    'Red Hook', 'Rego Park', 'Richmond Hill', 'Richmondtown', 'Ridgewood', 'Riverdale', 
    'Rockaway Beach', 'Roosevelt Island', 'Rosebank', 'Rosedale', 'Rossville', 'Schuylerville', 
    'Sea Gate', 'Sheepshead Bay', 'Shore Acres', 'Silver Lake', 'SoHo', 'Soundview', 
    'South Beach', 'South Ozone Park', 'South Slope', 'Springfield Gardens', 'Spuyten Duyvil', 
    'St. Albans', 'St. George', 'Stapleton', 'Stuyvesant Town', 'Sunnyside', 'Sunset Park', 
    'Theater District', 'Throgs Neck', 'Todt Hill', 'Tompkinsville', 'Tottenville', 'Tremont', 
    'Tribeca', 'Two Bridges', 'Unionport', 'University Heights', 'Upper East Side', 
    'Upper West Side', 'Van Nest', 'Vinegar Hill', 'Wakefield', 'Washington Heights', 
    'West Brighton', 'West Farms', 'West Village', 'Westchester Square', 'Westerleigh', 
    'Whitestone', 'Williamsbridge', 'Williamsburg', 'Willowbrook', 'Windsor Terrace', 
    'Woodhaven', 'Woodlawn', 'Woodside'
]

room_types = [
    'Private room', 'Shared room'
]

# Create dummy variables
input_features = pd.get_dummies(input_features, columns=['neighbourhood_group', 'neighbourhood', 'room_type'])

# Adding columns for the output format
for col in output_columns[0]:
    input_features[col] = 0

# Add missing categorical columns with default False
for col in neighbourhood_groups + neighbourhoods + room_types:
    dummy_col = f'neighbourhood_group_{col}' if col in neighbourhood_groups else f'neighbourhood_{col}' if col in neighbourhoods else f'room_type_{col}'
    if dummy_col not in input_features.columns:
        input_features[dummy_col] = False

# Reorder the DataFrame to match the required output
input_features = input_features[output_columns + [f'neighbourhood_group_{col}' for col in neighbourhood_groups] +
                [f'neighbourhood_{col}' for col in neighbourhoods] +
                [f'room_type_{col}' for col in room_types]]

st.write('Processed Input Features')
st.write(input_features)

model = joblib.load('best_model.pkl')

def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

if st.button('Predict Price'):
    price = predict(input_features)
    st.write(f'Predicted Price: ₹{price:,.2f}')
