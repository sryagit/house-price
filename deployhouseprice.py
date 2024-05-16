# -*- coding: utf-8 -*-

import pickle
import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np

#loading the models

delhi_model = pickle.load(open('Delhi_model.sav','rb'))
delhi_scaler = pickle.load(open("StandardScaler_Delhi.sav",'rb'))
delhi_encoder = pickle.load(open("encoder_Delhi.sav",'rb'))

mumbai_model = pickle.load(open("mumbai_model.sav",'rb'))
mumbai_scaler = pickle.load(open("StandardScaler_mumbai.sav",'rb'))
mumbai_encoder = pickle.load(open("encoder_mumbai.sav",'rb'))

chennai_model = pickle.load(open("Chennai_model.sav",'rb'))
chennai_scaler = pickle.load(open("StandardScaler_Chennai.sav",'rb'))
chennai_encoder = pickle.load(open("encoder_Chennai.sav",'rb'))

bangalore_model = pickle.load(open("Bangalore_model.sav",'rb'))
bangalore_scaler = pickle.load(open("StandardScaler_Bangalore.sav",'rb'))
bangalore_encoder = pickle.load(open("encoder_bangalore.sav",'rb'))

Hyderabad_model = pickle.load(open("Hyderabad_model.sav",'rb'))
Hyderabad_scaler = pickle.load(open("StandardScaler_Hyderabad.sav",'rb'))
Hyderabad_encoder = pickle.load(open('encoder_Hyderabad.sav','rb'))


with st.sidebar:
    selected = option_menu('House Price Prediction of Indian Metropolitan Cities',
                           [ 'Home Page',
                            'Chennai Model',
                            'Delhi Model',
                            'Mumbai Model',
                            'Bengaluru Model',
                            'Hyderabad Model'],
                           icons=['house-fill','building-fill','buildings','buildings-fill','houses-fill','buildings'],
                           default_index=0)
    
    
if (selected == 'Home Page'):
    st.title("MetropolitanHouse: Predicting Real Estate Prices in Indian Urban Centers")
    st.image('House_steamlit.jpg', use_column_width=True)
    st.write("This app predicts house prices of Indian metropolitan cities using machine learning and Python code.")
    st.write('Welcome to our House Price Prediction App! Whether you are a prospective buyer eager to find your dream home or a seller curious about the market value of your property, our innovative tool is designed just for you. Powered by cutting-edge machine learning algorithms and crafted with precision in Python, our model considers a myriad of factors, from the area and location to the number of bedrooms and a plethora of amenities including gyms, pools, and landscaped gardens. With our intuitive interface, making informed decisions about real estate has never been easier. Join us on this journey as we revolutionize the way you navigate the housing market, empowering you to buy and sell with confidence.')
    st.write("Our prediction model takes into account several features such as area, location, number of bedrooms, resale value, and various amenities.")
    st.write("These amenities include maintenance staff, gymnasium, swimming pool, landscaped gardens, jogging track, rainwater harvesting, indoor games, shopping mall, intercom, sports facility, ATM, club house, 24x7 security, power backup, car parking, staff quarter, cafeteria, multipurpose room, hospital facilities, washing machine, gas connection, air conditioning, children's play area, availability of lift, vaastu compliance, microwave, TV, dining table, sofa, refrigerator, and more.")
    
if (selected == 'Chennai Model'):
    st.title('Chennai House Price Prediction Model Using Machine Learning')
    st.image("chennai_temple.jpeg", use_column_width=True)
    # Area, Location, and Number of Bedrooms
    st.header("Basic Information")
    area = st.text_input('Area')
    location = st.selectbox('Location',['Perungalathur', 'Madhavaram', 'Karapakkam', 'Thiruvidandhai',
       'Iyappanthangal', 'Mevalurkuppam', 'Kolapakkam', 'Kundrathur',
       'Pammal', 'Puzhal', 'Selaiyur', 'Thoraipakkam OMR', 'Anna Nagar',
       'Mogappair', 'Sholinganallur', 'Medavakkam', 'Avadi',
       'Tiruvottiyur', 'Manapakkam', 'Madipakkam', 'Thiruvanmiyur',
       'Ramapuram', 'Saidapet', 'Poonamallee', 'Pallavaram',
       'Maraimalai Nagar', 'Madambakkam', 'Perungudi', 'Villivakkam',
       'Adyar', 'Navallur', 'Moolacheri', 'Chromepet', 'Nandambakkam',
       'Kelambakkam', 'Vadapalani', 'Kumananchavadi', 'Porur',
       'Periyapanicheri', 'Manikandan Nagar', 'Kodambakkam', 'Velachery',
       'East Tambaram', 'Gopalapuram', 'Sunnambu Kolathur S Kolathur',
       'Perumbakkam', 'Cholambedu', 'Urapakkam', 'Raja Annamalai Puram',
       'Besant Nagar', 'Peerakankaranai', 'Nanmangalam', 'Jamalia',
       'Guduvancheri', 'Sembakkam', 'Adambakkam', 'Nungambakkam',
       'T Nagar', 'K K Nagar', 'Ambattur', 'Valasaravakkam',
       'Kanathur Reddikuppam', 'Mugalivakkam', 'Purasaiwakkam',
       'Maduravoyal', 'Gowrivakkam', 'Mudichur', 'West Tambaram',
       'Alwarpet', 'Annanagar West', 'Thiruverkadu', 'tambaram west',
       'Guindy', 'Korattur', 'Tambaram Sanatoruim', 'Irumbuliyur',
       'Kolathur', 'Thirumullaivoyal', 'Singaperumal Koil', 'Ayapakkam',
       'Perambur', 'Chetpet', 'Kilpauk', 'Egmore', 'Alandur', 'Kovur',
       'Vandalur', 'Pozhichalur', 'Vanagaram', 'Thoraipakkam',
       'Ullagaram', 'Kovilambakkam', 'Kattupakkam', 'Thirumazhisai',
       'Kattankulathur', 'Ayanambakkam', 'Sithalapakkam', 'Vengaivasal',
       'Kilkattalai', 'Annanagar', 'Chengalpattu', 'Pallikaranai',
       'Rajakilpakkam', 'Chitlapakkam', 'Palavakkam', 'Kotturpuram',
       'Nenmeli', 'Ramavaram', 'Padi', 'NehruNagar', 'Pazavanthangal',
       'Thatchoor', 'Padur', 'Iyyappanthangal', 'Mambakkam', 'Egatoor',
       'Semmancheri', 'Virugambakkam', 'Moolakadai', 'Siruseri',
       'Velappanchavadi', 'Ekkatuthangal', 'Royapettah', 'Nandanam',
       'Vellakkal', 'Annamalai Colony', 'Thalambur', 'Nanganallur',
       'Chembarambakkam', 'Teynampet', 'Mannivakkam', 'Thaiyur',
       'Injambakkam', 'Aminjikarai', 'CIT Nagar', 'Koyambedu',
       'Kil Ayanambakkam', 'Choolaimedu'])
    num_bedrooms = st.selectbox('Number of Bedrooms', [1, 2, 3, 4, 5])
    
    # Amenities in two columns
    st.header("Amenities")
    col1, col2 = st.columns(2)
    
    # Column 1
    with col1:
        maintenance_staff = st.checkbox('Maintenance Staff')
        gymnasium = st.checkbox('Gymnasium')
        swimming_pool = st.checkbox('Swimming Pool')
        landscaped_gardens = st.checkbox('Landscaped Gardens')
        jogging_track = st.checkbox('Jogging Track')
        rainwater_harvesting = st.checkbox('Rainwater Harvesting')
        indoor_games = st.checkbox('Indoor Games')
        shopping_mall = st.checkbox('Shopping Mall')
        intercom = st.checkbox('Intercom')
        sports_facility = st.checkbox('Sports Facility')
        atm = st.checkbox('ATM')
        club_house = st.checkbox('Club House')
        dining_table = st.checkbox('Dining Table')
        sofa = st.checkbox('Sofa')
        refrigerator = st.checkbox('Refrigerator')
        tv = st.checkbox('TV')
        
        # Column 2
    with col2:
        security_24x7 = st.checkbox('24X7 Security')
        power_backup = st.checkbox('Power Backup')
        car_parking = st.checkbox('Car Parking')
        staff_quarter = st.checkbox('Staff Quarter')
        cafeteria = st.checkbox('Cafeteria')
        multipurpose_room = st.checkbox('Multipurpose Room')
        hospital = st.checkbox('Hospital')
        washing_machine = st.checkbox('Washing Machine')
        gas_connection = st.checkbox('Gas Connection')
        ac = st.checkbox('AC')
        children_play_area = st.checkbox("Children's Play Area")
        lift_available = st.checkbox('Lift Available')
        bed = st.checkbox('BED')
        vaastu_compliant = st.checkbox('Vaastu Compliant')
        microwave = st.checkbox('Microwave')
        resale = st.checkbox('Resale')
        
    price_predict = ''
    
    if st.button('Predict price'):
        
        # Transform area using scaler
        area = chennai_scaler.transform(np.array([[float(area)]]))  # Convert to float and 2D array
            
            # Transform location using encoder
        try:
            location = chennai_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array
        except:
            location = 'other'
            location = chennai_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array

        # Convert input data to array
        input_data = np.array([
            area[0][0], location, num_bedrooms, resale, maintenance_staff, gymnasium, swimming_pool, landscaped_gardens,
            jogging_track, rainwater_harvesting, indoor_games, shopping_mall, intercom, sports_facility, atm,
            club_house, security_24x7, power_backup, car_parking, staff_quarter, cafeteria, multipurpose_room,
            hospital, washing_machine, gas_connection, ac, children_play_area, lift_available, bed, vaastu_compliant,
            microwave, tv, dining_table, sofa, refrigerator
        ]).reshape(1, -1)

        # Predict price using the model
        price_predict = chennai_model.predict(input_data)
    st.success(price_predict)
         
        
    
if (selected == 'Delhi Model'):
    st.title('Delhi House Price Prediction Model Using Machine Learning')
    st.image("Inida_gate.jpg", use_column_width=True)
    # Area, Location, and Number of Bedrooms
    st.header("Basic Information")
    area = st.text_input('Area')
    location = st.selectbox('Location',['Sector 10 Dwarka', 'Uttam Nagar', 'Sarita Vihar', 'Dwarka Mor',
       'Sector 7 Dwarka', 'Sector 6 Dwarka', 'Sector 5 Dwarka',
       'Sector 23 Rohini', 'Mayur Vihar II', 'Sector 24 Rohini',
       'Sector 11 Dwarka', 'Sector 23 Dwarka', 'Sector 12 Dwarka',
       'West End', 'Sector 9 Rohini', 'Mundka', 'Sector 13 Rohini',
       'Jamia Nagar', 'Sector 19 Dwarka', 'Sector 17 Dwarka', 'Bindapur',
       'Sector-18 Dwarka', 'Vasant Kunj', 'Shastri Nagar',
       'Sector-8 Rohini', 'Sector 9 Dwarka', 'Shanti Park Dwarka',
       'Govindpuri', 'Sector 22 Dwarka', 'Matiala', 'Saket',
       'Mahavir Enclave', 'Burari', 'Shahdara', 'Babarpur', 'Khanpur',
       'Sector 13 Dwarka', 'Mansa Ram Park', 'Green Park', 'Kalkaji',
       'Sector 4 Dwarka', 'DLF Phase 5', 'Sector 3 Dwarka',
       'Chittaranjan Park', 'Chattarpur', 'Greater Kailash',
       'Sector-14 Rohini', 'Paschim Vihar', 'Pitampura',
       'Sector 18B Dwarka', 'Sector 2 Dwarka', 'Jasola',
       'Pochanpur Colony', 'Palam', 'Saidabad', 'Budh Vihar',
       'Sector 25 Rohini', 'Sector 18A Dwarka', 'Sewak Park',
       'Sector 23B Dwarka', 'Rohini sector 24', 'Sector 28 Rohini',
       'Rohini Sector 9', 'Rohini Extension', 'nawada', 'Alaknanda',
       'Sector 22 Rohini', 'Lajpat Nagar', 'South Extension 2',
       'Sector 16B Dwarka', 'Sheikh Sarai', 'Sidhartha Nagar',
       'Sector-D Vasant Kunj', 'Hauz Khas', 'Kalkaji Extension',
       'Greater kailash 1', 'Lajpat Nagar III', 'Safdarjung Enclave',
       'Greater Kailash II', 'Sainik Farms', 'Sector 20 Rohini',
       'greater kailash Enclave 1', 'DLF Farms', 'Mehrauli', 'Mahipalpur',
       'mayur vihar phase 1', 'Sarvodaya Enclave', 'Karol Bagh',
       'West Sagarpur', 'Ashok Vihar', 'Sector 21 Dwarka',
       'East of Kailash', 'Khirki Extension', 'Dashrath Puri',
       'SULTANPUR', 'Patparganj', 'Kaushambi', 'Shakurbasti',
       'Hari Nagar', 'Siri Fort', 'Katwaria Sarai', 'Mayur Vihar',
       'Nasirpur'])
    num_bedrooms = st.selectbox('Number of Bedrooms', [1, 2, 3, 4, 5])
    
    # Amenities in two columns
    st.header("Amenities")
    col1, col2 = st.columns(2)
    
    # Column 1
    with col1:
        maintenance_staff = st.checkbox('Maintenance Staff')
        gymnasium = st.checkbox('Gymnasium')
        swimming_pool = st.checkbox('Swimming Pool')
        landscaped_gardens = st.checkbox('Landscaped Gardens')
        jogging_track = st.checkbox('Jogging Track')
        rainwater_harvesting = st.checkbox('Rainwater Harvesting')
        indoor_games = st.checkbox('Indoor Games')
        shopping_mall = st.checkbox('Shopping Mall')
        intercom = st.checkbox('Intercom')
        sports_facility = st.checkbox('Sports Facility')
        atm = st.checkbox('ATM')
        club_house = st.checkbox('Club House')
        school = st.checkbox('School')
        dining_table = st.checkbox('Dining Table')
        sofa = st.checkbox('Sofa')
        refrigerator = st.checkbox('Refrigerator')
        tv = st.checkbox('TV')
        
        # Column 2
    with col2:
        security_24x7 = st.checkbox('24X7 Security')
        power_backup = st.checkbox('Power Backup')
        car_parking = st.checkbox('Car Parking')
        staff_quarter = st.checkbox('Staff Quarter')
        cafeteria = st.checkbox('Cafeteria')
        multipurpose_room = st.checkbox('Multipurpose Room')
        washing_machine = st.checkbox('Washing Machine')
        gas_connection = st.checkbox('Gas Connection')
        ac = st.checkbox('AC')
        children_play_area = st.checkbox("Children's Play Area")
        lift_available = st.checkbox('Lift Available')
        bed = st.checkbox('BED')
        vaastu_compliant = st.checkbox('Vaastu Compliant')
        microwave = st.checkbox('Microwave')
        resale = st.checkbox('Resale')
        
    price_predict = ''
    
    if st.button('Predict price'):
        
        # Transform area using scaler
        area = delhi_scaler.transform(np.array([[float(area)]]))  # Convert to float and 2D array
            
            # Transform location using encoder
        try:
            location = delhi_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array
        except:
            location = 'other'
            location = delhi_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array

        # Convert input data to array
        input_data = np.array([
            area[0][0], location, num_bedrooms, resale, maintenance_staff, gymnasium, swimming_pool, landscaped_gardens,
            jogging_track, rainwater_harvesting, indoor_games, shopping_mall, intercom, sports_facility, atm,
            club_house,school,security_24x7, power_backup, car_parking, staff_quarter, cafeteria, multipurpose_room,
            washing_machine, gas_connection, ac, children_play_area, lift_available, bed, vaastu_compliant,
            microwave, tv, dining_table, sofa, refrigerator
        ]).reshape(1, -1)

        # Predict price using the model
        price_predict = delhi_model.predict(input_data)
    st.success(price_predict)
    
if (selected == 'Mumbai Model'):
    st.title('Mumbai House Price Prediction Model Using Machine Learning')
    st.image("tajhotel.webp", use_column_width=True)
    st.header("Basic Information")
    area = st.text_input('Area')
    location = st.selectbox('Location',['Kharghar', 'Sector-13 Kharghar', 'Sector 18 Kharghar',
       'Sector 20 Kharghar', 'Sector 15 Kharghar', 'Dombivali',
       'Churchgate', 'Prabhadevi', 'Jogeshwari West', 'Kalyan East',
       'Malad East', 'Virar East', 'Virar', 'Malad West', 'Borivali East',
       'Mira Road East', 'Goregaon West', 'Kandivali West',
       'Borivali West', 'Kandivali East', 'Andheri East', 'Goregaon East',
       'Wadala', 'Ulwe', 'Dahisar', 'kandivali', 'Goregaon',
       'Bhandup West', 'thakur village kandivali east', 'Santacruz West',
       'Kanjurmarg', 'I C Colony', 'Dahisar W', 'Marol', 'Parel',
       'Lower Parel', 'Worli', 'Jogeshwari East', 'Chembur Shell Colony',
       'Central Avenue', 'Chembur East', 'Diamond Market Road', 'Mulund',
       'Nalasopara West', 'raheja vihar', 'Powai Lake', 'MHADA Colony 20',
       'Tolaram Colony', 'Taloja', 'Thane West', 'Vangani',
       'Sector 5 Ulwe', 'Sector12 New Panvel', 'Sector 17 Ulwe',
       'Sector9 Kamothe', 'Sector 19 Kharghar', 'Navi Basti',
       'Sector12 Kamothe', 'Sector 21 Kamothe', 'Rutu Enclave',
       'taloja panchanand', 'Virar West', 'Chembur', 'Sector 20 Kamothe',
       'Sector 22 Kamothe', 'Sector 18 Kamothe', 'Sector-5 Kamothe',
       'Sector-6A Kamothe', 'Sector 11 Kamothe', 'Sector-18 Ulwe',
       'Sector-12 Kamothe', 'azad nagar', 'Sindhi Society Chembur',
       'Kurla', 'Sahkar Nagar', 'Deonar', 'Thane', 'Jankalyan Nagar',
       'Badlapur', 'Ambarnath', 'Ambernath West', 'Vakola', 'Kamothe',
       'Kamothe Sector 16', 'Almeida Park', 'Khar', 'Bandra West',
       'Pali Hill', '15th Road', 'Palghar', 'Sector13 Kharghar',
       'Sector 21 Kharghar', 'Sector 12 Kharghar', 'Vivek Vidyalaya Marg',
       'Vasai east', 'Nahur', 'Badlapur West', 'Panvel', 'Kalyan',
       'Badlapur East', 'Mira Bhayandar', 'Juhu', 'Naigaon East',
       'Sector 21 Ulwe', 'Bandra East', 'Dronagiri', 'Nerul', 'Karanjade',
       'Sanpada', 'Sector-8 Ulwe', 'Sector-3 Ulwe', 'Sector 23 Ulwe',
       'ULWE SECTOR 19', 'Ghodbunder Road', 'Bhiwandi', 'Vasai',
       'Nala Sopara', 'Dadar East', 'Ghatkopar', 'Breach Candy',
       'Worli South Mumbai', 'Asangaon', 'Koparkhairane Station Road',
       'Kopar Khairane Sector 19A', 'Koper Khairane',
       'Eastern Express Highway Vikhroli', 'Magathane', 'Rawal Pada',
       'Ambernath East', 'Dokali Pada', 'Dattapada', 'Rajendra Nagar',
       'Kulupwadi', 'Samata Nagar Thakur Village', 'Mira Road and Beyond',
       'West Amardeep Colony', 'Pant Nagar', 'mumbai', 'Four Bungalows',
       'no 9', 'kolshet', 'Hiranandani Meadows', 'Kalpataru', 'Petali',
       'Kharghar Sector 34C', 'Ghatkopar East',
       'Mumbai Agra National Highway', 'vasant vihar thane west',
       'Kalyan West', 'Shirgaon', 'Pokhran 2', 'juhu tara', 'Peddar Road',
       'Palm Beach', 'Sector 10', 'Sector 19 Kamothe', 'Tilak Nagar',
       'Ghatkopar West', 'Tardeo', 'Napeansea Road', 'Mahalaxmi',
       'Dahisar West', 'Mulund West', 'Natakwala Lane', 'Link Road',
       'Devidas Cross Lane', 'Soniwadi Road', 'Haridas Nagar', 'Shimpoli',
       'TPS Road', 'Off Shimpoli road', 'Rustomjee Global City',
       'Sunil Nagar', 'Sector 30 Kharghar', 'Sector 12 A', 'Sector 18',
       'Sector13 Khanda Colony', 'Sector16 Airoli', 'Ranjanpada',
       'Sector 15', 'Sector 35G', 'Sector 5', 'Sector 35I Kharghar',
       'Sector35D Kharghar', 'Sector34 A Kharghar', 'Sector 30',
       'Sector 36 Kharghar', 'Sector 11 Belapur', 'Sector-34B Kharghar',
       'Dombivali East', 'Roadpali', 'Sector-50 Seawoods',
       'Mumbai Highway', 'Sector 7 Kharghar', 'Lokhandwala Township',
       'Andheri', 'Andheri West', 'Shastri Nagar', 'Wadala East Wadala',
       'Kalwa', 'PARSIK NAGAR', 'Maharashtra Nagar', 'Patlipada',
       'Belapur', 'Seawoods', 'Majiwada', '4 Bunglows', 'Airoli',
       'Kolshet Road', 'Sector 10 Khanda Colony', 'Pokharan Road',
       'Kharegaon', 'Panch Pakhadi', 'Sector 36 Kamothe',
       'Dombivli (West)', 'DN Nagar Road', 'Godrej Hill', 'Ganesh Nagar',
       'Haware City', 'Mahatma Gandhi Road', 'Akurli Nagar',
       'Kasar vadavali', 'Vasai West', 'Mumbai Nashik Expressway',
       'Katrap', 'Mira Road', 'Kasheli',
       'Western Express Highway Kandivali East', 'Vasind', 'KASHELI',
       'Thakurli', 'Shakti Nagar', 'Bhayandar East', 'Dahisar East',
       'ulhasnagar 4', 'Sector-26 Taloja', 'Koproli'])
    num_bedrooms = st.selectbox('Number of Bedrooms', [1, 2, 3, 4, 5])
    
    # Amenities in two columns
    st.header("Amenities")
    col1, col2 = st.columns(2)
    
    # Column 1
    with col1:
        maintenance_staff = st.checkbox('Maintenance Staff')
        gymnasium = st.checkbox('Gymnasium')
        swimming_pool = st.checkbox('Swimming Pool')
        landscaped_gardens = st.checkbox('Landscaped Gardens')
        jogging_track = st.checkbox('Jogging Track')
        rainwater_harvesting = st.checkbox('Rainwater Harvesting')
        indoor_games = st.checkbox('Indoor Games')
        shopping_mall = st.checkbox('Shopping Mall')
        intercom = st.checkbox('Intercom')
        sports_facility = st.checkbox('Sports Facility')
        atm = st.checkbox('ATM')
        club_house = st.checkbox('Club House')
        school = st.checkbox('School')
        dining_table = st.checkbox('Dining Table')
        sofa = st.checkbox('Sofa')
        refrigerator = st.checkbox('Refrigerator')
        tv = st.checkbox('TV')
        
        # Column 2
    with col2:
        power_backup = st.checkbox('Power Backup')
        staff_quarter = st.checkbox('Staff Quarter')
        cafeteria = st.checkbox('Cafeteria')
        multipurpose_room = st.checkbox('Multipurpose Room')
        washing_machine = st.checkbox('Washing Machine')
        gas_connection = st.checkbox('Gas Connection')
        ac = st.checkbox('AC')
        wifi = st.checkbox('Wifi')
        children_play_area = st.checkbox("Children's Play Area")
        lift_available = st.checkbox('Lift Available')
        bed = st.checkbox('BED')
        vaastu_compliant = st.checkbox('Vaastu Compliant')
        microwave = st.checkbox('Microwave')
        resale = st.checkbox('Resale')
        golfcourse = st.checkbox('Golfcourse')
        wardrobe = st.checkbox('Wardrobe')
        hospital = st.checkbox('Hospital')
        
    price_predict = ''
    
    if st.button('Predict price'):
        
        # Transform area using scaler
        area = mumbai_scaler.transform(np.array([[float(area)]]))  # Convert to float and 2D array
            
            # Transform location using encoder
        try:
            location = mumbai_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array
        except:
            location = 'other'
            location = mumbai_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array

        # Convert input data to array
        input_data = np.array([
            area[0][0], location, num_bedrooms, resale, maintenance_staff, gymnasium, swimming_pool, landscaped_gardens,
            jogging_track, rainwater_harvesting, indoor_games, shopping_mall, intercom, sports_facility, atm,
            club_house,school, power_backup, staff_quarter, cafeteria, multipurpose_room,hospital,
            washing_machine, gas_connection, ac,wifi,children_play_area, lift_available, bed, vaastu_compliant,
            microwave,golfcourse,tv, dining_table, sofa,wardrobe,refrigerator
        ]).reshape(1, -1)

        # Predict price using the model
        price_predict = mumbai_model.predict(input_data)
    st.success(price_predict)
    
if (selected == 'Bengaluru Model'):
    st.title('Bengaluru House Price Prediction Model Using Machine Learning')
    st.image("bengaluru.jpg", use_column_width=True)

    st.header("Basic Information")
    area = st.text_input('Area')
    location = st.selectbox('Location',['JP Nagar Phase 1', 'Dasarahalli on Tumkur Road',
       'Kannur on Thanisandra Main Road', 'Doddanekundi', 'Kengeri',
       'Horamavu', 'Thanisandra', 'Ramamurthy Nagar',
       'Whitefield Hope Farm Junction', 'Electronic City Phase 1',
       'Yelahanka', 'Anjanapura', 'Jalahalli', 'Kasavanahalli',
       'Bommasandra', 'Bellandur', 'RR Nagar', 'Begur', 'Hosa Road',
       'Sahakar Nagar', 'Kadugodi', 'Jakkur', 'Jigani', 'Krishnarajapura',
       'Brookefield', 'Banashankari', 'Nelamangala', 'Attibele',
       'Banaswadi', 'Kodigehalli', 'ITPL', 'Uttarahalli Hobli',
       'Chikkagubbi on Hennur Main Road', 'Varthur', 'Vidyaranyapura',
       'Electronic City Phase 2', 'J. P. Nagar', 'K. Chudahalli',
       'Narayanaghatta', 'Anekal City', 'Sarjapur', 'Koramangala',
       'Hebbal', 'Budigere Cross', 'Bommanahalli', 'Electronics City',
       'Chikkalasandra', 'Kogilu', 'Nayandahalli', 'Bilekahalli',
       'Muneshwara Nagar', 'Junnasandra',
       'Narayanapura on Hennur Main Road', 'Kothanur',
       'Kadugodi Industrial Area',
       'Sarjapur Road Wipro To Railway Crossing', 'RMV Extension Stage 2',
       'Kudlu', 'Talaghattapura', 'Kumbalgodu', 'Carmelaram',
       'Uttarahalli', 'Anagalapura Near Hennur Main Road',
       'Avalahalli Off Sarjapur Road', 'R T  Nagar', 'JP Nagar Phase 7',
       'Subramanyapura', 'JP Nagar Phase 4', 'JP Nagar Phase 8',
       'Amruthahalli', 'Nagarbhavi', 'Chandapura', 'Marsur',
       'JP Nagar Phase 3', 'JP Nagar Phase 9', 'Gottigere',
       'Kanakapura Road Beyond Nice Ring Road', 'Harlur', 'Konanakunte',
       'Richmond Town', 'Jayanagar', 'Domlur', 'Devanahalli', 'Hulimavu',
       'Kumaraswamy Layout', 'Bikasipura', 'Singasandra',
       'JP Nagar Phase 6', 'Sanjaynagar', 'CV Raman Nagar',
       'Padmanabhanagar', 'Hennur', 'KPC Layout', 'R.K. Hegde Nagar',
       'Kannamangala', 'Yerthiganahalli', 'Badamanavarthekaval',
       'Kanakapura', 'Bannerughatta', 'BTM Layout',
       'Kuvempu Layout on Hennur Main Road', 'Marathahalli',
       'Rajajinagar', 'Whitefield', 'RMV'])
    num_bedrooms = st.selectbox('Number of Bedrooms', [1, 2, 3, 4, 5])
    
    # Amenities in two columns
    st.header("Amenities")
    col1, col2 = st.columns(2)
    
    # Column 1
    with col1:
        gymnasium = st.checkbox('Gymnasium')
        swimming_pool = st.checkbox('Swimming Pool')
        jogging_track = st.checkbox('Jogging Track')
        rainwater_harvesting = st.checkbox('Rainwater Harvesting')
        indoor_games = st.checkbox('Indoor Games')
        shopping_mall = st.checkbox('Shopping Mall')
        intercom = st.checkbox('Intercom')
        sports_facility = st.checkbox('Sports Facility')
        atm = st.checkbox('ATM')
        club_house = st.checkbox('Club House')
        school = st.checkbox('School')
        dining_table = st.checkbox('Dining Table')
        sofa = st.checkbox('Sofa')
        refrigerator = st.checkbox('Refrigerator')
        tv = st.checkbox('TV')
        hospital = st.checkbox('Hospital')
        
        # Column 2
    with col2:
        security = st.checkbox('24X7 Security')
        power_backup = st.checkbox('Power Backup')
        carparking = st.checkbox('Carparking')
        staff_quarter = st.checkbox('Staff Quarter')
        cafeteria = st.checkbox('Cafeteria')
        washing_machine = st.checkbox('Washing Machine')
        gas_connection = st.checkbox('Gas Connection')
        ac = st.checkbox('AC')
        children_play_area = st.checkbox("Children's Play Area")
        lift_available = st.checkbox('Lift Available')
        bed = st.checkbox('BED')
        vaastu_compliant = st.checkbox('Vaastu Compliant')
        microwave = st.checkbox('Microwave')
        resale = st.checkbox('Resale')
        golfcourse = st.checkbox('Golfcourse')
        
    price_predict = ''
    
    if st.button('Predict price'):
        
        # Transform area using scaler
        area = bangalore_scaler.transform(np.array([[float(area)]]))  # Convert to float and 2D array
            
            # Transform location using encoder
        try:
            location = bangalore_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array
        except:
            location = 'other'
            location = bangalore_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array

        # Convert input data to array
        input_data = np.array([
            area[0][0], location, num_bedrooms, resale,gymnasium, swimming_pool,
            jogging_track, rainwater_harvesting, indoor_games, shopping_mall, intercom, sports_facility, atm,
            club_house,school,security, power_backup,carparking, staff_quarter, cafeteria,hospital,
            washing_machine, gas_connection, ac,children_play_area, lift_available, bed, vaastu_compliant,
            microwave,golfcourse,tv, dining_table, sofa,refrigerator
        ]).reshape(1, -1)

        # Predict price using the model
        price_predict = bangalore_model.predict(input_data)
    st.success(price_predict)
    
    
    
  
if (selected == 'Hyderabad Model'):
    st.title('Hyderabad House Price Prediction Model Using Machine Learning')
    st.image("Hyderabad.webp", use_column_width=True)
    st.header("Basic Information")
    area = st.text_input('Area')
    location = st.selectbox('Location',['Nizampet', 'Hitech City', 'Manikonda', 'Alwal', 'Kukatpally',
       'Gachibowli', 'Tellapur', 'Kokapet', 'Hyder Nagar', 'Mehdipatnam',
       'Narsingi', 'Khajaguda Nanakramguda Road', 'Madhapur',
       'Puppalaguda', 'Begumpet', 'Banjara Hills', 'AS Rao Nagar',
       'Pragathi Nagar Kukatpally', 'Miyapur', 'Mallampet',
       'Nanakramguda', 'Attapur', 'West Marredpally', 'Kompally',
       'Sri Nagar Colony', 'Hakimpet', 'Pocharam', 'Nagole', 'LB Nagar',
       'Meerpet', 'Kachiguda', 'Masab Tank', 'Kondapur', 'Saroornagar',
       'Uppal Kalan', 'Mallapur', 'Rajendra Nagar', 'Beeramguda',
       'Moosapet', 'Bachupally', 'Toli Chowki', 'Lakdikapul', 'Tarnaka',
       'Kistareddypet', 'Hafeezpet', 'Shaikpet', 'Amberpet', 'Kapra',
       'Trimalgherry', 'Habsiguda', 'Sanath Nagar', 'Darga Khaliz Khan',
       'Kothaguda', 'Balanagar', 'Jubilee Hills', 'raidurgam',
       'Murad Nagar', 'Chandanagar', 'East Marredpally', 'Aminpur',
       'Gajularamaram', 'Serilingampally', 'Malkajgiri', 'Mettuguda',
       'Venkat Nagar Colony', 'Kondakal', 'Gopanpally', 'Somajiguda',
       'Nallagandla Gachibowli', 'Krishna Reddy Pet', 'Bolarum',
       'Zamistanpur', 'Madhura Nagar', 'Ghansi Bazaar', 'Chintalakunta',
       'Chinthal Basthi', 'Nallakunta', 'Bowenpally', 'Bandlaguda Jagir',
       'Boduppal', 'Neknampur', 'Appa Junction Peerancheru',
       'Ambedkar Nagar', 'Vanasthalipuram', 'Moula Ali', 'Gandipet',
       'Nacharam', 'Appa Junction', 'Qutub Shahi Tombs', 'Abids',
       'Dilsukh Nagar', 'Quthbullapur', 'Sainikpuri', 'KTR Colony',
       'Bollaram', 'Karmanghat', 'Gajulramaram Kukatpally', 'Uppal',
       'Cherlapalli', 'Himayat Nagar', 'Rhoda Mistri Nagar', 'Chintalmet',
       'Hitex Road', 'ECIL', 'Boiguda', 'ECIL Main Road',
       'ECIL Cross Road', 'Rajbhavan Road Somajiguda',
       'Ramachandra Puram', 'TellapurOsman Nagar Road', 'Mansoorabad',
       'KRCR Colony Road', 'Pragati Nagar', 'Padmarao Nagar',
       'Paramount Colony Toli Chowki', 'BK Guda Internal Road',
       'muthangi', 'Pragathi Nagar', 'Yapral', 'Narayanguda', 'Kollur',
       'Bachupally Road', 'Old Bowenpally', 'Alapathi Nagar',
       'Arvind Nagar Colony', 'Matrusri Nagar', 'Pragathi Nagar Road',
       'Padma Colony', 'Happy Homes Colony', 'Old Nallakunta',
       'Sangeet Nagar', 'NRSA Colony', 'Adibatla', 'Methodist Colony',
       'Ameerpet', 'ALIND Employees Colony', 'Khizra Enclave', 'Medchal',
       'Dammaiguda', 'Suchitra', 'Whitefields', 'Mayuri Nagar',
       'Adda Gutta', 'Miyapur HMT Swarnapuri Colony',
       'Central Excise Colony Hyderabad', 'Basheer Bagh', 'Gopal Nagar',
       'Bachupaly Road Miyapur', 'Kushaiguda', 'Ashok Nagar',
       'Barkatpura', 'Madinaguda', 'Bagh Amberpet', 'new nallakunta',
       'BHEL', 'Sun City', 'Hydershakote', 'BK Guda Road',
       'Nallagandla Road', 'IDPL Colony', 'Ramnagar Gundu',
       'Alkapur township', 'Banjara Hills Road Number 12',
       'Panchavati Colony Manikonda', 'New Maruthi Nagar',
       'Madhavaram Nagar Colony', 'Miyapur Bachupally Road',
       'nizampet road', 'Kokapeta Village', 'HMT Hills', 'Tilak Nagar',
       'Chititra Medchal', 'Isnapur', 'D D Colony', 'DD Colony',
       'Patancheru Shankarpalli Road', 'Patancheru', 'Jhangir Pet',
       'Almasguda', 'Allwyn Colony', 'financial District',
       'Beeramguda Road', 'Pati', 'Karimnagar', 'Kollur Road',
       'Sun City Padmasri Estates', 'Chaitanyapuri', 'Nandagiri Hills',
       'Whitefield', 'Film Nagar', 'Kismatpur', 'Dr A S Rao Nagar Rd',
       'Dullapally', 'KPHB', 'Vivekananda Nagar Colony', 'Ameenpur',
       'Chintradripet', 'Ring Road', 'Saket', 'Kavuri Hills', 'manneguda',
       'Moti Nagar', 'Usman Nagar', 'Shadnagar', 'Bongloor',
       'Mailardevpally', 'Uppalguda', 'Tirumalgiri', 'Chikkadapally',
       'JNTU', 'hyderabad', 'Shamshabad', 'Srisailam Highway',
       'Domalguda', 'Lingampalli', 'Residential Flat Machavaram',
       'Whisper Valley', 'Tukkuguda Airport View Point Road',
       'Santoshnagar', 'Tolichowki', 'Domalguda Road', 'Shankarpalli',
       'Kothapet', 'Baghlingampally', 'Picket', 'Safilguda',
       'Sikh Village', 'Neredmet', 'Macha Bolarum', 'Kowkur',
       'Rakshapuram', 'west venkatapuram', 'Vidyanagar Adikmet',
       'Aushapur', 'Old Alwal', 'Secunderabad Railway Station Road',
       'Balapur', 'Hastinapur', 'chandrayangutta'])
    num_bedrooms = st.selectbox('Number of Bedrooms', [1, 2, 3, 4, 5])
    
    # Amenities in two columns
    st.header("Amenities")
    col1, col2 = st.columns(2)
    
    # Column 1
    with col1:
        maintenancestaff = st.checkbox('Maintenancestaff')
        gymnasium = st.checkbox('Gymnasium')
        swimming_pool = st.checkbox('Swimming Pool')
        landscaped_gardens = st.checkbox('landscaped Gardens')
        jogging_track = st.checkbox('Jogging Track')
        rainwater_harvesting = st.checkbox('Rainwater Harvesting')
        indoor_games = st.checkbox('Indoor Games')
        shopping_mall = st.checkbox('Shopping Mall')
        intercom = st.checkbox('Intercom')
        sports_facility = st.checkbox('Sports Facility')
        atm = st.checkbox('ATM')
        club_house = st.checkbox('Club House')
        dining_table = st.checkbox('Dining Table')
        sofa = st.checkbox('Sofa')
        wardrobe = st.checkbox('Wardrobe')
        refrigerator = st.checkbox('Refrigerator')
        tv = st.checkbox('TV')
        
        # Column 2
    with col2:
        security = st.checkbox('24X7 Security')
        power_backup = st.checkbox('Power Backup')
        carparking = st.checkbox('Carparking')
        staff_quarter = st.checkbox('Staff Quarter')
        cafeteria = st.checkbox('Cafeteria')
        multipurpose_room = st.checkbox('Multipurpose Room')
        washing_machine = st.checkbox('Washing Machine')
        gas_connection = st.checkbox('Gas Connection')
        ac = st.checkbox('AC')
        wifi = st.checkbox('Wifi')
        children_play_area = st.checkbox("Children's Play Area")
        lift_available = st.checkbox('Lift Available')
        bed = st.checkbox('BED')
        vaastu_compliant = st.checkbox('Vaastu Compliant')
        resale = st.checkbox('Resale')
        golfcourse = st.checkbox('Golfcourse')
        
    price_predict = ''
    
    if st.button('Predict price'):
        
        # Transform area using scaler
        area = Hyderabad_scaler.transform(np.array([[float(area)]]))  # Convert to float and 2D array
            
            # Transform location using encoder
        try:
            location = Hyderabad_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array
        except:
            location = 'other'
            location = Hyderabad_encoder.transform([location])[0]  # Ensure to access the first element of the resulting array

        # Convert input data to array
        input_data = np.array([
            area[0][0], location, num_bedrooms, resale,maintenancestaff,gymnasium, swimming_pool,landscaped_gardens,
            jogging_track, rainwater_harvesting, indoor_games, shopping_mall, intercom, sports_facility, atm,
            club_house,security, power_backup,carparking, staff_quarter, cafeteria,multipurpose_room,
            washing_machine, gas_connection, ac,wifi,children_play_area, lift_available, bed, vaastu_compliant,
            golfcourse,tv, dining_table, sofa,wardrobe,refrigerator
        ]).reshape(1, -1)

        # Predict price using the model
        price_predict = Hyderabad_model.predict(input_data)
    st.success(price_predict)
