import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the relative path to the data folder
# Note: Since the script is in 'scripts/', '../data/' goes up one level to 'Sentinel-X/' then into 'data/'
DATA_PATH = 'C:\\Users\\Madhura Meenatchi A\\OneDrive\\Documents\\Sentinel-X\\data\\'

# --- 1. Variable Initialization ---
# Initialize DataFrames to None to prevent NameError if loading fails
ot_df = None
it_df = None
osint_df = None

# Define the columns expected in each file (MUST match your actual CSV files)
OT_COLS = ['timestamp', 'sensor_1_temp', 'sensor_2_flow', 'drone_altitude', 'is_anomaly']
IT_COLS = ['timestamp', 'source_ip', 'protocol_type', 'service', 'flag'] # Example columns
OSINT_COLS = ['timestamp', 'keyword_phrases', 'sentiment_score'] # Example columns

# --- 2. Data Loading (Robust) ---
print("--- Starting Data Loading ---")

try:
    ot_df = pd.read_csv(DATA_PATH + 'ot_data.csv')
    ot_df = ot_df.rename(columns={col: col.lower() for col in ot_df.columns}) # Standardize column names to lowercase
    print(f"OT Data Loaded. Shape: {ot_df.shape}")
except Exception as e:
    print(f"Error loading OT data (ot_data.csv): {e}")

try:
    it_df = pd.read_csv(DATA_PATH + 'it_data.csv')
    it_df = it_df.rename(columns={col: col.lower() for col in it_df.columns})
    print(f"IT Data Loaded. Shape: {it_df.shape}")
except Exception as e:
    print(f"Error loading IT data (it_data.csv): {e}")

try:
    osint_df = pd.read_csv(DATA_PATH + 'osint_data.csv')
    osint_df = osint_df.rename(columns={col: col.lower() for col in osint_df.columns})
    print(f"OSINT Data Loaded. Shape: {osint_df.shape}")
except Exception as e:
    print(f"Error loading OSINT data (osint_data.csv): {e}")


# --- 3. Time Synchronization and Cleanup ---
print("\n--- Starting Time Synchronization ---")

# Function to standardize time columns
def process_dataframe(df):
    if df is not None and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df
    return df

ot_df = process_dataframe(ot_df)
it_df = process_dataframe(it_df)
osint_df = process_dataframe(osint_df)

if ot_df is not None: print(f"OT Time Sync Complete. New Shape: {ot_df.shape}")


# --- 4. Domain-Specific Feature Engineering ---
print("\n--- Starting Feature Engineering ---")

# 4.1 OT Data Scaling (for LSTM Autoencoder)
if ot_df is not None and not ot_df.empty:
    ot_numerical_cols = ['sensor_1_temp', 'sensor_2_flow', 'drone_altitude']

    if all(col in ot_df.columns for col in ot_numerical_cols):
        scaler_ot = MinMaxScaler()
        ot_df[ot_numerical_cols] = scaler_ot.fit_transform(ot_df[ot_numerical_cols])
        print("OT Data Scaled (MinMaxScaler applied).")
    else:
        print("Warning: OT scaling columns not found. Check data structure.")

# 4.2 IT Data Encoding (for Isolation Forest)
if it_df is not None and not it_df.empty:
    it_categorical_cols = [col for col in it_df.columns if it_df[col].dtype == 'object' and col != 'timestamp']
    
    # Perform One-Hot Encoding
    it_df_encoded = pd.get_dummies(it_df, columns=it_categorical_cols, prefix=it_categorical_cols, drop_first=True)
    
    # Update the IT DataFrame reference to the encoded version
    it_df = it_df_encoded
    
    print(f"IT Data Encoded (One-Hot Applied). New Columns: {len(it_df.columns)}")

# 4.3 OSINT Data Prep (for BERT)
if osint_df is not None and not osint_df.empty:
    print("OSINT Data Ready for BERT Tokenization (Phase 2).")


# --- 5. Saving Processed DataFrames ---
print("\n--- Saving Processed Files ---")

# Save OT
if ot_df is not None and not ot_df.empty:
    ot_df.to_csv(DATA_PATH + 'ot_processed.csv', index=False)
    print("Saved ot_processed.csv")

# Line 103 (in the saving section):
if it_df is not None and not it_df.empty:
    # Line 104 (where the error is pointing):
    # Save the encoded DataFrame
    it_df.to_csv(DATA_PATH + 'it_processed.csv', index=False)
    print("Saved it_processed.csv")