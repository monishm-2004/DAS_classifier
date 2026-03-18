import scipy.io

# Load one of your files
mat_data = scipy.io.loadmat('/home/monish-m/Downloads/das/70-30 Data/train/02_dig/220104_sys_dig_01_single_data_2.mat')

# Print the keys to see the variable names
print("Variables in file:", mat_data.keys())

# Replace 'data_var' with the actual variable name from the keys above
data = mat_data['data']  # Example variable name, replace with actual variable name
print("Data Shape:", data.shape)
