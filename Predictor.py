import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import LSTM, Dense

# Đọc dữ liệu từ tệp Excel
data = pd.read_excel('previous_data.xlsx')  # Thay 'previous_data.xlsx' bằng tên tệp của bạn
# Lấy các bộ số từ dữ liệu
numbers = data[['1st_number', '2nd_number', '3rd_number', '4th_number', '5th_number', '6th_number']].values

#Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaled_numbers = scaler.fit_transform(numbers)
#Tạo dãy số đầu vào và đầu ra
X, y = [], []
look_back = 600 # Số lượng bộ số trước đó để sử dụng
for i in range(len(scaled_numbers) - look_back):
    X.append(scaled_numbers[i:i + look_back])
    y.append(scaled_numbers[i + look_back])

X, y = np.array(X), np.array(y)
#Tạo mô hình LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, 6)))
model.add(Dense(6, activation='linear')) # 6 là số lượng số trong mỗi bộ số

#Biên dịch mô hình
model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(optimizer='adam', loss='mean_squared_error')

#Huấn luyện mô hình
model.fit(X, y, epochs=150, batch_size=32, verbose=1)#Dự đoán bộ số tiếp theo
last_sequence = scaled_numbers[-look_back:]
next_sequence = model.predict(np.array([last_sequence]))
predicted_numbers = scaler.inverse_transform(next_sequence)
predicted_numbers = np.round(predicted_numbers).astype(int)
predicted_numbers = np.clip(predicted_numbers, 1, 45)
print("Bộ số tiếp theo dự đoán:", predicted_numbers[0])
