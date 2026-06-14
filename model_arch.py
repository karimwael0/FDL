from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Reshape, Input

def create_hybrid_model(input_dim):
    # Ensure input_dim is a standard Python integer
    input_dim = int(input_dim)
    
    model = Sequential([
        # 1. THE ENTRY POINT: Explicitly tell Keras the shape of one CSV row
        Input(shape=(input_dim,)),
        
        # 2. RESHAPE: Turn (features,) into (1, features) for CNN/LSTM
        Reshape((1, input_dim)),
        
        # 3. 1D CNN: Extract spatial features from the packet data
        Conv1D(filters=64, kernel_size=1, activation='relu'),
        Dropout(0.2),
        
        # 4. LSTM: Capture temporal sequences in the traffic flow
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        # 5. DENSE: Final classification layers
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # Binary Output: 0 (Benign) or 1 (Mirai)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model