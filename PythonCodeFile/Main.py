# 1. Load Data..
dataset= pd.read_csv("Give file's directory")
print(dataset) # Should read data..

# 2. Set inpıt-output data 
X = dataset.values[:, 0:23]
y = dataset.values[:, 25:26]

print("X-Shape :",X.shape)
print("Y-Shape :",y.shape

# 3. Splitting data into Train and Test
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, 
                                                                    y, 
                                                                    test_size=0.33, 
                                                                    random_state=42)

print("X-Train Shape..:",X_train.shape)
print("X-Val and Test-Shape..:",X_val_and_test.shape)
print("Y-Train Shape..:",y_train.shape)
print("Y-Val and Test-Shape..:",y_val_and_test.shape)
      
# 4. Let's set validation data
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, 
                                                y_val_and_test, 
                                                test_size=0.45, 
                                                random_state=42)

print("X-Val Shape..:",X_val.shape)
print("X-Test Shape..:",X_test.shape)
print("Y-Val Shape..:",y_val.shape)
print("Y-Test Shape..:", y_test.shape
      
# 5. Normalleştirilme İşlemleri..
input_scaler = MinMaxScaler(feature_range=(-5, 5))
output_scaler = MinMaxScaler(feature_range=(0, 1))

# 6. Normalleştirme İşlemlerini Uyguluyoruz..
input_scaler.fit(X)
X_train_scaled = input_scaler.transform(X_train)
X_val_scaled = input_scaler.transform(X_val)
X_test_scaled = input_scaler.transform(X_test)

# 7. with the output scaler we fit all the output space and then scale each splitted part of the dataset used as output for the neural network
output_scaler.fit(y)
y_train_scaled = output_scaler.transform(y_train)
y_val_scaled = output_scaler.transform(y_val)
      
      
# 8 Starting Keras Process
model_nn_for_o3 = Sequential([Dense(units=20, 
                                    activation='sigmoid', 
                                    input_shape=(23,)),
                              Dense(units=1, 
                                    activation='sigmoid')])

model_nn_for_o3.compile(optimizer='sgd',  
                        loss='mean_squared_error',  
                        metrics=['mae'])

# 9 Training (fitting) Process
start_training = time.time()

hist = model_nn_for_o3.fit(X_train_scaled, 
                           y_train_scaled, 
                           batch_size=10, 
                           epochs=300, 
                           validation_data=(X_val_scaled, y_val_scaled), 
                           verbose=1)

end_training = time.time()
print("Training time ", round((end_training - start_training), 3), "s" )

# 10 Visualization Process
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
      
# 11 Alternative Way Visualization Process
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# 12 Evaluate of Model
# Lost and Val_loss Measuring
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# 13 Mae and Val_Mae Drawing
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# 14 Using a Trained Neural Network to Predict Ozone Predictions
predict_train = model_nn_for_o3.predict(X_train_scaled)
predict_val = model_nn_for_o3.predict(X_val_scaled)
predict_test = model_nn_for_o3.predict(X_test_scaled)
print(predict_train) # We should look data..

predict_train_inverse = output_scaler.inverse_transform(predict_train)
predict_val_inverse = output_scaler.inverse_transform(predict_val)
predict_test_inverse = output_scaler.inverse_transform(predict_test)
print(predict_train_inverse) #We should look data 

# 15 Save the Model.. Optional.
model_nn_for_o3.save("Give Directory")

# 16 Mean Absolute Error Process
mae_train = mean_absolute_error(y_train, predict_train_inverse)
mae_val = mean_absolute_error(y_val, predict_val_inverse)
mae_test = mean_absolute_error(y_test, predict_test_inverse)

print(f'Train MAE: {round(mae_train, 3)} DU, Val MAE: {round(mae_val, 3)} DU, Test MAE: {round(mae_test, 3)} DU')

# 17 Mean Squared Error Process
rmse_train = mean_squared_error(y_train, predict_train_inverse, squared=False)
rmse_val = mean_squared_error(y_val, predict_val_inverse, squared=False)
rmse_test = mean_squared_error(y_test, predict_test_inverse, squared=False)

print(f'Train RMSE: {round(rmse_train, 3)} DU, Val RMSE: {round(rmse_val, 3)} DU, Test RMSE: {round(rmse_test, 3)} DU')

# 18 Pearson Correlation Coefficient
pearson_train = np.sqrt(r2_score(y_train, predict_train_inverse))
pearson_val = np.sqrt(r2_score(y_val, predict_val_inverse))
pearson_test = np.sqrt(r2_score(y_test, predict_test_inverse))

print(f'Train Pearson: {round(pearson_train, 3)}, Val Pearson: {round(pearson_val, 3)}, Test Pearson: {round(pearson_test, 3)}')

# 19 Standart Deviation
print("\n" "Evaluation of mean and standard deviation of the estimated values of ozone total column compared to the actual ones")
print(f'Ozone Total Column true mean: {round(np.mean(y_test), 3)} DU --- Ozone Total Column estimated mean: {round(float(np.mean(predict_test_inverse)), 3)} DU')
print(f'Ozone Total Column true std: {round(np.std(y_test), 3)} DU --- Ozone Total Column estimated std: {round(float(np.std(predict_test_inverse)), 3)} DU')
     
  
      

      
 
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

