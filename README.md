# Air-Quality-Monitoring
--------------------------------

In this project, we will train a neural network based on Sentinel-5P/TROPOMI data to predict the ozone concentration of total water. The data we will be using is data received on 29 December 2018 via Sicily (Italy). Also, this data is a tagged database of Sentinel-5P/TROPOMI. 

The data we will use in this project has 2500 rows and 26 columns. 2500 rows, we can liken it to 2500 different locations across Sicily, (Italy). The 26 columns represent the following information:
* Column 1-21: Spectral luminance value for 21 wavelengths [rad_325.0_nm, rad_335.0_nm] sensitive to ozone column abundance and derived from TROPOMI Level-1B Band-3

* Column 22: Solar zenith angle [sza] derived from TROPOMI Level-1B Band-3

* Column 23: Sensor zenith angle [vza] derived from TROPOMI Level-1B Band-3

* Column 24: latitude [latitude]

* Column 25: longitude [lon]

* Column 26: Total Ozone column abundance derived from TROPOMI Level-2 Total Column Ozone product.
