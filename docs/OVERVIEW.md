## Preprocessing

CHMer requires panchromatic satellite imagery, digital elevation models (DEMs), and a small sample of lidar-produced CHMs for a given site. Additionally, the model operates on geometric information extracted from the image metadata, specifically the target azimuth and off-nadir angles of the satellite and solar elevation and solar azimuth specific to the date and time of acquisition of the satellite imagery.

The satellite imagery, the 30-m digital elevation models (DEMs), and lidar-produced canopy height models (CHMs) will be tiled into 512x512 pixels (256 x 256 m) resampled to 0.5m resolution. Sensor and solar tiles, generated from angle metadata, are meant to represent planes tilted orthogonal to the sensor and solar angles. Geolocations of the satellite images are stored and later to be applied to predictions. Lastly, training, validation, and testing splits are generated 70/15/15 and organized for efficient data loading for MS-Net.


## Neural Network : MS-net

The architecture was originally introduced by @Santos:2021 for the prediction of flow fields in porous media. For a comprehensive explanation, readers are encouraged to consult that work. MS-net is a collection of convolutional neural networks that operates on copies of the input image that have been coarsened to various levels or scales, which reconstruct to generate a final prediction. There are three fully convolutional networks with 15 layers, operating at different scales. Each scale coarsened by a factor of 4. Scale 0, with 8 filters, captures the full domain size while scale 2, with 128 filters, operates on the coarsest scale. The loss function used for MS-net accounts for the prediction error at each scale. Each scale's contribution is calculated as the mean squared error between the predicted field and the corresponding coarsened true field, normalized by the variance of the true field. The overall loss function is then expressed as a weighted sum of these scale-specific errors [@Marcato:2023; @Santos: 2021]. Other modifications to the original architecture include replacing the 3D convolutional kernels with 2D kernels. A ReLU activation function replaced the CELU function to avoid negative estimates of canopy height [@Abolt:2025].


## Postprocessing

Pixelwise and treewise comparisons are used to evaluate model performance. With pixelwise comparison, each predicted pixel is compared to the corresponding target pixels, but this method suffers from exploding losses and is difficult to interpret. This is largely due to the fact that tree crowns are not stable across images and time (satellite images and LiDAR-produced CHMs); slight spatial shifts in tree positions and variability in crown shapes between predictions and targets can lead to large pixelwise errors.

To address this, treewise comparisons are used as a more interpretable approach. These comparisons rely on a watershed segmentation method to delineate Tree Approximate Objects (TAOs) from the CHM [@Jeronimo:2018]. For each TAO, the highest point is identified, and the predicted and target heights are compared using Mean Absolute Error (MAE). This provides more useful metrics for understanding and refining predictions, particularly for end-users such as wildland-fire modelers and ecologists [@Marcozzi:2025], who are more concerned with tree-level structure than with exact pixel matches.

