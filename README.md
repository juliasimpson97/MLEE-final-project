# Final Project: Prediction of Ocean pCO2 using Various Machine Learning Algorithms
## Machine Learning for Environmental Engineering
Completed by Julia Simpson

Fall 2022

### <span style="color:blue">*Import Data:*</span> 
#### *In Terminal*
> cd /home/julias/MLEE-final-project/raw_data/

> wget -O cesm009.tar.gz https://ndownloader.figshare.com/files/16129532

> tar -xf cesm009.tar.gz

> mv -f /home/julias/MLEE-final-project/raw_data/member_009/* /home/julias/MLEE-final-project/raw_data/

> rm cesm009.tar.gz

> rmdir member_009

- Note that data files exceed Github file limitation and therefore are ignored in pushes

### <span style="color:blue">*Goal:*</span>
#### *Predict pCO2 using sea surface salinity (SSS), sea surface temperature (SST), chorophyll-a, and mixed layer depth (MLD).* 
- Use SOCAT observations as the "true" set of values for training to create a benchmark comparison using random forest (RF), eXtreme Gradient Boosting (XGB), and neural network (NN). 
- Separately, train another NN and a convolutional long short-term memory (ConvLSTM) network on Community Earth System Model data, then use transfer learning on MPI/SOCAT data to fine-tune hyperparameters.* 

### <span style="color:blue">*Methods:*</span> 
#### *Utilize and compare the following machine learning algorithms:*
##### Benchmark Algorithms
- Random forest (RF); 
- eXtreme Gradient Boosting (XGB); and
- Neural network (NN).
##### Transfer Learning Tests
- Additional neural network, maybe with same architecture as first; and
- Convolution long short-term memory (ConvLSTM) network. 
    *(Note that LSTM may not work well if data is not sequential enough)*

#### *Following the training (which will be done on SOCAT and, where transfer learning is used, CESM), testing of algorithms will be conducted on non-SOCAT track data points.*

### <span style="color:blue">*Using Machine Learning Results:*</span> 
#### *Compare approaches*
- Do all algorithms lead to reasonable-looking pCO2 predictions?
- Does the use of transfer learning lead to significant improvement in pCO2 predictions?
- How do the two neural networks compare? Use as a window into the impact of the transfer learning.
- How does the ConvLSTM compare with the other algorithms? Comment on the introduction of CESM into training and use of a convolutional algorithm to capture some spatio-temporal relationships.



