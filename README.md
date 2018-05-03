Total 5 Scripts that are used to preprocess and build models:

USAGE:
STEP 1: Download the full Adressa News Dataset at http://reclab.idi.ntnu.no/dataset/three_month.tar.gz, unzip the tar.gz file


STEP 2: have all files of the three month in the same folder as the python scripts


STEP 3: Run the following commands to retrieve the articles and user-article interactions
     python3 getArticle.py 
     python3 getUserEvent.py

This will generate the following files:
     articles_unfiltered.json contains information regarding all articles found in the dataset
     articles_filtered.json: contains information only on articles that were not filtered out
     eventUser_url_only.json: contains information on all subscribing user's news interactions.


STEP 4: To subset data, run the following commands:
    # this subsets user news interaction to only include ones that have activeTime higher than 0	
    python3 misc.py eventUser_url_only.json activeTime=0
    # this subsets the output of the above command to have only user interaction regarding articles present in articles_filtered
    python3 misc.py activeTime0_eventUser_url_only.json article=articles_filtered.json
    # Further subsetting the above output to include only users that have over 500 or 1000 interactions
    python3 misc.py articles_filtered_activeTime0_eventUser_url_only.json num=500
    python3 misc.py articles_filtered_activeTime0_eventUser_url_only.json num=1000 

The final outputed files will includes the ones shown above plus:
    num500_articles_filtered_activeTime0_eventUser_url_only.json
    num1000_articles_filtered_activeTime0_eventUser_url_only.json


STEP 5: Produce training and testing data, run the following commands (example on the num500_articles_filtered_activeTime0_eventUser_url_only.json):
    python3 preprocessingMisc.py articles_filtered.json num500_articles_filtered_activeTime0_eventUser_url_only.json 

This will generate the following files:
d30_Xfixed.npy, d30_Tfixed.npy, d30_Xcoldfixed.npy, d30_Tcoldfixed.npy: These are the fixed sequence data, a sequence of 30 interactions was used to predict the 31st interaction(article)
time_Xbatch.npy, time_Tbatch.npy, time_Xcold.npy, time_Tcold.npy: These are the time sequence data, all interactions but the last within 3 hours was used to predict the final interaction(article) 

files that have the keyword 'cold' in the filename are cold start data, interactions with articles in the cold data are not seen in the training data
The default sequence length is 30, and default time period is 3 hours, all can be changed by modifying the codes.


STEP 6: Hyper parameter tuning using Hyperas
RUN following commands:
python3 hyperTuneLSTM.py d30_Xfixed.npy d30_Tfixed.npy d30_Xcoldfixed.npy d30_Tcoldfixed.npy fixed
python3 hyperTuneLSTM.py time_Xbatch.npy time_Tbatch.npy time_Xcold.npy time_Tcold.npy time 
The default parameters trains each configuration of hyperparameters 5 epochs and optimizes a total of 50 times.
A better way is to set a high epoch number and perform early stopping to make sure of model convergence (was not performed due to time concern).


STEP 7: Training best model using Hyperas best configuration
Modify the network structure in the newsLSTM.py according the best configurations given by STEP 6 output, and training model: default (30 epochs)
python3 newsLSTM.py d30_Xfixed.npy d30_Tfixed.npy d30_Xcoldfixed.npy d30_Tcoldfixed.npy fixed
python3 newsLSTM.py time_Xbatch.npy time_Tbatch.npy time_Xcold.npy time_Tcold.npy time

This will output Tensorboard logs to view in tensorboard, the weights of the best model, and the training and validation history of the best model.


STEP 8: Evaluation of model on Cold Start Data
The previous cold start test data (*cold*.npy) was used to select the best models. Once the models are trained, load the weights and perform evaluation on the models:
evalLSTM.py time time_weights.hdf5