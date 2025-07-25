import os, sys

#import the following functions from the other files...
from PhoneInfo import phonetokens
from PhoneInfo import GetPhoneIndex
from PhoneInfo import avephonetimes, phonepriors
from NumberModels import NumberModel
from CNNHMMrecog import CreateHMMModels
from CNNHMMrecog import CreateAndInitModelMemory
from CNNHMMrecog import HMMrec
from SimCNNdata import ZEROsimData, ONEsimData, TWOsimData, THREEsimData, FOURsimData, FIVEsimData
from SimCNNdata import SIXsimData, SEVENsimData, EIGHTsimData, NINEsimData, ONE_REAL_DATA1

import time
start_time = time.time()

#define hop time of CNN as macro:
HOPTIME = 0.032  # seconds - this is our frame rate or hop rate for the CNN

# Get the phone index
phoneindex = GetPhoneIndex(phonetokens)

# Create the HMM models
numbermodels = CreateHMMModels(NumberModel, avephonetimes, phonepriors, HOPTIME)

# Initialize the model memory...make sure to do this for every word that you want to recognize. 
modelmem = CreateAndInitModelMemory(numbermodels)


#Insert Test Data from SimCNN
#phoneprobmatrixfromcnn = ZEROsimData
#phoneprobmatrixfromcnn = ONEsimData
#phoneprobmatrixfromcnn = TWOsimData
#phoneprobmatrixfromcnn = THREEsimData
#phoneprobmatrixfromcnn = FOURsimData
#phoneprobmatrixfromcnn = FIVEsimData
#phoneprobmatrixfromcnn = SIXsimData
#phoneprobmatrixfromcnn = SEVENsimData
#phoneprobmatrixfromcnn = EIGHTsimData
#phoneprobmatrixfromcnn = NINEsimData
phoneprobmatrixfromcnn = ONE_REAL_DATA1

#initialize ID and decision:
ID = 'No Command' 
decision = -1
#ID not found => decision = -1
#ID is the Digit Said

# loop through all of the cnn vectors
for phoneprobframe in phoneprobmatrixfromcnn:
    decision, ID = HMMrec(numbermodels, modelmem, phoneprobframe, phoneindex)
    if decision != 0:
        break

if decision <= 0:  #either command not found or ran out of frames 
    print('Number not found\n')
else : # a command was found (decision = 1)
    print('Command Recognized: ' + ID)
#ran out of frames (couldn't make a decision in given frames) => decision = 0
#made decision => decision = 1

end = time.time()
print("--- %s seconds ---" % (end - start_time))

