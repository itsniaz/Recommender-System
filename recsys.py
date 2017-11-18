import numpy as np
from random import randint
from lightfm.datasets import fetch_movielens
from lightfm import LightFM


data = fetch_movielens(min_rating=5.0) # minmium rating for movies to take into account.  The lower the rating would be the grater number of row we have to deal with.Recommendation quality will also fall.



print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss='bpr') # https://lyst.github.io/lightfm/docs/lightfm.html



#epoch : https://stackoverflow.com/questions/31155388/meaning-of-an-epoch-in-neural-networks-training
#num_threads : num_of_threads the programme will use. Tested on windows, It can't support more than one core for this programme due to OpenMp issues. 
#Tested on Linux, can use multithread.


model.fit(data['train'],epochs=30,num_threads=10,verbose=True)

print("-"*30)

def simple_rec(model, data, user_ids):

	n_users, n_items = data['train'].shape

	for id in user_ids:

		user_rated_top = data['item_labels'][data['train'].tocsr()[id].indices][:3]

		scores = model.predict(id, np.arange(n_items))

		top_items = data['item_labels'][np.argsort(-scores)] # argsort : https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

		print ("User No: "+str(id))
		print ('\n')
		print("User Rated :")

		for movie in user_rated_top: #Items that were rated greater or equal to  min rating
			print (movie)

		print("\n")

		print ("Recommended:")

		top3 = top_items[:3]

		for each_item in top3:
			print(each_item)

		print ('-'*30)

user_ids = [randint(1,943), randint(1,943), randint(1,943)] #Generating a list of three random user ids

simple_rec(model, data, user_ids)


	
