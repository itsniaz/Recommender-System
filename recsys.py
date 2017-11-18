import numpy as np
from random import randint
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from os import system
data = fetch_movielens(min_rating=5.0)



print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss='warp')

model.fit(data['train'],epochs=30,num_threads=10)

print("-"*30)

def simple_rec(model, data, user_ids):

	n_users, n_items = data['train'].shape

	for id in user_ids:

		user_rated_top = data['item_labels'][data['train'].tocsr()[id].indices][:3]

		scores = model.predict(id, np.arange(n_items))

		top_items = data['item_labels'][np.argsort(-scores)]

		print ("User No: "+str(id))
		print ('\n')
		print("User Rated :")

		for movie in user_rated_top:
			print (movie)

		print("\n")

		print ("Recommended:")

		top3 = top_items[:3]

		for each_item in top3:
			print(each_item)

		print ('-'*30)

user_ids = [randint(1,943), randint(1,943), randint(1,943)]

simple_rec(model, data, user_ids)


	
