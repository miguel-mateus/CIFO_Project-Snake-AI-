window_size = 7 #Size of window around snake head
hidden_size = 15 #Size of hidden layers of the NN

#These variables have their own file and are not defined on main.py since they are both needed on main.py and ArrayFlat.py
#ArrayFlat.py is indirectly imported by main.py, so if it also imported main.py there would be errors due to "circular" imports