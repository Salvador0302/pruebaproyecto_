import pickle
model = pickle.load(open('models/modelo_optimo_1.0_9_100_7_0.01_0.7_42.sav', 'rb'))
print(model.get_booster().feature_names)