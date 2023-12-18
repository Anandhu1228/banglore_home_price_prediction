import json
import pickle
import numpy as np

__location = None
__data_columns = None
__model = None

def load_saved_artifacts():
    print("Loading artifacts....start")
    global __data_columns
    global __location

    with open("./artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)["data_columns"]
        __location = __data_columns[3:]
    global __model
    with open("./artifacts/bangaluru_house_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)
        print("Loading artifacts....done")

load_saved_artifacts()
def get_location_names():
    return __location

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    xn = np.zeros(len(__data_columns))
    xn[0] = sqft
    xn[1] = bath
    xn[2] = bhk
    if loc_index >= 0:
        xn[loc_index] = 1

    return round(__model.predict([xn])[0])

if __name__ == "__main__":
    print(get_location_names())
