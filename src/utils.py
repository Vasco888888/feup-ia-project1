import pandas as pd
import os

def load_cities(size):
 """
 Loads cities for a given instance size.
 Expects data to be in data/{size}/cities.csv
 """
 path = os.path.join('data', str(size), 'cities.csv')
 if not os.path.exists(path):
  raise FileNotFoundError(f"Instance size {size} not found at {path}")
 
 return pd.read_csv(path)

def load_benchmarks(size):
 """
 Loads benchmark paths for a given instance size.
 Expects data to be in data/{size}/benchmarks.csv
 """
 path = os.path.join('data', str(size), 'benchmarks.csv')
 if not os.path.exists(path):
  return None
 
 return pd.read_csv(path)
