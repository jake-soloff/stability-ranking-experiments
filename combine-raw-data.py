import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os


#downloaded data is in the folder named "netflix"
#parse all files in a directory and combine

popdata = []
for filename in os.listdir('netflix'):
    if filename.endswith('.txt'):
        with open(os.path.join('netflix', filename), 'r') as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    user_id, rating, date = line.split(',')
                    popdata.append((int(user_id), movie_id, int(rating)))

popdf = pd.DataFrame(popdata, columns=['user_id', 'movie_id', 'rating'])

#save the raw data
popdf.to_csv('fullData.csv', index=False)
