import Dcleaner



DIRTY_DATA_FILEPATH=r'D:\self-driving-car-data\raw-data'
CLEANED_DATA_FILEPATH=r'D:\self-driving-car-data\clean-data' # recorded data files + some cleaned data files -->> should be about more than 500 files

Dcleaner.start2(DIRTY_DATA_FILEPATH,CLEANED_DATA_FILEPATH)
