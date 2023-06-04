import processdata
import clean_data
import Dcleaner

SAVE_PROCESSED_DATA_FILEPATH=r'D:\self-driving-car-data\processed-data'

processdata.processdata(clean_data.CLEANED_DATA_FILEPATH,SAVE_PROCESSED_DATA_FILEPATH)

Dcleaner.getlabelinfo(SAVE_PROCESSED_DATA_FILEPATH)