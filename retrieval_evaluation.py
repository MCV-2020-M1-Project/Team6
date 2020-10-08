import ml_metrics as metrics

actual=[] #just a list of all images from the query folder - not ordered
predicted=[] #order predicted list of images for the method used on particular image
k=5 #number of k

metrics.kdd_mapk(actual=actual,predicted=predicted,k=k)

#calculates mean of all mapk values for particular method
def calculate_mean_all(mapk_values):
    mean_of_mapk=sum(mapk_values)/len(mapk_values)
    return mean_of_mapk