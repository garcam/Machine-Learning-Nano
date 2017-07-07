""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    x_max=max(arr)
    x_min=min(arr)
    output=[None]*len(arr)
    j=0
    for i in arr:
        output[j]=float(i-x_min)/float(x_max-x_min)
        j += 1         
    return output

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)