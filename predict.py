import numpy as np
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd

temp= pd.read_csv("sample_submission.csv")
sample= temp.values
sample=np.array(sample)[:,0]
# print(sample)
# exit()
tag_in=open("test_tag.pickle","rb")
test_tag=pickle.load(tag_in)
test_tag=test_tag.reshape(len(test_tag),1)
# print(test_tag)
# print(np.shape(test_tag))
# exit()

model=tf.keras.models.load_model("trained.model")
test_pickle_in=open("test.pickle","rb")
test_data=pickle.load(test_pickle_in)
predictions = model.predict(test_data)
# print(np.shape(predictions))
# exit()
predict=np.hstack((test_tag,predictions))
predict_sort=np.msort(predict)
# result=[]
# result=np.array(result)
# for i in sample:
#     print(predict.index(i))
#     exit()
#     result=np.append(result,predict(predict.index(i)))
np.savetxt('predicts.csv', predict, delimiter=',', fmt='%s')
np.savetxt('predicts_sort.csv', predict_sort, delimiter=',', fmt='%s')
# print(predictions)
# print(len(test_tag))
# print(np.shape(test_tag))
print(test_tag[17])

print(np.argmax(predictions[17]))

plt.imshow(test_data[17])
plt.show()