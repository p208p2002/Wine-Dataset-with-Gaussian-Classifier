import numpy as np
from sklearn.model_selection import train_test_split
import logging
FORMAT = 'line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

def load_data(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')
    data = data[:-1]
    data = list(map(lambda d_str:d_str.split(','),data))
    data = list(map(lambda d:[float(feature) for feature in d],data))
    data_x = list(map(lambda d:d[1:],data))
    data_y = list(map(lambda d:d[0],data))

    data_x = np.asarray(data_x, dtype=np.float32)
    data_y = np.asarray(data_y, dtype=np.float32)
    return data_x,data_y

class GaussianClassifier():
    def __init__(self):
        pass

    def _transpose_1d(self, numpy_1d_array):
        # [1,2,3] -> [[1],[2],[3]]
        # [[1],[2],[3]] -> [1,2,3]
        if(len(numpy_1d_array.shape) == 1):
            return numpy_1d_array.reshape(-1, 1)
        else:
            return np.squeeze(numpy_1d_array.reshape(1, -1))

    def _count_class_value(self, class_data_xs, class_prior_p, x):
        # 計算每個類別的分數
        transpose_1d = self._transpose_1d
        #
        x = transpose_1d(x)
        class_data_xs = np.transpose(class_data_xs)
        feature_mean = transpose_1d(np.mean(class_data_xs,axis=1))

        #
        cov_m = np.cov(class_data_xs)
        inv_cov_m = np.linalg.inv(cov_m)
        var = (x - feature_mean)
        
        # shape: 1*13 -> 13*13 -> 13*1
        res = -1*np.log(class_prior_p) \
            +0.5*(transpose_1d(var).dot(inv_cov_m).dot(var)) \
            + 0.5*np.log(np.linalg.det(cov_m))
        return np.squeeze(res)

    def _get_class(self,class_num,data_x,data_y):
        class_num = float(class_num+1)
        spec_class_x = []
        spec_class_y = []
        for x,y in zip(data_x,data_y):
            if(y == class_num):
                spec_class_x.append(x)
                spec_class_y.append(y)
        return spec_class_x, spec_class_y
    
    def fit(self, TOTAL_CLASS, x_train, y_train):
        self.TOTAL_CLASS = TOTAL_CLASS
        # 取得個別class data
        class_datas = []
        for i in range(TOTAL_CLASS):
            class_datas.append(self._get_class(i,x_train,y_train))
        class_datas = np.array(class_datas)
        self.class_datas = class_datas

        # 先驗機率 Prior Probability
        class_prior_ps = []
        for class_data in class_datas:
            class_data_xs, class_data_ys = class_data
            class_prior_ps.append(len(class_data_xs)/len(x_train))
        class_prior_ps = np.array(class_prior_ps)
        self.class_prior_ps = class_prior_ps
    
    def predict(self,x):
        class_values = []
        for i in range(self.TOTAL_CLASS):
            class_data_xs, _ = self.class_datas[i]
            class_value = self._count_class_value(class_data_xs, self.class_prior_ps[0], test_data).tolist()
            class_values.append(class_value)
        return class_values

if __name__ == "__main__":
    data_x,data_y = load_data('wine.data')
    TOTAL_CLASS = len(list(set(data_y))) #分?類
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.5, random_state=1)
    
    gc = GaussianClassifier()
    gc.fit(TOTAL_CLASS, x_train, y_train)
    test_data = np.array([12.08,1.33,2.3,23.6,70,2.2,1.59,.42,1.38,1.74,1.07,3.21,625],dtype=np.float32)
    res = gc.predict(test_data)
    print(res)
        

    # # gaussianNB
    # class_data_xs, _ = class_datas[0]
    # print(gaussianNB(class_data_xs, class_prior_ps[0], test_data))

    # class_data_xs, _ = class_datas[1]
    # print(gaussianNB(class_data_xs, class_prior_ps[1], test_data))

    # class_data_xs, _ = class_datas[2]
    # print(gaussianNB(class_data_xs, class_prior_ps[2], test_data))
    # # print('res',res)

