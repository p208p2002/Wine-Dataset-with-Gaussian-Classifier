import numpy as np
from sklearn.model_selection import train_test_split
import logging
FORMAT = 'line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

def load_data(file_path):
    f = open(file_path,'r',encoding='utf-8')
    data = f.read()
    f.close()

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
        self.cov_ms = []
        self.feature_means = []
        self.class_prior_ps = []
        self.class_datas = []

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
        return np.squeeze(res),cov_m,feature_mean

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
        self.cov_ms = []
        self.feature_means = []
        for i in range(self.TOTAL_CLASS):
            class_data_xs, _ = self.class_datas[i]
            class_value, cov_m, feature_mean = self._count_class_value(class_data_xs, self.class_prior_ps[0], x)
            class_value = class_value.tolist()
            self.cov_ms.append(cov_m)
            self.feature_means.append(feature_mean)
            class_values.append(class_value)

        return np.argmin(class_values)+1,class_values

class BhattacharyyaBound():
    def __init__(self,c1_num, c2_num, c1_cov_m, c2_cov_m, c1_feature_mean, c2_feature_mean):
        self.c1_num = c1_num
        self.c2_num = c2_num
        self.cov_ms = [c1_cov_m, c2_cov_m]
        self.feature_means = [c1_feature_mean, c2_feature_mean]
    
    def compute(self):
        # part 1
        mean_err = self.feature_means[0] - self.feature_means[1]
        _res = (np.array([1/8]).dot(np.transpose(mean_err)))
        cov_m_plus = self.cov_ms[0] + self.cov_ms[1]
        part1 = _res.dot(np.linalg.inv(cov_m_plus/2)).dot(mean_err)
        # print(part1.shape)

        # part 2
        cov_ms_avg_det = np.linalg.det(np.array(self.cov_ms[0]+self.cov_ms[1])/2)
        cov_ms_det_square_root = np.power(np.linalg.det(self.cov_ms[0]) * np.linalg.det(self.cov_ms[1]),0.5)
        part2 = 0.5*np.log(cov_ms_avg_det/cov_ms_det_square_root)
        # print(np.squeeze(part1+part2))
        mu = np.squeeze(part1+part2)

        # pp
        c1_pp = self.c1_num/(self.c1_num+self.c2_num)
        c2_pp = 1.0 - c1_pp
        
        bb_res = np.power(np.array(c1_pp*c2_pp),0.5)*np.exp(-1*mu)
        # print(bb_res)
        # return np.squeeze(part1+part2)
        return bb_res
        
if __name__ == "__main__":
    # GaussianClassifier
    data_x,data_y = load_data('wine.data')
    TOTAL_CLASS = len(list(set(data_y))) #分?類
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.5, random_state=0)
    
    gc = GaussianClassifier()
    gc.fit(TOTAL_CLASS, x_train, y_train)
    
    TOTAL_TEST = len(x_test)
    correct = 0
    for x,y in zip(x_test,y_test):
        predict_class,class_values = gc.predict(x)
        predict_class = float(predict_class)
        if(y == predict_class):
            correct += 1
    print('acc:%f'%(correct/TOTAL_TEST))

    # test_data = np.array([12.08,1.33,2.3,23.6,70,2.2,1.59,.42,1.38,1.74,1.07,3.21,625],dtype=np.float32)
    # predict_class,class_values = gc.predict(test_data)
    # print(predict_class)
    # print(class_values)

    # BhattacharyyaBound
    c1_data_num = len(gc.class_datas[0][0])
    c2_data_num = len(gc.class_datas[1][0])
    c3_data_num = len(gc.class_datas[2][0])
    
    bb12 = BhattacharyyaBound(c1_data_num, c2_data_num, gc.cov_ms[0], gc.cov_ms[1], gc.feature_means[0], gc.feature_means[1]).compute()
    print('class 1 and class 2: %s'%bb12)

    bb13 = BhattacharyyaBound(c1_data_num, c3_data_num, gc.cov_ms[0], gc.cov_ms[2], gc.feature_means[0], gc.feature_means[2]).compute()
    print('class 1 and class 3: %s'%bb13)

    bb23 = BhattacharyyaBound(c2_data_num, c3_data_num, gc.cov_ms[1], gc.cov_ms[2], gc.feature_means[1], gc.feature_means[2]).compute()
    print('class 2 and class 3: %s'%bb23)
