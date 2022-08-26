#Imports
import random
from scipy.stats import norm
from config import *
from NullImportance import *

#Test Dataset
from sklearn.datasets import load_wine

class ClassicNullPermutation(NullPermutation):
    '''
        The class inherits form the NullPermutation and implements the feature selection by under the assumtion the null importance is normally distributed
        ---------------------------------------------------------------------------------------------------------------
        df - type: pandas DataFrame
             description: a pandas dataframe which includes all the features and the label

        label_column - type: string
                       description: label/target column name

        p_value - type: float(between 0 and 1)
                  description: the p-value threshold for selecting features
                  default: 0.05

        returns: list of selected features and list of eliminated ones  
    '''
    def __init__(self, df, label_column, p_value=P_Value):
        super().__init__(df, label_column)
        self.p_value = p_value

    def non_parametric_estimation(self, null_importance):
        """A non-parametric estimation of the null importance normal distribution parameters
            Parameters
            ----------
            null_importance: Dictionary

            Returns
            ------
            null_importance_mean_and_std: Dictionary[Tupple]
        """

        null_importance_copy = null_importance.copy()
        mean_null_importance = self.get_mean_values_for_each_key(null_importance)
        std_null_importance = {}
        null_importance_mean_and_std = {}

        for key in list(null_importance_copy.keys()):
            std_null_importance[key] = np.std(null_importance_copy[key])

        mean_std = statistics.mean(list(std_null_importance.values()))

        for key in list(std_null_importance.keys()):
            std = max(mean_std, std_null_importance[key])
            mean = mean_null_importance[key]
            null_importance_mean_and_std[key] = (mean, std)

        return null_importance_mean_and_std

    def feature_selection(self, null_importance, actual_importance, null_importance_mean_and_std):
        """Feature selection with p-value statistical test
            Parameters
            ----------
            null_importance: Dictionary
            actual_importance: Dictionary
            null_importance_mean_and_std: Dictionary

            Returns
            ------
            selected_features = List[String]
            elimenated_features = List[String]
        """
        selected_features = []
        elimenated_features = []

        for feature in list(null_importance.keys()):
            mean = null_importance_mean_and_std[feature][0]
            std = null_importance_mean_and_std[feature][1]
            cdf = norm(mean, std).cdf(actual_importance[feature])

            if 1-cdf <= self.p_value:
                selected_features.append(feature)
            else:
                elimenated_features.append(feature)

        return selected_features, elimenated_features

    def pipline(self):
            """ Pipline - runs the entire process from beginning to end
                Parameters
                ---------
                Returns
                ------
                selected_features = List[String]
                elimenated_features = List[String]
            """
            random.seed(self.seed)
            X, y, data_columns = self.seperate_data_and_label()
            null_importance = self.get_null_importances(X,y,data_columns)
            actual_importance = self.get_importance_according_to_real_label(X,y,data_columns)
            null_importance_mean_and_std = self.non_parametric_estimation(null_importance)
            selected_features, elimenated_features = self.feature_selection(null_importance, actual_importance, null_importance_mean_and_std)  
                
            return selected_features, elimenated_features


if __name__ == '__main__':
    data = load_wine(as_frame=True)
    df = data['frame']
    null_permutation_instance = ClassicNullPermutation(df, 'target', 0.05)
    selected_features, elimenated_features= null_permutation_instance.pipline()
    print(f'The selected features are: {selected_features}')
    print(f'The eliminated features are: {elimenated_features}')