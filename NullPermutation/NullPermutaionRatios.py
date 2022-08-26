import random
from config import *
from NullImportance import *

#Test Dataset
from sklearn.datasets import load_wine

class NullPermutationRatios(NullPermutation):
    '''
        The class inherits form the NullPermutation and implements the feature selection by calculating the different ratios between the actual feature importance
        and the null feature importance
        ---------------------------------------------------------------------------------------------------------------
        df - type: pandas DataFrame
             description: a pandas dataframe which includes all the features and the label

        label_column - type: string
                       description: label/target column name

        ratio - type: String
                description: the options implemented are:
                   1. 'actual/max' - The actual feature importance devide by the maximum value of null feature importance
                   2. 'actual/mean'  - The actual feature importance devide by the mean of null feature importance.
                   3. 'actual/percentile' - The actual feature importance devide by the X percentile of null feature importance (X is a number between 0 and 100)

        percentile - type: int
                     description: precentile value (a number between 0 and 100)
                     default: None   

        returns: list of selected features and list of eliminated ones  
    '''
    def __init__(self, df, label_column, ratio, percentile=None):
        super().__init__(df, label_column)
        self.ratio = ratio
        self.percentile = percentile

        #Validation
        if ratio == 'actual/percentile' and percentile is None:
            raise ValueError('Percentile values must bt between 0 and 100')

        if  ratio == 'actual/percentile' and percentile is not None:
            if percentile<0 or percentile>100:
                raise ValueError('Percentile values must bt between 0 and 100')

        if ratio not in ['actual/max', 'actual/mean','actual/percentile']:
            raise ValueError('The ratio must be one of the followings: actual/max, actual/mean or actual/percentile')   
        
    def denominator_calc(self, null_importence):
        """ Calculation of the desiered denominator of the desiered ratio
            Parameters
            ----------
            null_importence: Dictionary

            Returns
            ------
            denominator: Dictionary
        """
        denominator_dict = {}

        for feature in list(null_importence.keys()):
            if self.ratio == 'actual/max':
                denominator_dict[feature] = max(null_importence[feature])

            elif self.ratio == 'actual/mean':
                denominator_dict[feature] = np.mean(null_importence[feature])

            elif self.ratio == 'actual/percentile':
                denominator_dict[feature] = np.percentile(null_importence[feature], self.percentile)

        return denominator_dict

    def feature_selection(self, actual_importance, denominator_dict):
        """Feature selection with p-value statistical test
            Parameters
            ----------
            denominator_dict: Dictionary
            actual_importance: Dictionary

            Returns
            ------
            selected_features = List[String]
            elimenated_features = List[String]
        """
        selected_features = []
        elimenated_features = []
        for feature in list(actual_importance.keys()):
            nominator = actual_importance[feature]
            denominator = denominator_dict[feature]
            ratio = nominator/denominator

            if ratio <= 1:
                elimenated_features.append(feature)
            else:
                selected_features.append(feature)

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
            np.random.seed(self.seed)

            X, y, data_columns = self.seperate_data_and_label()
            null_importence = self.get_null_importances(X,y,data_columns)
            actual_importance = self.get_importance_according_to_real_label(X,y,data_columns)
            denominator_dict = self.denominator_calc(null_importence)
            selected_features, elimenated_features = self.feature_selection(actual_importance, denominator_dict)  
            return selected_features, elimenated_features

#Usage Example
if __name__ == '__main__':
    data = load_wine(as_frame=True)
    df = data['frame']
    null_permutation_instance = NullPermutationRatios(df, 'target' ,'actual/percentile', 75)
    selected_features, elimenated_features= null_permutation_instance.pipline()
    print(f'the selected features are: {selected_features}')
    print(f'the eliminated features are: {elimenated_features}')