#Imports
import random
import numpy as np
import statistics
import abc

from config import *

class PermutationImportance:
    '''
        This class implements the Permutation Importance feature selection algorithm in accordance with the following article:
        André Altmann, Laura Toloşi, Oliver Sander, Thomas Lengauer, Permutation importance: a corrected feature importance measure, Bioinformatics,
         Volume 26, Issue 10, 15 May 2010, Pages 1340–1347, https://doi.org/10.1093/bioinformatics/btq134
        ---------------------------------------------------------------------------------------------------------------
        df - type: pandas DataFrame
             description: a pandas dataframe which includes all the features and the label

        label_column - type: string
                       description: label/target column name
                       default: 'target'

        task - type: string
               description: The options are: 'classification' or 'regression', depending on the prediction task
               default: 'classification'
               
        num_classes - type: int
                      description: The number of label classes (only relevant to classification tasks)
                      default: 3
                      
        classifier - type: classifier instance.
                     description: A classifier instance with feature importance method (from Sckit-Learn or similar).
                     default: XGBoost()

        iteration_number - type: int
                            description: The length of the null importance vector (equivalent to the number of times the classifier's fit model will be called)
                            default: 50

        seed - type: int
               descriprion: random seed number.
               default: 12
        '''
    def __init__(self, df, label_column=LabelColumn, task=Task, num_classes=NumClasses, classifier=Classifier, iteration_number=IterationNumber,
                seed=Seed):

        self.df = df
        self.classifier = classifier
        self.label_column = label_column
        self.task = task
        self.num_classes=num_classes
        self.iteration_number = iteration_number
        self.seed = seed

    def seperate_data_and_label(self):
        """Seperate dataframe to data and label. 
            Parameters
            ----------
            Returns
            ------
            X : DataFrame
            y : Series
            data_columns : List[String]
        """
        data_columns = [x for x in list(self.df.columns) if x != self.label_column]
        X = self.df[data_columns]
        y = self.df[self.label_column]
        return X, y, data_columns

    def get_min_and_max_values_of_prediction(self, y):
        """Get the min and max value of the target column y. 
            Parameters
            ----------
            y : Series

            Returns
            ------
            min_range_value : float
            max_range_value : float
        """
        min_range_value = min(y)
        max_range_value = max(y)
        return min_range_value, max_range_value

    def create_randomized_label(self, X, y):
        """Create random permutations of target column y. 
            Parameters
            ----------
            X : DataFrame
            y : Series

            Returns
            ------
            random_label : Series
        """
        if self.task == 'classification':
            random_label = np.random.choice(range(self.num_classes), size=X.shape[0])

        elif self.task == 'regression':
            min_range_value, max_range_value = self.get_min_and_max_values_of_prediction(y)
            random_label = random.randrange(min_range_value, max_range_value, X.shape[0])

        else:
            raise ValueError('The task must be classification or regression')

        return random_label

    def get_null_importances(self, X, y, data_columns):
        """Calculate the null importances vectors
            Parameters
            ----------
            X : DataFrame
            y : Series
            data_columns: List[String]

            Returns
            ------
            null_importence: Dictionary
        """
        null_importence = {}
        data_columns = [x for x in list(self.df.columns) if x != self.label_column]

        for i in range(self.iteration_number):
            random_label = self.create_randomized_label(X,y)
            classifier = self.classifier
            classifier.fit(X, random_label)
            feature_iportance = classifier.feature_importances_

            for j, column in enumerate(data_columns):
                null_importence.setdefault(column,[]).append(feature_iportance[j])

        return null_importence

    @staticmethod
    def get_mean_values_per_key(dic):
        """Calculate mean values for every key in dictionary
            Parameters
            ----------
            dic : Dictionary
            
            Returns
            ------
            dic : Dictionary
        """
        for key in list(dic.keys()):
            dic[key] = statistics.mean(dic[key])

        return dic

    def get_importance_according_to_real_label(self, X, y, data_columns):
        """Calculate the importances values for each feature according to the real target.
            Parameters
            ----------
            X : DataFrame
            y : Series
            data_columns: List[String]
            
            Returns
            ------
            mean_importance_according_to_real_label : Dictionary
        """
        actual_importance = {}

        for i in range(self.iteration_number):
            classifier = self.classifier
            classifier.fit(X, y)
            feature_iportance = classifier.feature_importances_

            for j, column in enumerate(data_columns):
                actual_importance.setdefault(column,[]).append(feature_iportance[j])

        mean_importance_according_to_real_label = self.get_mean_values_per_key(actual_importance)

        return mean_importance_according_to_real_label

    @abc.abstractmethod
    def feature_selection(self, null_importence, actual_importance, null_importance_mean_and_std):
        pass

    @abc.abstractmethod
    def pipline(self):
        pass