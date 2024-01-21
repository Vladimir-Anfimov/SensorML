import pandas as pd
import seaborn as sns

class FeatureSelection:
    NORMALIZED = './data/normalized_data.csv'
    FEATURED_SELECTED_NORMALIZED = './data/featured_selected_normalized_data.csv'

    def eliminate_high_covariance_low_variance(self, covariance_threshold=0.9, variance_threshold=0.01):
        normalized_data = pd.read_csv(self.NORMALIZED)
        correlation_matrix = normalized_data.corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

        high_covariance_columns = set()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                if abs(correlation_matrix.iloc[i, j]) > covariance_threshold:
                    print(colname_i, colname_j, abs(correlation_matrix.iloc[i, j]))
                    high_covariance_columns.add(colname_j)

        low_variance_columns = normalized_data.columns[normalized_data.var() < variance_threshold]
        columns_to_eliminate = high_covariance_columns.union(low_variance_columns)

        print(columns_to_eliminate)

        # featured_selected_normalized_data = normalized_data.drop(columns=columns_to_eliminate)
        # featured_selected_normalized_data.to_csv(self.FEATURED_SELECTED_NORMALIZED, index=False)


if __name__ == '__main__':
    FeatureSelection().eliminate_high_covariance_low_variance()